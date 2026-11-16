import json
import math
import os
import re
from bisect import bisect_left


def camel_keys_to_snake(d: dict, recursive=True):
    """Convert HLS config keys to snake_case and LayerType/LayerName identifiers to lowercase."""

    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        new_key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", key)
        new_key = new_key.lower()  # acronyms stay whole: IOType -> io_type
        if not (recursive and isinstance(value, dict)):
            new_dict[new_key] = value
        elif new_key in ["layer_type", "layer_name"]:
            new_dict[new_key] = {  # hls4ml matches layers case-insensitively
                name.lower(): camel_keys_to_snake(config)
                for name, config in value.items()
            }
        else:
            new_dict[new_key] = camel_keys_to_snake(value)

    return new_dict


def to_lower_keys(d: dict, recursive=True):
    """
    Transforms dictionary keys to lower case.

    Args:
        d (dict): Targeted dictionary for keys conversion.
        recursive (bool, optional): Whether to apply to nested dictionaries. Defaults to True.

    Returns:
        dict: Transformed dictionary.
    """

    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        new_key = key.lower()
        if recursive and isinstance(value, dict):
            new_dict[new_key] = to_lower_keys(value, recursive=True)
        else:
            new_dict[new_key] = value

    return new_dict


def unwrap_nested_dicts(d, prefix=None):
    """Flatten nested dicts, joining keys with underscores."""

    if not isinstance(d, dict):
        return d

    items = []
    for key, value in d.items():
        new_key = f"{prefix}_{key}" if prefix is not None else key
        if isinstance(value, dict):
            items += unwrap_nested_dicts(value, prefix=new_key).items()
        else:
            items.append((new_key, value))

    return dict(items)


def get_reuse_factor(hls_config, layer, name=None):
    """Reuse factor hls4ml gives a layer record: by name, then type, then model."""

    layer_type = layer["class_name"].lower()
    if layer.get("activation") == "softmax":
        layer_type = "softmax"
    hls_types = {
        "qdense": "dense",
        "qconv1d": "conv1d",
        "qconv2d": "conv2d",
        "qbatchnormalization": "batchnormalization",
        "maxpooling1d": "pooling1d",
        "averagepooling1d": "pooling1d",
        "maxpooling2d": "pooling2d",
        "averagepooling2d": "pooling2d",
        "relu": "activation",
        "flatten": "reshape",
        "add": "merge",
        "inputlayer": "input",
    }
    layer_type = hls_types.get(layer_type, layer_type)

    reuse_factor = hls_config["model"]["reuse_factor"]
    type_config = hls_config.get("layer_type", {}).get(layer_type, {})
    reuse_factor = type_config.get("reuse_factor", reuse_factor)
    if name is not None:
        name_config = hls_config.get("layer_name", {}).get(name.lower(), {})
        reuse_factor = name_config.get("reuse_factor", reuse_factor)

    return reuse_factor


def get_weight_bits(hls_config, layer, name=None):
    """Weight bit width hls4ml gives a layer record, in hls4ml's precision lookup order."""

    layer_type = layer["class_name"].lower().removeprefix("q")  # QDense is a Dense
    if layer_type not in ("dense", "conv1d", "conv2d"):
        return None

    model_precision = hls_config["model"]["precision"]
    levels = [
        hls_config.get("layer_type", {}).get(layer_type, {}).get("precision"),
        model_precision,
    ]
    if name is not None:
        layer_config = hls_config.get("layer_name", {}).get(name.lower(), {})
        levels.insert(0, layer_config.get("precision"))

    for precision in levels:
        if isinstance(precision, dict):
            precision = precision.get("weight", precision.get("default"))
        if precision is not None:
            break

    if precision == "auto":  # hls4ml infers the model default
        if isinstance(model_precision, dict):
            model_precision = model_precision["default"]
        precision = model_precision

    precision = re.sub(r"\s", "", precision)
    if match := re.fullmatch(r"(?:ap_)?u?int<(\d+)>", precision, re.IGNORECASE):
        return int(match[1])

    signed = re.sub(r"^((?:ap_)?)u(?=fixed)", r"\1", precision, flags=re.IGNORECASE)

    return fixed_precision_to_bit_width(signed)[0]


def name_weighted_layers(model_config, hls_config):
    """Name a record's weighted layers after their LayerName weight precisions.

    Records drop layer names, and hls4ml lists LayerName entries in layer order.
    """

    model_config = [dict(layer) for layer in model_config]
    if any("name" in layer for layer in model_config):
        return model_config

    names = [
        name
        for name, config in (hls_config.get("layer_name") or {}).items()
        if isinstance(config.get("precision"), dict) and "weight" in config["precision"]
    ]
    if not names:  # precision set by type or model
        return model_config

    layers = [
        layer
        for layer in model_config
        if get_weight_bits(hls_config, layer) is not None
    ]
    if len(names) != len(layers):
        raise ValueError(
            f"{len(names)} LayerName weight precisions for {len(layers)} layers"
        )

    for layer, name in zip(layers, names):
        layer["name"] = name

    return model_config


def ensure_input_layer(model_config, hls_config, name="input1"):
    """Prepend the InputLayer that Keras model.layers omits for Sequential models."""

    if model_config[0]["class_name"] == "InputLayer":
        return model_config

    input_layer = {
        "class_name": "InputLayer",
        "input_shape": model_config[0]["input_shape"],
        "output_shape": model_config[0]["input_shape"],
        "parameters": 0,
        "trainable_parameters": 0,
        "dtype": model_config[0]["dtype"],
    }
    input_layer["reuse_factor"] = get_reuse_factor(hls_config, input_layer, name)

    return [input_layer, *model_config]


def get_activation_name(activation):
    match = re.match(r"quantized_(\w+)(?:\(?[^)]*\)?)", activation)
    if match:
        return match.group(1)

    return activation


def fixed_precision_to_bit_width(precision: str):
    """Return (total_bits, integer_bits) of an hls4ml ap_fixed precision string."""

    # hls4ml fixed-point grammar: width, integer bits, rounding, saturation, sat bits
    rounding = r"(?:ap_)?(?:trn|trn_zero|rnd|rnd_zero|rnd_inf|rnd_min_inf|rnd_conv)"
    saturation = r"(?:ap_)?(?:wrap|sat|sat_zero|sat_sym)"
    match = re.fullmatch(
        rf"(?:ap_)?fixed<(\d+),(-?\d+)(?:,{rounding}(?:,{saturation}(?:,\d+)?)?)?>",
        re.sub(r"\s", "", precision),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            f'Invalid precision: {precision}, expecting "ap_fixed<W,I[,Q[,O[,N]]]>"'
        )

    return int(match[1]), int(match[2])


def get_supported_boards():
    with open(
        os.path.join(os.path.dirname(__file__), "supported_boards.json")
    ) as json_file:
        supported_boards = json.load(json_file)

    return supported_boards


def get_board_from_part(part_number):
    supported_boards = get_supported_boards()

    for key, value in supported_boards.items():
        if value["part"] == part_number:
            return key

    raise ValueError(f"Part number {part_number} not supported.")


def get_part_from_board(board_name):
    supported_boards = get_supported_boards()

    if board_name not in supported_boards:
        raise ValueError(f"Board {board_name} not supported.")

    return supported_boards[board_name]["part"]


# https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/backends/fpga/fpga_backend.py#L262
def _validate_reuse_factor(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    multiplier_limit = math.ceil((n_in * n_out) / float(multfactor))
    _assert = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
    _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
    _assert = _assert and (((n_in * n_out) % rf) == 0)

    return _assert


# https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/backends/fpga/fpga_backend.py#L277
def get_closest_reuse_factor(n_in, n_out, chosen_rf):
    """
    Returns closest value to chosen_rf.
    If two numbers are equally close, return the smallest number.
    """

    max_rf = n_in * n_out
    valid_reuse_factors = []
    for rf in range(1, max_rf + 1):
        _assert = _validate_reuse_factor(n_in, n_out, rf)
        if _assert:
            valid_reuse_factors.append(rf)

    valid_rf = sorted(valid_reuse_factors)

    pos = bisect_left(valid_rf, chosen_rf)
    if pos == 0:
        return valid_rf[0]
    if pos == len(valid_rf):
        return valid_rf[-1]

    before = valid_rf[pos - 1]
    after = valid_rf[pos]
    if after - chosen_rf < chosen_rf - before:
        return after
    else:
        return before
