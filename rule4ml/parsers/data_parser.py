import json
import os
from dataclasses import dataclass, field
from glob import glob

import numpy as np
import pandas as pd

from rule4ml.parsers.utils import fixed_precision_to_bit_width, unwrap_nested_dicts

# Loading boards info
boards_path = os.path.join(os.path.dirname(__file__), "supported_boards.json")
boards_data = {}
with open(boards_path) as json_file:
    boards_data = json.load(json_file)

# Default ordinal encodings for categorical inputs
default_board_map = {key.lower(): idx + 1 for idx, key in enumerate(boards_data.keys())}
default_layer_type_map = {
    "inputlayer": 1,
    "dense": 2,
    "relu": 3,
    "sigmoid": 4,
    "tanh": 5,
    "softmax": 6,
    "batchnormalization": 7,
    "add": 8,
    "concatenate": 9,
    "dropout": 10,
}
default_precision_map = {
    'ap_fixed<2, 1>': 1,
    'ap_fixed<8, 3>': 2,
    'ap_fixed<8, 4>': 3,
    'ap_fixed<16, 6>': 4,
}
default_strategy_map = {"latency": 1, "resource": 2}


@dataclass
class ParsedDataFilter:
    """
    Helper data class for filtering parsed data.

    Args: (all optional, default values inside data class)
        min_layers (int): _description_
        max_layers (int): _description_
        min_reuse_factor (int): _description_
        max_reuse_factor (int): _description_

        min_input_size (int): _description_
        max_input_size (int): _description_

        min_output_size (int): _description_
        min_output_size (int): _description_

        min_softmax_count (int): _description_
        max_softmax_count (int): _description_
        output_softmax_only (bool): _description_

        exclude_layers (list): _description_
        strategies (list): _description_
        precisions (list): _description_
    """

    # Default settings

    # The min/max settings are ignored if <= 0
    min_layers: int = 0
    max_layers: int = 0
    min_reuse_factor: int = 0
    max_reuse_factor: int = 0

    min_input_size: int = 0
    max_input_size: int = 0

    min_output_size: int = 0
    max_output_size: int = 0

    min_softmax_count: int = 0
    max_softmax_count: int = 0
    output_softmax_only: bool = False

    # Exclude models that contains specific layers
    exclude_layers: list = field(default_factory=lambda: [])

    strategies: list = field(default_factory=lambda: ["Resource", "Latency"])

    precisions: list = field(
        default_factory=lambda: [
            "ap_fixed<2, 1>",
            "ap_fixed<8, 3>",
            "ap_fixed<8, 4>",
            "ap_fixed<16, 6>",
        ]
    )

    boards: list = field(default_factory=lambda: ["pynq-z2", "zcu102", "alveo-u200"])


def filter_match(parsed_data, data_filter: ParsedDataFilter):
    """
    _summary_

    Args:
        parsed_data (_type_): _description_
        data_filter (ParsedDataFilter): _description_

    Returns:
        _type_: _description_
    """

    n_layers = len(parsed_data["model_config"])
    if (data_filter.min_layers > 0 and n_layers < data_filter.min_layers) or (
        data_filter.max_layers > 0 and n_layers > data_filter.max_layers
    ):
        return False

    input_shape = np.asarray(parsed_data["model_config"][0]["input_shape"]).flatten()
    input_size = np.prod([x for x in input_shape if x is not None])
    if (data_filter.min_input_size > 0 and input_size < data_filter.min_input_size) or (
        data_filter.max_input_size > 0 and input_size > data_filter.max_input_size
    ):
        return False

    output_shape = np.asarray(parsed_data["model_config"][-1]["output_shape"]).flatten()
    output_size = np.prod([x for x in output_shape if x is not None])
    if (data_filter.min_output_size > 0 and output_size < data_filter.min_output_size) or (
        data_filter.max_output_size > 0 and output_size > data_filter.max_output_size
    ):
        return False

    softmax_count = 0
    for idx, layer_data in enumerate(parsed_data["model_config"]):
        reuse_factor = layer_data["reuse_factor"]
        if (data_filter.min_reuse_factor > 0 and reuse_factor < data_filter.min_reuse_factor) or (
            data_filter.max_reuse_factor > 0 and reuse_factor > data_filter.max_reuse_factor
        ):
            return False

        layer_class = layer_data["class_name"]
        if layer_class in data_filter.exclude_layers:
            return False

        if layer_class == "Activation":
            if layer_data["activation"].lower() == "softmax":
                if data_filter.output_softmax_only and idx != (n_layers - 1):
                    return False
                softmax_count += 1
                if data_filter.output_softmax_only and idx < n_layers - 1:
                    return False
            if layer_data["activation"] in data_filter.exclude_layers:
                return False

    if (data_filter.min_softmax_count > 0 and softmax_count < data_filter.min_softmax_count) or (
        data_filter.max_softmax_count > 0 and softmax_count > data_filter.max_softmax_count
    ):
        return False

    hls_config = parsed_data["hls_config"]
    strategy = hls_config["model"]["strategy"]
    if strategy not in data_filter.strategies:
        return False

    precision = hls_config["model"]["precision"]
    if precision not in data_filter.precisions:
        return False

    board = hls_config["board"]
    if board not in data_filter.boards:
        return False

    return True


def read_from_json(file_patterns, data_filter: ParsedDataFilter = None):
    """
    _summary_

    Args:
        file_patterns (_type_): _description_
        data_filter (ParsedDataFilter): _description_

    Returns:
        _type_: _description_
    """

    json_data = []
    if isinstance(file_patterns, (list, tuple)):
        file_patterns = list(file_patterns)

        json_files = glob(file_patterns[0])
        json_files = np.asarray(json_files)
        if len(file_patterns) > 1:
            for name in file_patterns[1:]:
                json_files = np.append(json_files, glob(name))
    else:
        json_files = glob(file_patterns)

    for filename in json_files:
        with open(filename) as json_file:
            json_data += json.load(json_file)

    # Optionally filter the json data
    if data_filter is not None:
        filtered_data = []
        for model_data in json_data:
            if filter_match(model_data, data_filter):
                filtered_data.append(model_data)

        json_data = filtered_data

    return json_data


def get_global_data(parsed_data):
    """
    _summary_

    Args:
        parsed_data (_type_): _description_

    Returns:
        _type_: _description_
    """

    meta = []
    global_inputs = []
    targets = []
    for model_data in parsed_data:
        model_config = model_data["model_config"]
        hls_config = model_data["hls_config"]

        meta.append(model_data["meta_data"])
        global_inputs.append(get_global_inputs(model_config, hls_config))
        targets.append(get_prediction_targets(model_data, norm_board=hls_config["board"]))

    return (meta, global_inputs, targets)


def get_sequential_data(parsed_data):
    """
    _summary_

    Args:
        parsed_data (_type_): _description_

    Returns:
        _type_: _description_
    """

    max_model_depth = 0
    for model_data in parsed_data:
        max_model_depth = max(max_model_depth, len(model_data["model_config"]))

    sequential_inputs = []
    for model_data in parsed_data:
        model_config = model_data["model_config"]
        layers_data = get_layers_data(model_config)

        # Zero padding for equal model depth across dataset
        for idx in range(max_model_depth - len(layers_data)):
            padding_dict = {}
            for key in layers_data[0].keys():
                padding_dict[key] = 0

            layers_data.append(padding_dict)

        sequential_inputs.append(layers_data)

    return sequential_inputs


def get_layers_data(model_config):
    """
    _summary_

    Args:
        model_config (_type_): _description_
    """

    layers_data = []
    for layer_config in model_config:
        layer_type = layer_config["class_name"]
        if layer_type == "Activation":
            layer_type = layer_config["activation"]

        input_shape = layer_config["input_shape"]
        if layer_type in ["Add", "Concatenate"]:
            input_shape = layer_config["output_shape"]
        input_shape = np.asarray(input_shape).flatten()
        input_size = np.prod([x for x in input_shape if x is not None])

        output_shape = np.asarray(layer_config["output_shape"]).flatten()
        output_size = np.prod([x for x in output_shape if x is not None])
        layer_parameters = layer_config["parameters"]
        reuse_factor = layer_config["reuse_factor"]

        layers_data.append(
            {
                "layer_type": layer_type.lower(),
                "layer_input_size": input_size,
                "layer_output_size": output_size,
                "layer_parameter_count": layer_parameters,
                "layer_reuse": reuse_factor,
            }
        )

    return layers_data


def get_global_inputs(model_config, hls_config):
    """
    _summary_

    Args:
        model_config (_type_): _description_
        hls_config (_type_): _description_

    Returns:
        _type_: _description_
    """

    features_to_extract = {
        "dense": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "parameters": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "reuse": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "batchnormalization": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "parameters": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "add": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "concatenate": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "dropout": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "relu": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "sigmoid": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "tanh": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "softmax": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
    }

    def get_layer_features(feature_dict, value):
        if isinstance(value, (list, tuple)):
            value = np.asarray(value).flatten()
            value = np.prod([x for x in value if x is not None])

        if "mean" in feature_dict:
            feature_dict["mean"] += value
        if "min" in feature_dict and value < feature_dict["min"]:
            feature_dict["min"] = value
            if "min_idx" in feature_dict:
                feature_dict["min_idx"] = idx + 1
        if "max" in feature_dict and value > feature_dict["max"]:
            feature_dict["max"] = value
            if "max_idx" in feature_dict:
                feature_dict["max_idx"] = idx + 1

        return feature_dict

    def adjust_feature_values(features, keys=[]):
        for feature in features.values():
            count = feature.get("count", 1)
            for k1 in keys:
                if k1 in feature:
                    for k2 in feature[k1].keys():
                        if np.isinf(feature[k1][k2]):  # Replace inf values
                            feature[k1][k2] = 0

                    if count > 0:  # Total -> mean
                        feature[k1]["mean"] /= count

        return features

    for idx, layer in enumerate(model_config):
        layer_class = layer["class_name"].lower()
        if layer_class == "activation":
            layer_class = layer["activation"]

        if layer_class in features_to_extract:
            features_to_extract[layer_class]["count"] += 1
            for feature_key, layer_key in zip(
                ["inputs", "outputs", "parameters", "reuse"],
                ["input_shape", "output_shape", "parameters", "reuse_factor"],
            ):
                if feature_key in features_to_extract[layer_class]:
                    features_to_extract[layer_class][feature_key] = get_layer_features(
                        features_to_extract[layer_class][feature_key], layer[layer_key]
                    )

    adjusted_features = adjust_feature_values(
        features_to_extract, keys=["inputs", "outputs", "parameters", "reuse"]
    )
    extracted_features = unwrap_nested_dicts(adjusted_features)

    reuse_factor_mean = np.mean([x["reuse_factor"] for x in model_config])

    precision = hls_config["model"]["precision"]
    total_bits, fractional_bits = fixed_precision_to_bit_width(precision)

    strategy = hls_config["model"]["strategy"]
    board = hls_config["board"]

    inputs = {
        "strategy": strategy.lower(),
        "board": board.lower(),
        "precision": precision.lower(),
        "bit_width": total_bits,
        "integer_bits": total_bits - fractional_bits,
        "fractional_bits": fractional_bits,
        "global_reuse": hls_config["model"]["reuse_factor"],
        "reuse_mean": reuse_factor_mean,
    }
    inputs.update(extracted_features)

    fixed_ops = get_network_fixed_ops(model_config, precision)
    inputs.update(fixed_ops)

    return inputs


def get_prediction_targets(model_data, norm_board=None):
    """
    _summary_

    Args:
        model_data (_type_): _description_
        norm_board (_type_): _description_

    Returns:
        _type_: _description_
    """

    resource_report = model_data["resource_report"]
    latency_report = model_data["latency_report"]

    bram = resource_report["bram"]
    dsp = resource_report["dsp"]
    ff = resource_report["ff"]
    lut = resource_report["lut"]

    if norm_board is None:
        targets = {
            "bram": bram,
            "dsp": dsp,
            "ff": ff,
            "lut": lut,
        }
    else:
        board_data = boards_data[norm_board]
        max_bram = board_data["max_bram"]
        max_dsp = board_data["max_dsp"]
        max_ff = board_data["max_ff"]
        max_lut = board_data["max_lut"]

        targets = {
            "bram": max(1 / max_bram, min(bram / max_bram, 2.0)) * 100,
            "dsp": max(1 / max_dsp, min(dsp / max_dsp, 2.0)) * 100,
            "ff": max(1 / max_ff, min(ff / max_ff, 2.0)) * 100,
            "lut": max(1 / max_lut, min(lut / max_lut, 2.0)) * 100,
        }

    cycles_min = latency_report["cycles_min"]
    cycles_max = latency_report["cycles_max"]
    targets.update({"cycles": (cycles_min + cycles_max) / 2.0})

    return targets


def get_network_fixed_ops(model_json, precision):
    """
    _summary_

    Args:
        model_json (_type_): _description_
        precision (_type_): _description_

    Returns:
        _type_: _description_
    """

    total_mult = 0
    total_add = 0
    total_logical = 0
    total_lookup = 0

    total_bits, fractional_bits = fixed_precision_to_bit_width(precision)

    for layer_json in model_json:
        flat_input_shape = np.asarray(layer_json["input_shape"]).flatten()

        if layer_json["class_name"] == "Add":
            input_size = np.prod([x for x in flat_input_shape if x is not None])
            total_add += input_size
        elif layer_json["class_name"] == "Concatenate":
            pass

        elif layer_json["class_name"] == "Conv2D":
            # Assuming "channel-last" format
            input_size = np.prod([x for x in layer_json["input_shape"][:-1] if x is not None])
            use_bias = layer_json["use_bias"]
            mult_in = (
                layer_json["kernel_size"][0]
                * layer_json["kernel_size"][1]
                * layer_json["n_channel"]
            )
            mult_out = layer_json["n_filter"]
            total_mult += (input_size * int(use_bias)) * mult_in * mult_out
            total_add += (input_size * int(use_bias)) * mult_in * mult_out

        else:
            input_size = np.prod([x for x in flat_input_shape if x is not None])
            if layer_json["class_name"] == "Dense":
                neurons = layer_json["neurons"]
                use_bias = layer_json["use_bias"]
                total_mult += (input_size + int(use_bias)) * neurons
                total_add += (input_size + int(use_bias)) * (neurons - 1)

            elif layer_json["class_name"] == "Activation":
                if layer_json["activation"] == "relu":
                    total_logical += input_size
                elif layer_json["activation"] in ["tanh", "sigmoid"]:
                    total_lookup += input_size
                elif layer_json["activation"] == "softmax":
                    # all exps (input_size) and then invert of the exps" sum (+1)
                    total_lookup += input_size + 1
                    # exps sum
                    total_add += input_size - 1
                    # mult each exp with the inverted sum
                    total_mult += input_size

            elif layer_json["class_name"] == "BatchNormalization":
                total_mult += input_size
                total_add += input_size

    total_mult *= total_bits
    total_add *= total_bits
    total_logical *= total_bits

    return {
        "total_mult": total_mult,
        "total_add": total_add,
        "total_logical": total_logical,
        "total_lookup": total_lookup,
    }


def to_dataframe(
    meta_data,
    global_inputs,
    sequential_inputs,
    global_categorical_maps,
    sequential_categorical_maps,
    targets,
):
    """
    _summary_

    Args:
        meta_data (_type_): _description_
        global_inputs (_type_): _description_
        sequential_inputs (_type_): _description_
        global_categorical_maps (_type_): _description_
        sequential_categorical_maps (_type_): _description_
        targets (_type_): _description_

    Returns:
        _type_: _description_
    """

    meta_data = [unwrap_nested_dicts(d) for d in meta_data]
    global_inputs = [unwrap_nested_dicts(d) for d in global_inputs]
    targets = [unwrap_nested_dicts(d) for d in targets]

    tmp = []
    for inputs in sequential_inputs:
        layers_df = pd.DataFrame([unwrap_nested_dicts(d) for d in inputs])
        tmp.append({"sequential_inputs": layers_df})
    sequential_inputs = tmp

    data_df = pd.concat(
        [
            pd.DataFrame(meta_data),
            pd.DataFrame(global_inputs),
            pd.DataFrame(sequential_inputs),
            pd.DataFrame(targets),
        ],
        axis=1,
    )

    # Ordinal coding of global categorical inputs
    for key, value in global_categorical_maps.items():
        data_df[key] = data_df[key].map(value)

    # Ordinal coding of sequential categorical inputs
    for key, value in sequential_categorical_maps.items():
        data_df["sequential_inputs"] = data_df["sequential_inputs"].apply(
            lambda df: df.assign(**{key: df[key].apply(lambda x: value[x] if x in value else x)})
        )

    return data_df
