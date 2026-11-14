import json
import os

import numpy as np

from rule4ml.parsers.utils import (
    camel_keys_to_snake,
    fixed_precision_to_bit_width,
    get_activation_name,
    to_lower_keys,
    unwrap_nested_dicts,
)

boards_path = os.path.join(os.path.dirname(__file__), "supported_boards.json")
boards_data = {}
with open(boards_path) as json_file:
    boards_data = json.load(json_file)

# Default ordinal encodings for categorical inputs
default_board_map = {key.lower(): idx + 1 for idx, key in enumerate(boards_data.keys())}
default_layer_type_map = {
    "inputlayer": 1,
    "dense": 2,
    "conv1d": 3,
    "conv2d": 4,
    "maxpooling1d": 5,
    "averagepooling1d": 6,
    "maxpooling2d": 7,
    "averagepooling2d": 8,
    "relu": 9,
    "sigmoid": 10,
    "tanh": 11,
    "softmax": 12,
    "batchnormalization": 13,
    "add": 14,
    "concatenate": 15,
    "dropout": 16,
    "flatten": 17,
}
default_strategy_map = {"latency": 1, "resource": 2}
default_hls4ml_map = {"0.8.1": 1, "1.1.0": 2}
default_vivado_map = {
    "2019.1": 1,
    "2019.2": 2,
    "2020.1": 3,
    "2020.2": 4,
    "2021.1": 5,
    "2021.2": 6,
    "2022.1": 7,
    "2022.2": 8,
    "2023.1": 9,
    "2023.2": 10,
    "2024.1": 11,
    "2024.2": 12,
}


def get_layers_data(model_config):
    """
    Processes the layers data of a model.

    Args:
        model_data (dict): The model data containing configuration and layer information.

    Returns:
        list: A list of processed layer data dictionaries.
    """

    layers_fixed_ops = []
    fixed_ops = get_network_fixed_ops(model_config)
    if "layers" in fixed_ops:
        layers_fixed_ops = fixed_ops["layers"]

    layers_data = []
    for idx, layer_config in enumerate(model_config):
        layer_type = layer_config["class_name"].lower()
        layer_type = layer_type.removeprefix("q")  # qkeras layers

        if layer_type == "activation":
            layer_type = get_activation_name(layer_config["activation"])
            if layer_type == "linear":  # ignore linear activations
                continue

        input_shape = layer_config["input_shape"]
        if layer_type in ["add", "concatenate"]:
            input_shape = layer_config["output_shape"]
        input_shape = np.asarray(input_shape).flatten()
        input_size = np.prod([x for x in input_shape if x is not None])

        output_shape = np.asarray(layer_config["output_shape"]).flatten()
        output_size = np.prod([x for x in output_shape if x is not None])

        layer_parameters = layer_config["parameters"]
        layer_trainable_parameters = layer_config["trainable_parameters"]

        layer_filters = layer_config.get("filters", 0)
        layer_kernels = layer_config.get("kernel_size", 0)
        if isinstance(layer_kernels, (list, tuple)):
            if len(layer_kernels) == 2:
                layer_kernel_height = layer_kernels[0]
                layer_kernel_width = layer_kernels[1]
            else:
                layer_kernel_height = layer_kernel_width = layer_kernels[0]
        else:
            layer_kernel_height = layer_kernel_width = layer_kernels

        layer_strides = layer_config.get("strides", 0)
        if isinstance(layer_strides, (list, tuple)):
            if len(layer_strides) == 2:
                layer_stride_height = layer_strides[0]
                layer_stride_width = layer_strides[1]
            else:
                layer_stride_height = layer_stride_width = layer_strides[0]
        else:
            layer_stride_height = layer_stride_width = layer_strides

        reuse_factor = layer_config["reuse_factor"]

        data = {
            "layer_type": layer_type.lower(),
            "layer_input_size": input_size,
            "layer_output_size": output_size,
            "layer_parameter_count": layer_parameters,
            "layer_trainable_parameter_count": layer_trainable_parameters,
            "layer_filters": layer_filters,
            "layer_kernel_height": layer_kernel_height,
            "layer_kernel_width": layer_kernel_width,
            "layer_stride_height": layer_stride_height,
            "layer_stride_width": layer_stride_width,
            "layer_reuse": reuse_factor,
            "layer_op_add": layers_fixed_ops[idx].get("add", 0),
            "layer_op_mult": layers_fixed_ops[idx].get("mult", 0),
            "layer_op_logical": layers_fixed_ops[idx].get("logical", 0),
            "layer_op_lookup": layers_fixed_ops[idx].get("lookup", 0),
        }
        layers_data.append(data)

    return layers_data


def get_global_inputs(model_config, hls_config, **kwargs):
    """Gets the global input features from the model and HLS configurations."""

    features_to_extract = {
        "dense": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "parameters": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "reuse": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "conv1d": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "parameters": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "filters": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "kernel_size": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "strides": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "reuse": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "conv2d": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "parameters": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "filters": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "kernel_size": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "strides": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "reuse": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "batchnormalization": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "parameters": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "add": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "concatenate": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "dropout": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "relu": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "sigmoid": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "tanh": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "count": 0,
        },
        "softmax": {
            "inputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
            "outputs": {
                "mean": 0,
                "min": np.inf,
                "min_idx": 0,
                "max": -np.inf,
                "max_idx": 0,
            },
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

    def adjust_feature_values(features, keys=()):
        for feature in features.values():
            count = feature.get("count", 1)
            for k1 in keys:
                if k1 in feature:
                    for k2 in feature[k1]:
                        if np.isinf(feature[k1][k2]):  # Replace inf values
                            feature[k1][k2] = 0

                    if count > 0:  # Total -> mean
                        feature[k1]["mean"] /= count

        return features

    layer_keys = {
        "inputs": "input_shape",
        "outputs": "output_shape",
        "parameters": "parameters",
        "reuse": "reuse_factor",
        "filters": "filters",
        "kernel_size": "kernel_size",
        "strides": "strides",
    }

    for idx, layer in enumerate(model_config):
        layer_class = layer["class_name"].lower()
        if layer_class[0] == "q" and layer_class[1:] in features_to_extract:
            layer_class = layer_class[1:]

        if layer_class == "activation":
            layer_class = layer["activation"]

        if layer_class in features_to_extract:
            features_to_extract[layer_class]["count"] += 1
            for feature_key, layer_key in layer_keys.items():
                if feature_key in features_to_extract[layer_class]:
                    features_to_extract[layer_class][feature_key] = get_layer_features(
                        features_to_extract[layer_class][feature_key], layer[layer_key]
                    )

    adjusted_features = adjust_feature_values(features_to_extract, keys=layer_keys)
    extracted_features = unwrap_nested_dicts(adjusted_features)

    reuse_factor_mean = np.mean([x["reuse_factor"] for x in model_config])

    hls_config = camel_keys_to_snake(hls_config)
    precision = hls_config["model"]["precision"]
    if isinstance(precision, dict):
        precision = precision["default"]
    total_bits, integer_bits = fixed_precision_to_bit_width(precision)

    strategy = hls_config["model"]["strategy"]
    board = hls_config["board"]
    clock_period = hls_config.get("clock_period")
    if clock_period is None:
        clock_period = kwargs.get("clock_period")

    inputs = {
        "strategy": strategy.lower(),
        "board": board.lower(),
        "precision": precision.lower(),
        "bit_width": total_bits,
        "integer_bits": integer_bits,
        "fractional_bits": total_bits - integer_bits,
        "global_reuse": hls_config["model"]["reuse_factor"],
        "reuse_mean": reuse_factor_mean,
        "clock_period": None if clock_period is None else float(clock_period),
        "hls4ml_version": kwargs.get("hls4ml_version", None),
        "vivado_version": kwargs.get("vivado_version", None),
    }

    inputs.update(extracted_features)

    fixed_ops = get_network_fixed_ops(model_config)
    if "layers" in fixed_ops:
        fixed_ops.pop("layers")
    inputs.update(fixed_ops)

    return inputs


def get_prediction_targets(model_data):
    """Extract raw resource and latency targets from synthesis reports."""

    latency_report = model_data["latency_report"]
    resource_report = model_data["resource_report"]

    resource_report = to_lower_keys(resource_report)
    latency_report = to_lower_keys(latency_report)

    bram = resource_report.get("bram", None)
    dsp = resource_report.get("dsp", None)
    ff = resource_report.get("ff", None)
    lut = resource_report.get("lut", None)

    if isinstance(bram, str):
        bram = float(bram.strip())
    if isinstance(dsp, str):
        dsp = float(dsp.strip())
    if isinstance(ff, str):
        ff = float(ff.strip())
    if isinstance(lut, str):
        lut = float(lut.strip())

    targets = {"bram": bram, "dsp": dsp, "ff": ff, "lut": lut}

    cycles_min = latency_report.get("cycles_min", None)
    cycles_max = latency_report.get("cycles_max", None)
    interval_min = latency_report.get("interval_min", None)
    interval_max = latency_report.get("interval_max", None)

    if isinstance(cycles_min, str):
        cycles_min = float(cycles_min.strip())
    if isinstance(cycles_max, str):
        cycles_max = float(cycles_max.strip())
    if isinstance(interval_min, str):
        interval_min = float(interval_min.strip())
    if isinstance(interval_max, str):
        interval_max = float(interval_max.strip())

    if cycles_min is not None and cycles_max is not None:
        targets["cycles"] = (cycles_min + cycles_max) / 2.0
    else:
        targets["cycles"] = None

    if interval_min is not None and interval_max is not None:
        targets["interval"] = (interval_min + interval_max) / 2.0
    else:
        targets["interval"] = None

    return targets


def get_network_fixed_ops(model_json):
    """Count fixed-point add, mult, logical and lookup ops per layer and in total."""

    fops_dict = {
        "total_add": 0,
        "total_mult": 0,
        "total_logical": 0,
        "total_lookup": 0,
        "layers": [],
    }

    for layer_json in model_json:
        layer_add = 0
        layer_mult = 0
        layer_logical = 0
        layer_lookup = 0

        input_shape = np.asarray(layer_json["input_shape"])
        flat_input_shape = input_shape.flatten()
        output_shape = np.asarray(layer_json["output_shape"])
        use_bias = layer_json.get("use_bias", False)

        layer_class = layer_json["class_name"].lower()
        layer_class = layer_class.removeprefix("q")  # QKeras layers

        if layer_class == "add":
            inbound_count = max(2, len(layer_json.get("inbound_layers", [])))
            output_size = np.prod([x for x in output_shape if x is not None])
            layer_add = (inbound_count - 1) * output_size
        elif layer_class == "concatenate":
            pass

        elif layer_class in ["conv1d", "conv2d"]:
            # Assuming "channel-last" format
            groups = layer_json.get("groups", 1)
            output_size = np.prod([x for x in output_shape[:-1] if x is not None])

            kernel_size = list(layer_json["kernel_size"])
            if (len(kernel_size) == 1) and (layer_class == "conv2d"):
                kernel_size = [kernel_size[0], kernel_size[0]]
            kernel_size = np.prod(kernel_size)

            fan_in = (kernel_size * layer_json["channels"]) // max(1, groups)
            filters = layer_json["filters"]

            layer_add = output_size * filters * (fan_in - 1 + int(use_bias))
            layer_mult = output_size * filters * fan_in

        else:
            input_size = np.prod([x for x in flat_input_shape if x is not None])
            if layer_class == "dense":
                neurons = layer_json.get("neurons", output_shape[-1])
                layer_add = (input_size - 1 + int(use_bias)) * neurons
                layer_mult = input_size * neurons

            elif layer_class == "activation":
                activation_type = get_activation_name(layer_json["activation"].lower())
                if activation_type == "relu":
                    layer_logical = input_size
                elif activation_type in ["tanh", "sigmoid"]:
                    layer_lookup = input_size
                elif activation_type == "softmax":
                    # all exps (input_size) and then invert of the sum of exps (+1)
                    layer_lookup = input_size + 1
                    # exps sum
                    layer_add = max(0, input_size - 1)
                    # mult each exp with the inverted sum
                    layer_mult = input_size

            elif layer_class == "batchnormalization":
                layer_mult = input_size
                layer_add = input_size

        fops_dict["layers"].append(
            {
                "add": layer_add,
                "mult": layer_mult,
                "logical": layer_logical,
                "lookup": layer_lookup,
            }
        )

        fops_dict["total_add"] += layer_add
        fops_dict["total_mult"] += layer_mult
        fops_dict["total_logical"] += layer_logical
        fops_dict["total_lookup"] += layer_lookup

    return fops_dict
