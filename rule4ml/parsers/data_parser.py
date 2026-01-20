import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from glob import glob

import numpy as np
import pandas as pd

from rule4ml.parsers.utils import (
    camel_keys_to_snake,
    fixed_precision_to_bit_width,
    get_activation_name,
    get_board_from_part,
    to_lower_keys,
    unwrap_nested_dicts,
)

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

    # Only include models that contain specific layers
    include_layers: list = field(default_factory=lambda: [])
    # Exclude models that contain specific layers
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

    resource_keys: list = field(default_factory=lambda: ["CSynthesisReport"])


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

    include_layers = data_filter.include_layers.copy()

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

        if include_layers and layer_class in include_layers:
            include_layers.remove(layer_class)

        if layer_class == "Activation":
            activation = layer_data["activation"]
            if activation.lower() == "softmax":
                if data_filter.output_softmax_only and idx != (n_layers - 1):
                    return False
                softmax_count += 1
                if data_filter.output_softmax_only and idx < n_layers - 1:
                    return False
            if activation in data_filter.exclude_layers:
                return False

            if include_layers and activation in include_layers:
                include_layers.remove(activation)

    if len(include_layers) > 0:
        return False

    if (data_filter.min_softmax_count > 0 and softmax_count < data_filter.min_softmax_count) or (
        data_filter.max_softmax_count > 0 and softmax_count > data_filter.max_softmax_count
    ):
        return False

    hls_config = parsed_data["hls_config"]

    if data_filter.strategies:
        strategy = hls_config["Model"]["Strategy"]
        if strategy not in data_filter.strategies:
            return False

    if data_filter.precisions:
        precision = hls_config["Model"]["Precision"]
        if precision not in data_filter.precisions:
            return False

    if data_filter.boards:
        board = hls_config["board"]
        if board not in data_filter.boards:
            return False

    if data_filter.resource_keys:
        resource_keys = list(parsed_data["resource_report"].keys())
        for resource_key in data_filter.resource_keys:
            if resource_key not in resource_keys:
                return False

    return True


def batch_iterable(iterable, batch_size):
    """
    Splits an iterable into batches of a specified size.

    Args:
        iterable (iterable): The input iterable to be batched.
        batch_size (int): The size of each batch.

    Yields:
        list: A batch of items from the iterable.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_json_files(filenames):
    """
    Reads multiple JSON files and returns their content as a list of dictionaries.

    Args:
        filenames (list): A list of paths to the JSON files.

    Returns:
        list: A list of dictionaries containing the JSON data.
    """

    json_data = []
    for filename in filenames:
        try:
            with open(filename) as json_file:
                data = json.load(json_file)
                if not isinstance(data, list):
                    data = [data]
                json_data.extend(data)
        except Exception as e:
            raise ValueError(f"Error reading JSON file \"{filename}\": {e}")

    return json_data


def read_from_json(
    file_patterns, data_filter: ParsedDataFilter = None, max_workers=1, batch_size=128
):
    """
    _summary_

    Args:
        file_patterns (_type_): _description_
        data_filter (ParsedDataFilter): _description_
        max_workers (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """

    if isinstance(file_patterns, (list, tuple)):
        file_patterns = list(file_patterns)

        json_files = glob(file_patterns[0])
        json_files = np.asarray(json_files)
        if len(file_patterns) > 1:
            for name in file_patterns[1:]:
                json_files = np.append(json_files, glob(name))
    else:
        json_files = glob(file_patterns)

    batches = list(batch_iterable(json_files, batch_size))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(read_json_files, batches)
    json_data = [entry for result in results for entry in result if entry]

    # Optionally filter the json data
    if data_filter is not None:
        json_data = [
            model_data for model_data in json_data if filter_match(model_data, data_filter)
        ]

    return json_data


def process_global_batch(model_batch, resource_key, normalize):
    batch_meta, batch_inputs, batch_targets = [], [], []

    for model_data in model_batch:
        if not model_data:
            continue

        model_config = model_data["model_config"]
        hls_config = model_data["hls_config"]
        if (target_part := model_data.get("target_part", None)) is not None:
            hls_config["board"] = get_board_from_part(target_part)

        clock_period = hls_config.get("clock_period", None)
        if not clock_period:
            clock_period = model_data["latency_report"].get("target_clock", None)
        if clock_period is not None:
            clock_period = float(clock_period)

        hls4ml_version = model_data.get("hls4ml_version", None)
        vivado_version = model_data.get("vivado_version", None)
        if not vivado_version:
            vivado_version = model_data.get("backend_version", None)

        batch_meta.append(unwrap_nested_dicts(model_data["meta_data"]))
        batch_inputs.append(
            unwrap_nested_dicts(
                get_global_inputs(
                    model_config,
                    hls_config,
                    clock_period=clock_period,
                    hls4ml_version=hls4ml_version,
                    vivado_version=vivado_version,
                )
            )
        )

        norm_board = hls_config["board"] if normalize else None
        batch_targets.append(
            unwrap_nested_dicts(get_prediction_targets(model_data, resource_key, norm_board))
        )

    return batch_meta, batch_inputs, batch_targets


def get_global_data(parsed_data, resource_key=None, normalize=False, max_workers=1, batch_size=128):
    batches = list(batch_iterable(parsed_data, batch_size))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_global_batch, b, resource_key, normalize) for b in batches
        ]

        # Flatten the batched results
        meta, inputs, targets = [], [], []
        for future in futures:
            m, i, t = future.result()
            meta.extend(m)
            inputs.extend(i)
            targets.extend(t)

    return meta, inputs, targets


def get_layers_data(model_config, target_depth=None):
    """
    Processes the layers data of a model.

    Args:
        model_data (dict): The model data containing configuration and layer information.
        target_depth (int, optional): The target depth for padding the layers data.

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
        if layer_type.startswith("q"):  # qkeras layers
            layer_type = layer_type[1:]

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

        # layer_use_bias = 1 if layer_config.get("use_bias", False) else 0
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

    # Optional zero-padding
    if target_depth is not None:
        pad_count = target_depth - len(layers_data)
        if pad_count > 0:
            pad = [{key: 0 for key in layers_data[0].keys()}] * pad_count
            layers_data.extend(pad)

    return layers_data


def process_model_batch(model_batch, max_model_depth):
    result = []
    for model_data in model_batch:
        if not model_data:
            continue

        layers_data = get_layers_data(model_data["model_config"], target_depth=max_model_depth)
        result.append(layers_data)
    return result


def get_sequential_data(parsed_data, max_workers=1, batch_size=128):
    max_model_depth = max(len(model["model_config"]) for model in parsed_data if model)

    batches = list(batch_iterable(parsed_data, batch_size))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_model_batch, b, max_model_depth) for b in batches]
        results = [r for f in futures for r in f.result()]

    return results


def get_global_inputs(model_config, hls_config, **kwargs):
    """
    Gets the global input features from the model and HLS configurations.

    Args:
        model_config (_type_): _description_
        hls_config (_type_): _description_
        **kwargs: Additional keyword arguments,
            such as 'clock_period', 'hls4ml_version',
            'vivado_version', etc.

    Returns:
        dict: A dictionary containing the extracted global features.
    """

    features_to_extract = {
        "dense": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "parameters": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "reuse": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "conv1d": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "parameters": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "filters": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "kernel_size": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "strides": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "reuse": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "count": 0,
        },
        "conv2d": {
            "inputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "outputs": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "parameters": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "filters": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "kernel_size": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
            "strides": {"mean": 0, "min": np.inf, "min_idx": 0, "max": -np.inf, "max_idx": 0},
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
        if layer_class[0] == "q" and layer_class[1:] in features_to_extract:
            layer_class = layer_class[1:]

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

    hls_config = camel_keys_to_snake(hls_config)
    precision = hls_config["model"]["precision"]
    if isinstance(precision, dict):
        precision = precision["default"]
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
        "clock_period": kwargs.get("clock_period", None),
        "hls4ml_version": kwargs.get("hls4ml_version", None),
        "vivado_version": kwargs.get("vivado_version", None),
    }
    inputs.update(extracted_features)

    fixed_ops = get_network_fixed_ops(model_config)
    if "layers" in fixed_ops:
        fixed_ops.pop("layers")
    inputs.update(fixed_ops)

    return inputs


def get_prediction_targets(model_data, resource_key, norm_board=None):
    """
    _summary_

    Args:
        model_data (_type_): _description_
        resource_key (_type_): _description_
        norm_board (_type_): _description_

    Returns:
        _type_: _description_
    """

    latency_report = model_data["latency_report"]
    if resource_key:
        resource_report = model_data["resource_report"][resource_key]
    else:
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

        # targets = {
        #     "bram": max(1 / max_bram, min(bram / max_bram, 2.0)) * 100,
        #     "dsp": max(1 / max_dsp, min(dsp / max_dsp, 2.0)) * 100,
        #     "ff": max(1 / max_ff, min(ff / max_ff, 2.0)) * 100,
        #     "lut": max(1 / max_lut, min(lut / max_lut, 2.0)) * 100,
        # }

        targets = {
            "bram": max(1 / max_bram, (bram / max_bram)) * 100 if bram is not None else None,
            "dsp": max(1 / max_dsp, (dsp / max_dsp)) * 100 if dsp is not None else None,
            "ff": max(1 / max_ff, (ff / max_ff)) * 100 if ff is not None else None,
            "lut": max(1 / max_lut, (lut / max_lut)) * 100 if lut is not None else None,
        }

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
    """
    _summary_

    Args:
        model_json (_type_): _description_

    Returns:
        _type_: _description_
    """

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
        if layer_class.startswith("q"):  # QKeras layers
            layer_class = layer_class[1:]

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


def batch_to_dataframe(
    meta_batch,
    global_batch,
    seq_batch,
    target_batch,
    global_categorical_maps,
    sequential_categorical_maps,
):
    # Unwrap dicts
    meta_batch = [unwrap_nested_dicts(m) for m in meta_batch]
    global_batch = [unwrap_nested_dicts(g) for g in global_batch]
    target_batch = [unwrap_nested_dicts(t) for t in target_batch]

    # Convert sequential to DataFrames
    sequential_wrapped = []
    for layers in seq_batch:
        df = pd.DataFrame([unwrap_nested_dicts(layer) for layer in layers])
        sequential_wrapped.append({"sequential_inputs": df})

    # Build per-batch DataFrame
    df = pd.concat(
        [
            pd.DataFrame(meta_batch),
            pd.DataFrame(global_batch),
            pd.DataFrame(sequential_wrapped),
            pd.DataFrame(target_batch),
        ],
        axis=1,
    )

    # Encode global categorical features
    for key, mapping in global_categorical_maps.items():
        df[key] = df[key].map(mapping)

    # Encode sequential categorical features
    for key, mapping in sequential_categorical_maps.items():
        df["sequential_inputs"] = df["sequential_inputs"].apply(
            lambda sdf: sdf.assign(**{key: sdf[key].map(lambda x: mapping.get(x, x))})
        )

    return df


def to_dataframe(
    meta_data,
    global_inputs,
    sequential_inputs,
    global_categorical_maps,
    sequential_categorical_maps,
    targets,
    max_workers=1,
    batch_size=128,
):
    """
    Batched and parallelized to_dataframe implementation.
    """

    max_depth = max(len(layers) for layers in sequential_inputs)
    for i in range(len(sequential_inputs)):
        pad_count = max_depth - len(sequential_inputs[i])
        if pad_count > 0:
            pad = [{key: 0 for key in sequential_inputs[i][0].keys()}] * pad_count
            sequential_inputs[i].extend(pad)

    n = len(global_inputs)
    batches = [
        (
            meta_data[i : i + batch_size],
            global_inputs[i : i + batch_size],
            sequential_inputs[i : i + batch_size],
            targets[i : i + batch_size],
        )
        for i in range(0, n, batch_size)
    ]

    args = [
        (m, g, s, t, global_categorical_maps, sequential_categorical_maps)
        for (m, g, s, t) in batches
    ]

    dataframes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for df in executor.map(lambda p: batch_to_dataframe(*p), args):
            dataframes.append(df)

    return pd.concat(dataframes, axis=0).reset_index(drop=True)


def json_to_df(json_data, global_maps, sequential_maps, normalize, max_workers=16):
    meta_data, global_inputs, targets = get_global_data(
        json_data, normalize=normalize, max_workers=max_workers
    )
    sequential_inputs = get_sequential_data(json_data, max_workers=max_workers)

    df = to_dataframe(
        meta_data=meta_data,
        global_inputs=global_inputs,
        sequential_inputs=sequential_inputs,
        global_categorical_maps=global_maps,
        sequential_categorical_maps=sequential_maps,
        targets=targets,
        max_workers=max_workers,
    )
    return df
