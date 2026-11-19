import sys

import torch

from rule4ml.parsers.data_parser import get_global_inputs, get_layers_data
from rule4ml.parsers.network_parser import (
    config_from_keras_model,
    config_from_onnx_model,
    config_from_torch_model,
)
from rule4ml.parsers.utils import (
    camel_keys_to_snake,
    ensure_input_layer,
    get_board_from_part,
    get_weight_bits,
    name_weighted_layers,
)

DEFAULT_CLOCK_PERIOD = 5.0  # hls4ml's, when the config sets none
DEFAULT_IO_TYPE = "io_parallel"


def model_features(model, hls_config, hls4ml_version=None, vivado_version=None):
    """Get the global and per-layer features of a Keras, Torch or ONNX model.

    Args:
        model (object or tuple): Model or (torch_model, input_shape) pair.
        hls_config (dict): HLS config, camelCase or snake_case.
        hls4ml_version (str, optional): hls4ml version of the prediction.
        vivado_version (str, optional): Vivado version of the prediction.

    Returns:
        tuple: Global feature dict and per-layer feature dicts.

    Raises:
        TypeError: If the model is not a Keras, Torch or ONNX model.
    """

    model, input_shape = unpack_model(model)
    if isinstance(model, torch.nn.Module):
        model_config = config_from_torch_model(model, hls_config, input_shape)
    elif "keras" in sys.modules and isinstance(model, sys.modules["keras"].Model):
        model_config = config_from_keras_model(model, hls_config)
    elif "onnx" in sys.modules and isinstance(model, sys.modules["onnx"].ModelProto):
        model_config = config_from_onnx_model(model, hls_config)
    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. "
            "Expected a Keras, Torch or ONNX model."
        )

    global_inputs = get_global_inputs(
        model_config,
        hls_config,
        clock_period=DEFAULT_CLOCK_PERIOD,
        hls4ml_version=hls4ml_version,
        vivado_version=vivado_version,
    )
    io_type = camel_keys_to_snake(hls_config or {}).get("io_type", DEFAULT_IO_TYPE)
    global_inputs["io_type"] = io_type.lower()  # metadata, not a model input

    return global_inputs, get_layers_data(model_config)


def unpack_model(model):
    """Split a (torch_model, input_shape) pair.

    Args:
        model (object or tuple): Model, or a Torch model and its input shape.

    Returns:
        tuple: The model and its input shape, None for a model alone.
    """

    if (
        isinstance(model, tuple)
        and len(model) == 2
        and isinstance(model[0], torch.nn.Module)
        and isinstance(model[1], (tuple, list))
    ):
        return model

    return model, None


def record_features(record):
    """Get the global and per-layer features of a synthesis record.

    Args:
        record (dict): Synthesis record.

    Returns:
        tuple: The global feature dict and a list of per-layer feature dicts.
    """

    hls_config = dict(record["hls_config"])
    if (target_part := record.get("target_part")) is not None:
        hls_config["board"] = get_board_from_part(target_part)

    snake_config = camel_keys_to_snake(hls_config)
    model_config = name_weighted_layers(
        ensure_input_layer(record["model_config"], snake_config), snake_config
    )

    for layer in model_config:
        weight_bits = get_weight_bits(snake_config, layer, layer.get("name"))
        if weight_bits is not None:
            layer["weight_bits"] = weight_bits

    global_inputs = get_global_inputs(
        model_config,
        hls_config,
        clock_period=(record.get("latency_report") or {}).get("target_clock")
        or DEFAULT_CLOCK_PERIOD,
        hls4ml_version=record.get("hls4ml_version"),
        vivado_version=record.get("vivado_version") or record.get("backend_version"),
    )

    return global_inputs, get_layers_data(model_config)
