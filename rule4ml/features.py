import sys

import torch

from rule4ml.parsers.data_parser import get_global_inputs, get_layers_data
from rule4ml.parsers.network_parser import (
    config_from_keras_model,
    config_from_torch_model,
)
from rule4ml.parsers.utils import get_board_from_part


def model_features(model, hls_config, hls4ml_version=None, vivado_version=None):
    """Return the global and per-layer features of a Keras or Torch model."""

    if isinstance(model, torch.nn.Module):
        model_config = config_from_torch_model(model, hls_config)
    elif "keras" in sys.modules and isinstance(model, sys.modules["keras"].Model):
        model_config = config_from_keras_model(model, hls_config)
    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}. Expected a Keras or Torch model."
        )

    global_inputs = get_global_inputs(
        model_config,
        hls_config,
        clock_period=10.0,  # when the config sets none
        hls4ml_version=hls4ml_version,
        vivado_version=vivado_version,
    )

    return global_inputs, get_layers_data(model_config)


def record_features(record):
    """Return the global and per-layer features of a synthesis record."""

    hls_config = dict(record["hls_config"])
    if (target_part := record.get("target_part")) is not None:
        hls_config["board"] = get_board_from_part(target_part)

    global_inputs = get_global_inputs(
        record["model_config"],
        hls_config,
        clock_period=(record.get("latency_report") or {}).get("target_clock"),
        hls4ml_version=record.get("hls4ml_version"),
        vivado_version=record.get("vivado_version") or record.get("backend_version"),
    )

    return global_inputs, get_layers_data(record["model_config"])
