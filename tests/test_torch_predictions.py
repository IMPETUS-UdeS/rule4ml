import os
from importlib.util import find_spec

import numpy as np
import pytest
import torch

from rule4ml.models.wrappers import KerasModelWrapper, TorchModelWrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only

TARGETS = ["BRAM", "DSP", "FF", "LUT", "CYCLES", "INTERVAL"]


def test_predict_defaults(estimator, model):
    prediction_df = estimator.predict(model)
    assert not prediction_df.empty
    assert not prediction_df.isna().any().any()


def test_predict_config(estimator, hls_config):
    predictions = estimator.predict(
        _sequential(16),
        hls_configs=hls_config,
        hls4ml_versions=["0.8.1"],
        vivado_versions=["2023.2"],
    )
    assert len(predictions) == 1
    assert np.isfinite(predictions[TARGETS]).all().all()


def test_predict_same_names(estimator, hls_config):
    predictions = estimator.predict(
        [_sequential(16), _sequential(64)],
        hls_configs=hls_config,
        hls4ml_versions=["0.8.1"],
        vivado_versions=["2023.2"],
    )
    assert list(predictions["Model"]) == ["Sequential", "Sequential"]


def test_mc_predict(check_mc_dropout):
    check_mc_dropout(
        TorchModelWrapper(), "v2/gnn/LUT.weights.pt", (["0.8.1"], ["2023.2"])
    )


@pytest.mark.skipif(find_spec("tensorflow") is not None, reason="TensorFlow installed")
def test_keras_without_tensorflow():
    with pytest.raises(ImportError, match=r"rule4ml\[keras\]"):
        KerasModelWrapper()


@pytest.fixture(params=["sequential", "subclassed"])
def model(request, top_quarks):
    return _sequential(10) if request.param == "sequential" else top_quarks


def _sequential(inputs):
    return torch.nn.Sequential(
        torch.nn.Linear(inputs, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )
