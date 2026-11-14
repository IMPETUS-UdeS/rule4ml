import os
from importlib.util import find_spec

import numpy as np
import pytest
import torch

from rule4ml.surrogates import KerasSurrogate, TorchSurrogate

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only

TARGETS = ["BRAM", "DSP", "FF", "LUT", "CYCLES", "INTERVAL"]


def test_predict_defaults(estimator, top_quarks):
    prediction_df = estimator.predict(top_quarks)

    assert not prediction_df.empty
    assert not prediction_df.isna().any().any()


def test_predict_batch_context(estimator, hls_config):
    settings = {
        "hls_configs": hls_config,
        "hls4ml_versions": ["0.8.1"],
        "vivado_versions": ["2023.2"],
        "round_digits": 10,
    }

    model = _sequential(16)
    alone = estimator.predict(model, **settings)
    together = estimator.predict([model, _sequential(64)], **settings)

    assert len(alone) == 1
    assert np.isfinite(alone[TARGETS]).all().all()
    assert list(together["Model"]) == ["Sequential", "Sequential"]
    np.testing.assert_allclose(
        together[TARGETS][:1], alone[TARGETS], rtol=1e-5, atol=1e-6
    )


def test_mc_predict(check_mc_dropout):
    check_mc_dropout(TorchSurrogate(), "v2/gnn/LUT.weights.pt", (["0.8.1"], ["2023.2"]))


@pytest.mark.skipif(find_spec("tensorflow") is not None, reason="TensorFlow installed")
def test_keras_without_tensorflow():
    with pytest.raises(ImportError, match=r"rule4ml\[keras\]"):
        KerasSurrogate()


def _sequential(inputs):
    return torch.nn.Sequential(
        torch.nn.Linear(inputs, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )
