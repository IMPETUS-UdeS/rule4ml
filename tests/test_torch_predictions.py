from importlib.util import find_spec

import numpy as np
import pytest
import torch

from rule4ml.surrogates import KerasSurrogate, TorchSurrogate

TARGETS = ["BRAM_36K", "DSP", "FF", "LUT", "CYCLES", "INTERVAL"]


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


def test_predict_forward_order(estimator, hls_config):
    settings = {
        "hls_configs": hls_config,
        "hls4ml_versions": ["0.8.1"],
        "vivado_versions": ["2023.2"],
        "round_digits": 10,
    }

    reordered = estimator.predict(Reordered(), **settings)
    functional = estimator.predict(Functional(), **settings)

    np.testing.assert_allclose(reordered[TARGETS], functional[TARGETS], rtol=1e-6)


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


class Reordered(torch.nn.Module):
    """Modules registered out of forward order, one called twice."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(12, 1)
        self.act = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.b = torch.nn.Linear(16, 4)
        self.a = torch.nn.Linear(16, 8)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        a = self.act(a)  # input is not the previous node

        return self.head(self.flatten(self.act(torch.cat([a, b], dim=1))))


class Functional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(16, 8)
        self.b = torch.nn.Linear(16, 4)
        self.head = torch.nn.Linear(12, 1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        a = torch.relu(a)

        return self.head(torch.relu(torch.cat([a, b], dim=1)).flatten(1))
