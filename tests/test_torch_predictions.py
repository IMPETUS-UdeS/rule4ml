from importlib.util import find_spec

import numpy as np
import pytest
import torch

from rule4ml.surrogates import KerasSurrogate, TorchSurrogate

TARGETS = [
    "BRAM_36K",
    "DSP",
    "FF",
    "LUT",
    "CYCLES_MIN",
    "CYCLES_MAX",
    "INTERVAL_MIN",
    "INTERVAL_MAX",
]


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


def test_predict_onnx_twin(estimator, hls_config):
    pytest.importorskip("onnxscript")
    settings = {
        "hls_configs": hls_config,
        "hls4ml_versions": ["0.8.1"],
        "vivado_versions": ["2023.2"],
        "round_digits": 10,
    }
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.BatchNorm2d(8),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 16 * 16, 10),
        torch.nn.Softmax(dim=1),
    ).eval()

    onnx_model = torch.onnx.export(model, (torch.zeros(1, 3, 32, 32),))
    torch_predictions = estimator.predict((model, (3, 32, 32)), **settings)
    onnx_predictions = estimator.predict(onnx_model.model_proto, **settings)

    np.testing.assert_allclose(
        onnx_predictions[TARGETS], torch_predictions[TARGETS], rtol=1e-6
    )


def test_predict_unseen_version(estimator, hls_config):
    with pytest.warns(UserWarning, match="2099.1"):
        prediction_df = estimator.predict(
            _sequential(16),
            hls_configs=hls_config,
            hls4ml_versions="0.8.1",
            vivado_versions="2099.1",
        )

    assert np.isfinite(prediction_df[TARGETS]).all().all()


def test_mc_predict(check_mc_dropout):
    check_mc_dropout(TorchSurrogate(), "gnn/LUT.weights.pt", (["0.8.1"], ["2023.2"]))


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
