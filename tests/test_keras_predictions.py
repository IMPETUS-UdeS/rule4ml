import numpy as np
import pytest
import torch

keras = pytest.importorskip("keras")
layers = keras.layers
from rule4ml.surrogates import KerasSurrogate


def test_predict_model_apis(estimator, model, hls_config):
    settings = {
        "hls_configs": hls_config,
        "hls4ml_versions": ["0.8.1"],
        "vivado_versions": ["2023.2"],
        "round_digits": 10,
    }
    targets = ["BRAM", "DSP", "FF", "LUT", "CYCLES", "INTERVAL"]

    x = inputs = keras.Input(shape=(16,))

    for units in [32, 32, 32]:
        x = layers.Activation("relu")(layers.Dense(units)(x))

    x = layers.Activation("softmax")(layers.Dense(5)(x))

    functional_predictions = estimator.predict(keras.Model(inputs, x), **settings)
    predictions = estimator.predict(model, **settings)

    np.testing.assert_allclose(
        predictions[targets], functional_predictions[targets], rtol=1e-6
    )


def test_predict_pooling_cnn(estimator):
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    hls_config = {
        "board": "alveo-u250",
        "model": {
            "strategy": "Resource",
            "precision": "ap_fixed<16,6>",
            "reuse_factor": 1,
        },
    }

    prediction_df = estimator.predict(
        keras.Model(inputs=inputs, outputs=outputs),
        hls_configs=[hls_config],
        hls4ml_versions=["0.8.1"],
        vivado_versions=["2023.2"],
    )

    assert len(prediction_df) == 1
    assert not prediction_df.isna().any().any()


def test_predict_mixed_models(estimator, top_quarks):
    hls_configs = [
        {
            "model": {
                "precision": "ap_fixed<8, 3>",
                "reuse_factor": 32,
                "strategy": strategy,
            },
            "board": board,
        }
        for board, strategy in [("pynq-z2", "Latency"), ("zcu102", "Resource")]
    ]

    prediction_df = estimator.predict(
        [top_quarks, _keras_sequential()], hls_configs=hls_configs
    )

    assert set(prediction_df["Model"]) == {"TopQuarks", "Jet Classifier"}
    assert set(prediction_df["Board"]) == {"pynq-z2", "zcu102"}


def test_predict_torch_twin(estimator, hls_config):
    settings = {
        "hls_configs": {**hls_config, "layer_name": {"dense2": {"reuse_factor": 16}}},
        "hls4ml_versions": ["0.8.1"],
        "vivado_versions": ["2023.2"],
        "round_digits": 10,
    }
    targets = ["BRAM", "DSP", "FF", "LUT", "CYCLES", "INTERVAL"]

    x = inputs = keras.Input(shape=(16,))
    x = layers.Activation("relu")(layers.Dense(32, name="dense1")(x))

    for idx in [2, 3]:
        residual = layers.Activation("relu")(layers.Dense(32, name=f"dense{idx}")(x))
        x = layers.Add()([x, residual])

    x = layers.Activation("softmax")(layers.Flatten()(layers.Dense(5)(x)))

    keras_predictions = estimator.predict(keras.Model(inputs, x), **settings)
    torch_predictions = estimator.predict(TorchJetClassifier(), **settings)

    np.testing.assert_allclose(
        torch_predictions[targets], keras_predictions[targets], rtol=1e-6
    )


def test_mc_predict(check_mc_dropout):
    check_mc_dropout(KerasSurrogate(), "v2/mlp/LUT.weights.h5", ([], []))


@pytest.fixture(params=["sequential", "subclassed"])
def model(request):
    if request.param == "sequential":
        return _keras_sequential()

    return _keras_subclassed()


class JetClassifier(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.dense3 = layers.Dense(32, activation="relu")
        self.dense4 = layers.Dense(5, activation="softmax")

    def call(self, inputs):
        return self.dense4(self.dense3(self.dense2(self.dense1(inputs))))


class TorchJetClassifier(torch.nn.Module):
    """Residual Jet Classifier in Torch, mixing activation forms."""

    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(16, 32)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(32, 32)
        self.dense3 = torch.nn.Linear(32, 32)
        self.dense4 = torch.nn.Linear(32, 5)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = x + torch.relu(self.dense2(x))
        x = x + self.dense3(x).relu()

        return self.softmax(self.dense4(x).view(-1, 5))


def _keras_sequential():
    return keras.Sequential(
        [
            keras.Input(shape=(16,)),
            layers.Dense(32),
            layers.Activation("relu"),
            layers.Dense(32),
            layers.Activation("relu"),
            layers.Dense(32),
            layers.Activation("relu"),
            layers.Dense(5),
            layers.Activation("softmax"),
        ],
        name="Jet Classifier",
    )


def _keras_subclassed():
    model = JetClassifier()
    model.build((None, 16))
    return model
