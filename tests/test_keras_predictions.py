import itertools
import os

import pytest

keras = pytest.importorskip("keras")
layers = keras.layers
from rule4ml.surrogates import KerasSurrogate

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only


def test_predict_defaults(estimator, model):
    prediction_df = estimator.predict(model)

    assert not prediction_df.empty
    assert not prediction_df.isna().any().any()


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
        for board, strategy in itertools.product(
            ["pynq-z2", "zcu102"], ["Latency", "Resource"]
        )
    ]

    prediction_df = estimator.predict(
        [top_quarks, _keras_sequential()], hls_configs=hls_configs
    )

    assert set(prediction_df["Model"]) == {"TopQuarks", "Jet Classifier"}
    assert set(prediction_df["Board"]) == {"pynq-z2", "zcu102"}


def test_mc_predict(check_mc_dropout):
    check_mc_dropout(KerasSurrogate(), "v2/mlp/LUT.weights.h5", ([], []))


@pytest.fixture(params=["sequential", "functional", "subclassed"])
def model(request):
    return {
        "sequential": _keras_sequential,
        "functional": _keras_functional,
        "subclassed": _keras_subclassed,
    }[request.param]()


class JetClassifier(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
        self.dense3 = layers.Dense(32, activation="relu")
        self.dense4 = layers.Dense(5, activation="softmax")

    def call(self, inputs):
        return self.dense4(self.dense3(self.dense2(self.dense1(inputs))))


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


def _keras_functional():
    inputs = keras.Input(shape=(16,))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(5, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="Jet Classifier")


def _keras_subclassed():
    model = JetClassifier()
    model.build((None, 16))
    return model
