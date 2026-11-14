import pytest

from rule4ml.parsers.data_parser import get_global_inputs
from rule4ml.parsers.network_parser import config_from_keras_model

keras = pytest.importorskip("keras")
layers = keras.layers

HLS_CONFIG = {
    "board": "zcu102",
    "model": {"precision": "ap_fixed<16,6>", "reuse_factor": 1, "strategy": "Resource"},
}


def test_parse_pooling_cnn(keras_cnn):
    layers_config = config_from_keras_model(keras_cnn, HLS_CONFIG)
    shapes = [
        (layer["class_name"], layer["input_shape"][1:], layer["output_shape"][1:])
        for layer in layers_config
        if layer["class_name"] not in ["InputLayer", "Activation"]
    ]
    inputs = get_global_inputs(layers_config, HLS_CONFIG)

    assert shapes == [
        ("Conv2D", (32, 32, 3), (32, 32, 8)),
        ("MaxPooling2D", (32, 32, 8), (16, 16, 8)),
        ("Flatten", (16, 16, 8), (2048,)),
        ("Dense", (2048,), (10,)),
    ]
    for statistic in ["filters", "kernel_size", "strides"]:
        assert inputs[f"conv2d_{statistic}_mean"] > 0


def test_parse_reuse_overrides():
    model = keras.Sequential(
        [
            keras.Input(shape=(16,), name="Features"),
            layers.Dense(32, activation="relu", name="Hidden"),
            layers.Dense(4),
            layers.Activation("softmax"),
        ]
    )
    hls_config = {
        **HLS_CONFIG,
        "LayerType": {"Dense": {"ReuseFactor": 8}, "Activation": {"ReuseFactor": 2}},
        "LayerName": {
            "FEATURES": {"ReuseFactor": 16},
            "HIDDEN_RELU": {"ReuseFactor": 4},
        },
    }

    layers_config = config_from_keras_model(model, hls_config)

    assert [layer["reuse_factor"] for layer in layers_config] == [16, 8, 4, 8, 1]


def test_parse_subclassed_dense():
    model = SubclassedDense()
    model.build([None, 16])

    layers_config = config_from_keras_model(model, HLS_CONFIG)
    dense = next(layer for layer in layers_config if layer["class_name"] == "Dense")

    assert (dense["input_shape"][1:], dense["output_shape"][1:]) == ((16,), (4,))


class SubclassedDense(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(4)

    def call(self, inputs):
        return self.dense(inputs)


@pytest.fixture(params=["functional", "sequential"])
def keras_cnn(request):
    cnn_layers = [
        layers.Conv2D(8, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]

    if request.param == "sequential":
        model = keras.Sequential(cnn_layers)
        model.build((None, 32, 32, 3))
        return model

    outputs = inputs = keras.Input(shape=(32, 32, 3))
    for layer in cnn_layers:
        outputs = layer(outputs)
    return keras.Model(inputs, outputs)
