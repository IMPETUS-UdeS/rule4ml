import pytest

from rule4ml.parsers.network_parser import config_from_keras_model

keras = pytest.importorskip("keras")
layers = keras.layers

HLS_CONFIG = {
    "model": {"precision": "ap_fixed<16,6>", "reuse_factor": 1, "strategy": "Resource"}
}
CNN_SHAPES = [  # (class_name, input_shape, output_shape) without batch dimension
    ("Conv2D", (32, 32, 3), (32, 32, 8)),
    ("MaxPooling2D", (32, 32, 8), (16, 16, 8)),
    ("Flatten", (16, 16, 8), (2048,)),
    ("Dense", (2048,), (10,)),
]


def test_parse_pooling_cnn(keras_cnn):
    shapes = [
        (layer["class_name"], layer["input_shape"][1:], layer["output_shape"][1:])
        for layer in config_from_keras_model(keras_cnn, HLS_CONFIG)
        if layer["class_name"] not in ["InputLayer", "Activation"]
    ]
    assert shapes == CNN_SHAPES


@pytest.mark.parametrize("build_shape", [(None, 16), [None, 16]])
def test_parse_subclassed_dense(build_shape):
    model = SubclassedDense()
    model.build(build_shape)

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
