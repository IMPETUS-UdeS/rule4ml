import itertools
import os
import unittest

import keras
import pandas as pd
import torch
from keras.layers import Dense, Input

from rule4ml.models.wrappers import MultiModelWrapper

# force tests on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class MultiModelWrapperTests(unittest.TestCase):
    def test_wrapper_instance(self):
        estimator = MultiModelWrapper()
        self.assertIsInstance(estimator, MultiModelWrapper)

    def test_load_default_models(self):
        estimator = MultiModelWrapper()
        estimator.load_default_models()

    def test_keras_sequential_prediction(self):
        input_size = 16
        model_to_predict = keras.Sequential(
            layers=[
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Dense(32, use_bias=True),
                keras.layers.Activation("relu"),
                keras.layers.Dense(32, use_bias=True),
                keras.layers.Activation("relu"),
                keras.layers.Dense(32, use_bias=True),
                keras.layers.Activation("relu"),
                keras.layers.Dense(5, use_bias=True),
                keras.layers.Activation("softmax"),
            ],
            name="Jet Classifier",
        )
        model_to_predict.build((None, input_size))

        estimator = MultiModelWrapper()
        estimator.load_default_models()

        prediction_df = estimator.predict(model_to_predict)
        self.assertIsInstance(prediction_df, pd.DataFrame)
        self.assertFalse(prediction_df.empty)

    def test_keras_functional_prediction(self):
        input_size = 16
        inputs = Input(shape=(input_size,))
        x = Dense(32, activation="relu")(inputs)
        x = Dense(32, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(5, activation="softmax")(x)

        model_to_predict = keras.Model(inputs=inputs, outputs=outputs, name="Jet Classifier")
        model_to_predict.build((None, input_size))

        estimator = MultiModelWrapper()
        estimator.load_default_models()

        prediction_df = estimator.predict(model_to_predict)
        self.assertIsInstance(prediction_df, pd.DataFrame)
        self.assertFalse(prediction_df.empty)

    def test_keras_subclassed_prediction(self):
        class JetClassifier(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = Dense(32, activation="relu")
                self.dense2 = Dense(32, activation="relu")
                self.dense3 = Dense(32, activation="relu")
                self.dense4 = Dense(5, activation="softmax")

            def build(self, input_shape):
                super().build(input_shape)

            def call(self, inputs):
                x = self.dense1(inputs)
                x = self.dense2(x)
                x = self.dense3(x)
                outputs = self.dense4(x)
                return outputs

        input_size = 16
        model_to_predict = JetClassifier()
        model_to_predict.build((None, input_size))

        estimator = MultiModelWrapper()
        estimator.load_default_models()

        prediction_df = estimator.predict(model_to_predict)
        self.assertIsInstance(prediction_df, pd.DataFrame)
        self.assertFalse(prediction_df.empty)

    def test_torch_sequential_prediction(self):
        input_size = 10
        model_to_predict = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )

        estimator = MultiModelWrapper()
        estimator.load_default_models()

        prediction_df = estimator.predict(model_to_predict)
        self.assertIsInstance(prediction_df, pd.DataFrame)
        self.assertFalse(prediction_df.empty)

    def test_torch_subclassed_prediction(self):
        class TopQuarks(torch.nn.Module):
            def __init__(self, input_size):
                super().__init__()

                self.dense1 = torch.nn.Linear(input_size, 32)
                self.relu = torch.nn.ReLU()
                self.dense2 = torch.nn.Linear(32, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, inputs):
                x = self.dense1(inputs)
                x = self.relu(x)
                x = self.dense2(x)
                outputs = self.sigmoid(x)

                return outputs

        model_to_predict = TopQuarks(input_size=10)

        estimator = MultiModelWrapper()
        estimator.load_default_models()

        prediction_df = estimator.predict(model_to_predict)
        self.assertIsInstance(prediction_df, pd.DataFrame)
        self.assertFalse(prediction_df.empty)

    def test_models_with_configs_prediction(self):
        models_to_predict = []

        class TopQuarks(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.dense1 = torch.nn.Linear(10, 32)
                self.relu = torch.nn.ReLU()
                self.dense2 = torch.nn.Linear(32, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, inputs):
                x = self.dense1(inputs)
                x = self.relu(x)
                x = self.dense2(x)
                outputs = self.sigmoid(x)

                return outputs

        models_to_predict.append(TopQuarks())

        input_size = 16
        model_to_predict = keras.Sequential(
            layers=[
                keras.layers.Input(shape=(input_size,)),
                keras.layers.Dense(32, use_bias=True),
                keras.layers.Activation("relu"),
                keras.layers.Dense(32, use_bias=True),
                keras.layers.Activation("relu"),
                keras.layers.Dense(32, use_bias=True),
                keras.layers.Activation("relu"),
                keras.layers.Dense(5, use_bias=True),
                keras.layers.Activation("softmax"),
            ],
            name="Jet Classifier",
        )
        model_to_predict.build((None, input_size))

        models_to_predict.append(model_to_predict)

        hls_configs = [
            {
                "model": {
                    "precision": "ap_fixed<8, 3>",
                    "reuse_factor": 32,
                    "strategy": strategy,
                },
                "board": board,
            }
            for board, strategy in itertools.product(["pynq-z2", "zcu102"], ["Latency", "Resource"])
        ]

        estimator = MultiModelWrapper()
        estimator.load_default_models()

        prediction_df = estimator.predict(models_to_predict, hls_configs=hls_configs)
        self.assertIsInstance(prediction_df, pd.DataFrame)
        self.assertFalse(prediction_df.empty)
