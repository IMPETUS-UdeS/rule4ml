import itertools
import json
import os
from dataclasses import dataclass, field

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
    Layer,
    LayerNormalization,
    Masking,
    MultiHeadAttention,
)
from keras.losses import Loss
from keras.optimizers import Adam, Optimizer
from packaging import version

try:
    import torch
except ImportError:
    torch = None

try:
    import onnx
except ImportError:
    onnx = None

from rule4ml.parsers.data_parser import (
    boards_data,
    get_global_inputs,
    get_layers_data,
    to_dataframe,
)
from rule4ml.parsers.network_parser import (
    config_from_keras_model,
    config_from_onnx_model,
    config_from_torch_model,
)
from rule4ml.parsers.utils import camel_keys_to_snake, fixed_precision_to_bit_width

# Check Keras version
if version.parse(keras.__version__).major >= 3:
    from keras import ops as kops
else:
    kops = None


@dataclass
class MLPSettings:
    """
    _summary_

    Args: (all optional, default values inside data class)
        embedding_outputs (list): _description_
        numerical_dense_layers (list): _description_
        dense_layers (list): _description_
        dense_dropouts (list): _description_
    """

    # Default settings

    embedding_outputs: list = field(default_factory=lambda: [16, 16, 16, 16])
    numerical_dense_layers: list = field(default_factory=lambda: [16])
    dense_layers: list = field(default_factory=lambda: [64, 32, 64, 32])
    dense_dropouts: list = field(
        default_factory=lambda: [
            0.0,
            0.2,
            0.2,
            0.0,
        ]
    )

    def to_config(self):
        return {
            "class_name": self.__class__.__name__,
            "embedding_outputs": self.embedding_outputs,
            "numerical_dense_layers": self.numerical_dense_layers,
            "dense_layers": self.dense_layers,
            "dense_dropouts": self.dense_dropouts,
        }

    def from_config(self, config):
        self.embedding_outputs = config["embedding_outputs"]
        self.numerical_dense_layers = config["numerical_dense_layers"]
        self.dense_layers = config["dense_layers"]
        self.dense_dropouts = config["dense_dropouts"]


@dataclass
class TransformerSettings:
    """
    _summary_

    Args: (all optional, default values inside data class)
        global_dense_layers (list): _description_
        seq_dense_layers (list): _description_

        global_numerical_dense_layers (list): _description_
        layer_numerical_dense_layers (list): _description_

        num_blocks (int): _description_
        num_heads (int): _description_
        ff_dim (int): _description_
        output_dim (int): _description_
        dropout_rate (int): _description_

        embedding_outputs (list): _description_
        dense_layers (list): _description_
        dense_dropouts (list): _description_
    """

    # Default settings

    global_dense_layers: list = field(default_factory=lambda: [64, 128])
    seq_dense_layers: list = field(default_factory=lambda: [64, 128])

    global_numerical_dense_layers: list = field(default_factory=lambda: [32])
    seq_numerical_dense_layers: list = field(default_factory=lambda: [32])

    num_blocks: int = 2
    num_heads: int = 4
    ff_dim: int = 128
    output_dim: int = 64
    dropout_rate: float = 0.1

    embedding_outputs: list = field(default_factory=lambda: [16, 16, 16, 16])
    dense_layers: list = field(default_factory=lambda: [128, 128, 64])
    dense_dropouts: list = field(
        default_factory=lambda: [
            0.2,
            0.2,
        ]
    )

    def to_config(self):
        return {
            "class_name": self.__class__.__name__,
            "global_dense_layers": self.global_dense_layers,
            "seq_dense_layers": self.seq_dense_layers,
            "global_numerical_dense_layers": self.global_numerical_dense_layers,
            "seq_numerical_dense_layers": self.seq_numerical_dense_layers,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "embedding_outputs": self.embedding_outputs,
            "dense_layers": self.dense_layers,
            "dense_dropouts": self.dense_dropouts,
        }

    def from_config(self, config):
        self.global_dense_layers = config["global_dense_layers"]
        self.seq_dense_layers = config["seq_dense_layers"]
        self.global_numerical_dense_layers = config["global_numerical_dense_layers"]
        self.seq_numerical_dense_layers = config["seq_numerical_dense_layers"]
        self.num_blocks = config["num_blocks"]
        self.num_heads = config["num_heads"]
        self.ff_dim = config["ff_dim"]
        self.output_dim = config["output_dim"]
        self.dropout_rate = config["dropout_rate"]
        self.embedding_outputs = config["embedding_outputs"]
        self.dense_layers = config["dense_layers"]
        self.dense_dropouts = config["dense_dropouts"]


@dataclass
class TrainSettings:
    """
    _summary_

    Args: (all optional, default values inside data class)
        num_epochs (int): _description_
        batch_size (int): _description_
        learning_rate (float): _description_

        optimizer (Optimizer): _description_
        loss_function (str, Loss): _description_
        metrics (list): _description_
    """

    # Default settings

    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4

    optimizer: Optimizer = Adam
    loss_function: (str, Loss) = "mae"
    metrics: list = field(default_factory=lambda: ["mape"])


class MultiModelEstimator:
    """
    _summary_
    """

    def __init__(self):
        self._models = {}

        self._default_boards = list(boards_data.keys())
        self._default_strategies = ["Latency", "Resource"]
        self._default_precisions = ["ap_fixed<2, 1>", "ap_fixed<8, 3>", "ap_fixed<16, 6>"]
        self._reuse_factors = [1, 2, 4, 8, 16, 32, 64]

    def add_model_wrapper(self, model_wrapper):
        key = "-".join([x.upper() for x in model_wrapper.output_labels])
        self._models[key] = model_wrapper

    def load_default_models(self):
        base_path = os.path.join(os.path.dirname(__file__), "weights")
        default_paths = [
            os.path.join(base_path, "BRAM_MLP"),
            os.path.join(base_path, "DSP_MLP"),
            os.path.join(base_path, "FF_MLP"),
            os.path.join(base_path, "LUT_MLP"),
            os.path.join(base_path, "CYCLES_MLP"),
        ]
        for path in default_paths:
            model_wrapper = ModelWrapper()
            model_wrapper.load(f"{path}_config.json", f"{path}.weights.h5")
            self.add_model_wrapper(model_wrapper)

    def predict(self, models_to_predict, hls_configs=None, verbose=0):
        """
        _summary_

        Args:
            models_to_predict (_type_): _description_
            hls_configs (_type_, optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
            Exception: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
        """

        if not isinstance(models_to_predict, (tuple, list)):
            models_to_predict = [models_to_predict]

        if hls_configs is None:
            hls_configs = [
                {
                    "model": {
                        "strategy": strategy,
                        "precision": precision,
                        "reuse_factor": reuse_factor,
                    },
                    "board": board,
                }
                for board, strategy, precision, reuse_factor in itertools.product(
                    self._default_boards,
                    self._default_strategies,
                    self._default_precisions,
                    self._reuse_factors,
                )
            ]
        else:
            if not isinstance(hls_configs, (tuple, list)):
                if not isinstance(hls_configs, dict):
                    raise TypeError(
                        "hls_configs argument expects a dictionary or list of dictionaries. See docstring example for correct formatting."
                    )
                hls_configs = [hls_configs]

        for idx in range(len(hls_configs)):
            hls_configs[idx] = camel_keys_to_snake(hls_configs[idx])

            if hls_configs[idx]["board"].lower() not in boards_data:
                raise ValueError(
                    f"Board {hls_configs[idx]['board']} is not currently supported. Supported boards: {boards_data.keys()}"
                )

        outputs = []
        for model_to_predict, hls_config in itertools.product(models_to_predict, hls_configs):
            model_name = model_to_predict.__class__.__name__
            if hasattr(model_to_predict, "name"):
                model_name = model_to_predict.name

            outputs.append(
                {
                    "Model": model_name,
                    "Board": hls_config["board"],
                    "Strategy": hls_config["model"]["strategy"],
                    "Precision": hls_config["model"]["precision"],
                    "Reuse Factor": hls_config["model"]["reuse_factor"],
                }
            )

        for key, estimator_model in self._models.items():
            estimator_predictions = estimator_model.predict(
                models_to_predict, hls_configs, verbose=verbose
            ).astype(float)

            target_labels = key.split("-")
            for idx, label in enumerate(target_labels):
                if "cycles" not in label.lower():
                    target_labels[idx] = f"{label} (%)"

            for idx, model_predictions in enumerate(estimator_predictions):
                outputs[idx].update(
                    {target_labels[k]: model_predictions[k] for k in range(len(model_predictions))}
                )

        outputs_df = pd.DataFrame(outputs).round(2)
        if not outputs_df.empty:
            sorted_boards = sorted(
                outputs_df["Board"].unique(), key=lambda x: self._default_boards.index(x)
            )
            outputs_df["Board"] = pd.Categorical(
                outputs_df["Board"], categories=sorted_boards, ordered=True
            )

            sorted_precisions = sorted(
                outputs_df["Precision"].unique(), key=lambda x: fixed_precision_to_bit_width(x)[0]
            )
            outputs_df["Precision"] = pd.Categorical(
                outputs_df["Precision"], categories=sorted_precisions, ordered=True
            )

        return outputs_df


class ModelWrapper:
    """
    _summary_
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.global_categorical_inputs = []
        self.sequential_categorical_inputs = []

        self.global_numerical_labels = []
        self.sequential_numerical_labels = []

        self.global_categorical_maps = {}
        self.sequential_categorical_maps = {}

        self.global_input_shape = ()
        self.sequential_input_shape = ()

        self.output_labels = []
        self.output_shape = ()

        if hasattr(self, "model") and self.model is not None:
            del self.model

        self.model = None
        self.model_name = ""
        self.model_settings = None

        self.dataset = None

    def build_mlp_model(
        self,
        mlp_settings: MLPSettings,
        input_shape,
        output_shape,
        categorical_maps,
        model_name="MLP_estimator",
        verbose=0,
        reset=True,
    ):
        if not isinstance(categorical_maps, dict):
            raise TypeError(
                "categorical_maps expects a dictionary of ordinal mappings for categorical inputs."
            )

        if reset:
            self.reset()

        squeeze_op = tf.squeeze if kops is None else kops.squeeze

        categorical_inputs = []
        embeddings = []

        for idx, key in enumerate(categorical_maps):
            # Keras 3 reorders the Input Tensors by name (tree.flatten), causing shape mismatches in fit() call
            # [g(lobal)|s(equential)]{idx} prefix helps keep the order deterministic
            input_layer = Input(shape=(1,), name=f"g{idx}_{key}_input")

            embedding_layer = squeeze_op(
                Embedding(
                    input_dim=len(categorical_maps[key]) + 1,
                    output_dim=mlp_settings.embedding_outputs[idx],
                    name=f"{key}_embedding",
                )(input_layer),
                axis=-2,
            )

            categorical_inputs.append(input_layer)
            embeddings.append(embedding_layer)

        self.global_categorical_inputs = categorical_inputs
        self.global_categorical_maps = categorical_maps

        numerical_inputs = Input(
            shape=(input_shape[-1] - len(categorical_inputs),),
            name=f"g{len(categorical_maps)}_numerical_inputs",
        )
        self.global_input_shape = input_shape

        numerical_outputs = numerical_inputs
        for idx, units in enumerate(mlp_settings.numerical_dense_layers):
            numerical_outputs = Dense(units, use_bias=False, name=f"numerical_dense_{idx}")(
                numerical_outputs
            )

        concat_op = tf.concat if kops is None else kops.concatenate
        x = concat_op([*embeddings, numerical_outputs], axis=-1)

        for idx, units in enumerate(mlp_settings.dense_layers):
            x = Dense(units, activation="relu")(x)
            if idx < len(mlp_settings.dense_dropouts) - 1:
                x = Dropout(mlp_settings.dense_dropouts[idx])(x)

        model_inputs = categorical_inputs + [numerical_inputs]
        model_outputs = Dense(output_shape[-1], activation="relu")(x)
        self.output_shape = output_shape

        self.model = keras.Model(inputs=model_inputs, outputs=model_outputs, name=model_name)
        self.model.build([None] + list(input_shape))

        if verbose > 0:
            self.model.summary()

        self.model_name = model_name
        self.model_settings = mlp_settings

    def build_transformer_model(
        self,
        transformer_settings: TransformerSettings,
        global_input_shape,
        sequential_input_shape,
        output_shape,
        global_categorical_maps,
        sequential_categorical_maps,
        model_name="Transformer_estimator",
        verbose=0,
        reset=True,
    ):
        if not isinstance(global_categorical_maps, dict):
            raise TypeError(
                "global_categorical_maps expects a dictionary of ordinal mappings for global categorical inputs."
            )

        if not isinstance(sequential_categorical_maps, dict):
            raise TypeError(
                "sequential_categorical_maps expects a dictionary of ordinal mappings for layer-wise categorical inputs."
            )

        if reset:
            self.reset()

        squeeze_op = tf.squeeze if kops is None else kops.squeeze
        concat_op = tf.concat if kops is None else kops.concatenate

        # Global inputs (global model/hls info)
        global_categorical_inputs = []
        global_embeddings = []

        for idx, key in enumerate(global_categorical_maps):
            input_layer = Input(shape=(1,), name=f"g{idx}_{key}_input")

            embedding_layer = squeeze_op(
                Embedding(
                    input_dim=len(global_categorical_maps[key]) + 1,
                    output_dim=transformer_settings.embedding_outputs[idx],
                    name=f"{key}_embedding",
                )(input_layer),
                axis=-2,
            )

            global_categorical_inputs.append(input_layer)
            global_embeddings.append(embedding_layer)

        self.global_categorical_inputs = global_categorical_inputs
        self.global_categorical_maps = global_categorical_maps

        global_numerical_inputs = Input(
            shape=(global_input_shape[-1] - len(global_categorical_inputs),),
            name=f"g{len(global_categorical_maps)}_numerical_inputs",
        )
        self.global_input_shape = global_input_shape

        global_numerical_outputs = global_numerical_inputs
        for idx, units in enumerate(transformer_settings.global_numerical_dense_layers):
            global_numerical_outputs = Dense(units, use_bias=False)(global_numerical_outputs)

        concat_global_inputs = concat_op([*global_embeddings, global_numerical_outputs], axis=-1)

        global_outputs = concat_global_inputs
        for idx, units in enumerate(transformer_settings.global_dense_layers):
            global_outputs = Dense(units, activation="relu")(global_outputs)

        # Sequential inputs (layer-wise info)
        sequential_categorical_inputs = []
        sequential_embeddings = []

        for idx, key in enumerate(sequential_categorical_maps):
            input_layer = Input(
                shape=(
                    None,
                    1,
                ),
                name=f"s{idx}_{key}_input",
            )

            embedding_layer = squeeze_op(
                Embedding(
                    input_dim=len(sequential_categorical_maps[key]) + 1,
                    output_dim=transformer_settings.embedding_outputs[0],
                    name=f"{key}_embedding",
                )(input_layer),
                axis=-2,
            )

            sequential_categorical_inputs.append(input_layer)
            sequential_embeddings.append(embedding_layer)

        self.sequential_categorical_inputs = sequential_categorical_inputs
        self.sequential_categorical_maps = sequential_categorical_maps

        sequential_numerical_inputs = Input(
            shape=(None, sequential_input_shape[-1] - len(sequential_categorical_inputs)),
            name=f"s{len(sequential_categorical_maps)}_numerical_inputs",
        )
        self.sequential_input_shape = sequential_input_shape

        sequential_numerical_outputs = sequential_numerical_inputs
        for idx, units in enumerate(transformer_settings.seq_numerical_dense_layers):
            sequential_numerical_outputs = Dense(units, use_bias=False)(
                sequential_numerical_outputs
            )

        concat_seq_inputs = concat_op(
            [*sequential_embeddings, sequential_numerical_outputs], axis=-1
        )
        masked_inputs = Masking(mask_value=0.0)(concat_seq_inputs)

        x = masked_inputs
        for idx, units in enumerate(transformer_settings.seq_dense_layers):
            x = Dense(units, activation="relu")(x)

        x = TransformerBlock(
            x.shape[-1],
            transformer_settings.num_heads,
            transformer_settings.ff_dim,
            transformer_settings.output_dim,
            transformer_settings.dropout_rate,
        )(x)

        x = Concatenate(axis=-1)([x, global_outputs])

        for idx, units in enumerate(transformer_settings.dense_layers):
            x = Dense(units, activation="relu")(x)
            if (
                idx < len(transformer_settings.dense_dropouts) - 1
                and transformer_settings.dense_dropouts[idx] > 0.0
            ):
                x = Dropout(transformer_settings.dense_dropouts[idx])(x)

        # The same order as the model inputs should be maintained, else Keras 3 breaks
        inputs = (
            global_categorical_inputs
            + [global_numerical_inputs]
            + sequential_categorical_inputs
            + [sequential_numerical_inputs]
        )
        outputs = Dense(output_shape[-1], activation="relu")(x)
        self.output_shape = output_shape

        self.model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
        self.model.build([None] + list(global_input_shape) + list(sequential_input_shape))

        if verbose > 0:
            self.model.summary()

        self.model_name = model_name
        self.model_settings = transformer_settings

    def build_dataset(
        self,
        inputs_df,
        targets_df,
        batch_size,
        val_ratio=0.2,
        train_repeats=1,
        shuffle=True,
        verbose=0,
    ):
        if self.model is None:
            raise Exception("A model needs to be built before creating a Tensorflow dataset.")

        input_dict = self.build_inputs(inputs_df)

        if verbose > 0:
            for key, value in input_dict.items():
                print(f"{key}: {value.shape}")

        self.output_labels = targets_df.columns.values
        self.dataset = tf.data.Dataset.from_tensor_slices((input_dict, targets_df.values))
        if val_ratio > 0.0:
            val_len = int(len(self.dataset) * val_ratio)
            val_data = self.dataset.take(val_len)
            val_data = val_data.batch(batch_size)
            train_data = self.dataset.skip(val_len).repeat(train_repeats)
            if shuffle:
                train_data = train_data.shuffle(len(self.dataset) - val_len)
            train_data = train_data.batch(batch_size)

            self.train_data = train_data
            self.val_data = val_data
        else:
            train_data = self.dataset.repeat(train_repeats)
            if shuffle:
                train_data = train_data.shuffle(len(train_data))
            train_data = train_data.batch(batch_size)

            self.train_data = train_data
            self.val_data = None

        self.batch_size = batch_size

    def build_inputs(self, inputs_df):
        input_dict = {}

        for layer, label in zip(
            self.global_categorical_inputs, self.global_categorical_maps.keys()
        ):
            input_dict[layer.name] = inputs_df[label].values

        global_numerical_inputs_df = inputs_df.drop(
            self.global_categorical_maps.keys(), axis=1
        ).select_dtypes(
            exclude=[object]
        )  # excluding nested "sequential_inputs" Series
        self.global_numerical_labels = global_numerical_inputs_df.columns.values

        input_dict[f"g{len(self.global_categorical_maps)}_numerical_inputs"] = (
            global_numerical_inputs_df.values
        )

        if "sequential_inputs" in inputs_df:
            seq_inputs_series = inputs_df["sequential_inputs"]

            if not seq_inputs_series.empty:
                for layer, label in zip(
                    self.sequential_categorical_inputs, self.sequential_categorical_maps.keys()
                ):
                    input_dict[layer.name] = seq_inputs_series.apply(lambda df: df[label]).values

                seq_numerical_inputs = seq_inputs_series.apply(
                    lambda df: df.drop(self.sequential_categorical_maps.keys(), axis=1)
                    .select_dtypes(exclude=[object])
                    .values
                ).values
                seq_numerical_inputs = np.asarray([arr for arr in seq_numerical_inputs])

                self.sequential_numerical_labels = (
                    seq_inputs_series.iloc[0]
                    .drop(self.sequential_categorical_maps.keys(), axis=1)
                    .select_dtypes(exclude=[object])
                    .columns.values
                )

                input_dict[f"s{len(self.sequential_categorical_maps)}_numerical_inputs"] = (
                    seq_numerical_inputs
                )

        return input_dict

    def fit(self, train_settings: TrainSettings, callbacks=[], verbose=0):
        if self.dataset is None:
            raise Exception("A dataset needs to be built before training the model.")

        self.model.compile(
            optimizer=train_settings.optimizer(learning_rate=train_settings.learning_rate),
            loss=train_settings.loss_function,
            metrics=train_settings.metrics,
        )

        return self.model.fit(
            self.train_data,
            epochs=train_settings.num_epochs,
            batch_size=train_settings.batch_size,
            validation_data=self.val_data,
            callbacks=callbacks,
            verbose=verbose,
        )

    def predict(self, models_to_predict, hls_configs, verbose=0):
        if not isinstance(models_to_predict, (tuple, list)):
            raise TypeError("models_to_predict expects a list of keras, torch or onnx models.")

        if not isinstance(hls_configs, (tuple, list)):
            raise TypeError("hls_configs expects a list of HLS configuration dictionaries.")

        global_inputs = []
        sequential_inputs = []

        for model_to_predict, hls_config in itertools.product(models_to_predict, hls_configs):
            if isinstance(model_to_predict, keras.Model):
                model_config = config_from_keras_model(
                    model_to_predict, hls_config["model"]["reuse_factor"]
                )

            elif torch is not None and isinstance(model_to_predict, torch.nn.Module):
                model_config = config_from_torch_model(
                    model_to_predict, hls_config["model"]["reuse_factor"]
                )

            elif onnx is not None and isinstance(model_to_predict, onnx.ModelProto):
                model_config = config_from_onnx_model(
                    model_to_predict, hls_config["model"]["reuse_factor"]
                )

            else:
                if torch is not None:
                    if onnx is not None:
                        raise TypeError(f"Unsupported model type: {type(model_to_predict)}")
                    else:
                        raise ImportError("Failed to import module onnx.")
                elif onnx is not None:
                    raise ImportError("Failed to import module torch.")

            global_inputs.append(get_global_inputs(model_config, hls_config))
            sequential_inputs.append(get_layers_data(model_config))

        inputs_df = to_dataframe(
            global_inputs=global_inputs,
            sequential_inputs=sequential_inputs,
            global_categorical_maps=self.global_categorical_maps,
            sequential_categorical_maps=self.sequential_categorical_maps,
            meta_data=[],
            targets=[],
        )

        return self.predict_from_df(inputs_df, verbose=verbose)

    def predict_from_df(self, inputs_df, verbose=0):
        feature_labels = list(self.global_numerical_labels) + list(
            self.global_categorical_maps.keys()
        )
        if len(self.sequential_numerical_labels) > 0:
            sequential_labels = list(self.sequential_numerical_labels) + list(
                self.sequential_categorical_maps.keys()
            )
            inputs_df["sequential_inputs"] = inputs_df["sequential_inputs"].apply(
                lambda x: x[sequential_labels]
            )
            feature_labels += ["sequential_inputs"]

        inputs_df = inputs_df[feature_labels]
        inputs = self.build_inputs(inputs_df)

        prediction = self.model.predict(inputs, 0, verbose=verbose)
        return prediction

    def to_config(self):
        return {
            "global_numerical_labels": list(self.global_numerical_labels),
            "sequential_numerical_labels": list(self.sequential_numerical_labels),
            "global_categorical_maps": self.global_categorical_maps,
            "sequential_categorical_maps": self.sequential_categorical_maps,
            "global_input_shape": self.global_input_shape,
            "sequential_input_shape": self.sequential_input_shape,
            "output_labels": list(self.output_labels),
            "output_shape": self.output_shape,
            "model_name": self.model_name,
            "model_settings": self.model_settings.to_config(),
        }

    def from_config(self, config):
        self.global_numerical_labels = config["global_numerical_labels"]
        self.sequential_numerical_labels = config["sequential_numerical_labels"]
        self.global_categorical_maps = config["global_categorical_maps"]
        self.sequential_categorical_maps = config["sequential_categorical_maps"]

        self.global_input_shape = config["global_input_shape"]
        self.sequential_input_shape = config["sequential_input_shape"]

        self.output_labels = config["output_labels"]
        self.output_shape = config["output_shape"]

        self.model_name = config["model_name"]

        settings_class = config["model_settings"]["class_name"]
        self.model_settings = globals()[settings_class]()
        self.model_settings.from_config(config["model_settings"])

    def save(self, save_dir):
        """
        Save the class config and model weights.

        Args:
            save_dir (_type_): _description_
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        config = self.to_config()
        config_path = os.path.join(save_dir, f"{self.model_name}_config.json")
        with open(config_path, "w") as config_file:
            json.dump(config, config_file)

        weights_path = os.path.join(save_dir, f"{self.model_name}.weights.h5")
        self.model.save_weights(weights_path)

    def load(self, config_path, weights_path):
        """
        Load saved class config and model weights.

        Args:
            config_path (_type_): _description_
            weights_path (_type_): _description_
        """

        if not config_path.endswith(".json"):
            raise ValueError("Config file name must end with .json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file does not exist: {config_path}")

        if not weights_path.endswith(".weights.h5"):
            raise ValueError("Weights file name must end with .weights.h5")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

        self.reset()

        with open(config_path) as json_file:
            # Load the class config
            config = json.load(json_file)
            self.from_config(config)

            # Build the model
            if isinstance(self.model_settings, MLPSettings):
                self.build_mlp_model(
                    self.model_settings,
                    self.global_input_shape,
                    self.output_shape,
                    self.global_categorical_maps,
                    self.model_name,
                    reset=False,
                )
            elif isinstance(self.model_settings, TransformerSettings):
                self.build_transformer_model(
                    self.model_settings,
                    self.global_input_shape,
                    self.sequential_input_shape,
                    self.output_shape,
                    self.global_categorical_maps,
                    self.sequential_categorical_maps,
                    self.model_name,
                    reset=False,
                )

            # Load the model weights
            self.model.load_weights(weights_path)


class TransformerBlock(Layer):
    """
    _summary_

    Args:
        embed_dim (_type_): _description_
        num_heads (_type_): _description_
        ff_dim (_type_): _description_
        output_dim (_type_): _description_
        dropout_rate (_type_): _description_. Defaults to 0.1
    """

    def __init__(self, embed_dim, num_heads, ff_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        # self.global_avg_pooling = GlobalAveragePooling1D()
        self.out_dense = Dense(output_dim)

        self.supports_masking = True  # needed to pass the mask to downstream layers

    def call(self, inputs, training=False, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)

        attn_output = self.att(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        outputs = self.layernorm2(out1 + ffn_output)

        # outputs = self.global_avg_pooling(outputs) # avg of hidden states
        outputs = outputs[:, -1, :]  # last hidden state

        outputs = self.out_dense(outputs)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return mask

    # Positional encoding doesn't seem to be important
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates
