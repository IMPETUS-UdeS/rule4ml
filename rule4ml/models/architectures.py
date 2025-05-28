from dataclasses import dataclass, field

import keras
import numpy as np
import tensorflow as tf
from packaging import version

try:
    import torch
except ImportError:
    torch = None

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
            "embedding_outputs": self.embedding_outputs,
            "numerical_dense_layers": self.numerical_dense_layers,
            "dense_layers": self.dense_layers,
            "dense_dropouts": self.dense_dropouts,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@dataclass
class GNNSettings:
    """
    _summary_

    Args: (all optional, default values inside data class)
    """

    # Default settings

    def to_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class KerasMLP(keras.Model):
    def __init__(
        self, settings: MLPSettings, input_shape, output_shape, categorical_maps, name="KerasMLP"
    ):
        squeeze_op = tf.squeeze if kops is None else kops.squeeze

        categorical_inputs = []
        embeddings = []

        for idx, key in enumerate(categorical_maps):
            # Keras 3 reorders the Input Tensors by name (tree.flatten), causing shape mismatches in fit() call
            # [g(lobal)|s(equential)]{idx} prefix helps keep the order deterministic
            input_layer = keras.layers.Input(shape=(1,), name=f"g{idx}_{key}_input")

            embedding_layer = squeeze_op(
                keras.layers.Embedding(
                    input_dim=len(categorical_maps[key]) + 1,
                    output_dim=settings.embedding_outputs[idx],
                    name=f"{key}_embedding",
                )(input_layer),
                axis=-2,
            )

            categorical_inputs.append(input_layer)
            embeddings.append(embedding_layer)

        numerical_inputs = keras.layers.Input(
            shape=(input_shape[-1] - len(categorical_inputs),),
            name=f"g{len(categorical_maps)}_numerical_inputs",
        )

        numerical_outputs = numerical_inputs
        for idx, units in enumerate(settings.numerical_dense_layers):
            numerical_outputs = keras.layers.Dense(
                units, use_bias=False, name=f"numerical_dense_{idx}"
            )(numerical_outputs)

        concat_op = tf.concat if kops is None else kops.concatenate
        x = concat_op([*embeddings, numerical_outputs], axis=-1)

        for idx, units in enumerate(settings.dense_layers):
            x = keras.layers.Dense(units, activation="relu")(x)
            if idx < len(settings.dense_dropouts) - 1:
                x = keras.layers.Dropout(settings.dense_dropouts[idx])(x)

        model_inputs = categorical_inputs + [numerical_inputs]
        model_outputs = keras.layers.Dense(output_shape[-1], activation="relu")(x)

        super().__init__(inputs=model_inputs, outputs=model_outputs, name=name)
        super().build([None] + list(input_shape))

        self.settings = settings

        self.global_input_keys = {
            "categorical": [layer.name for layer in categorical_inputs],
            "numerical": numerical_inputs.name,
        }
        self.global_input_shape = input_shape
        self.global_categorical_maps = categorical_maps

    def to_config(self):
        return {
            "name": self.name,
            "settings": self.settings.to_config(),
            "input_shape": self.global_input_shape,
            "output_shape": self.output_shape,
            "categorical_maps": self.global_categorical_maps,
        }

    @classmethod
    def from_config(cls, config):
        config["settings"] = MLPSettings.from_config(config["settings"])
        return cls(**config)


class KerasTransformer(keras.Model):
    def __init__(
        self,
        settings,
        global_input_shape,
        sequential_input_shape,
        output_shape,
        global_categorical_maps,
        sequential_categorical_maps,
        name="KerasTransformer",
    ):
        squeeze_op = tf.squeeze if kops is None else kops.squeeze
        concat_op = tf.concat if kops is None else kops.concatenate

        # Global inputs (global model/hls info)
        global_categorical_inputs = []
        global_embeddings = []

        for idx, key in enumerate(global_categorical_maps):
            input_layer = keras.layers.Input(shape=(1,), name=f"g{idx}_{key}_input")

            embedding_layer = squeeze_op(
                keras.layers.Embedding(
                    input_dim=len(global_categorical_maps[key]) + 1,
                    output_dim=settings.embedding_outputs[idx],
                    name=f"{key}_embedding",
                )(input_layer),
                axis=-2,
            )

            global_categorical_inputs.append(input_layer)
            global_embeddings.append(embedding_layer)

        global_numerical_inputs = keras.layers.Input(
            shape=(global_input_shape[-1] - len(global_categorical_maps),),
            name=f"g{len(global_categorical_maps)}_numerical_inputs",
        )

        global_numerical_outputs = global_numerical_inputs
        for idx, units in enumerate(settings.global_numerical_dense_layers):
            global_numerical_outputs = keras.layers.Dense(units, use_bias=False)(
                global_numerical_outputs
            )

        concat_global_inputs = concat_op([*global_embeddings, global_numerical_outputs], axis=-1)

        global_outputs = concat_global_inputs
        for idx, units in enumerate(settings.global_dense_layers):
            global_outputs = keras.layers.Dense(units, activation="relu")(global_outputs)

        # Sequential inputs (layer-wise info)
        sequential_categorical_inputs = []
        sequential_embeddings = []

        for idx, key in enumerate(sequential_categorical_maps):
            input_layer = keras.layers.Input(
                shape=(
                    None,
                    1,
                ),
                name=f"s{idx}_{key}_input",
            )

            embedding_layer = squeeze_op(
                # embedding_layer = (
                keras.layers.Embedding(
                    input_dim=len(sequential_categorical_maps[key]) + 1,
                    output_dim=settings.embedding_outputs[0],
                    name=f"{key}_embedding",
                    # )(input_layer)
                )(input_layer),
                axis=-2,
            )

            sequential_categorical_inputs.append(input_layer)
            sequential_embeddings.append(embedding_layer)

        sequential_numerical_inputs = keras.layers.Input(
            shape=(None, sequential_input_shape[-1] - len(sequential_categorical_inputs)),
            name=f"s{len(sequential_categorical_maps)}_numerical_inputs",
        )

        sequential_numerical_outputs = sequential_numerical_inputs
        for idx, units in enumerate(settings.seq_numerical_dense_layers):
            sequential_numerical_outputs = keras.layers.Dense(units, use_bias=False)(
                sequential_numerical_outputs
            )

        concat_seq_inputs = concat_op(
            [*sequential_embeddings, sequential_numerical_outputs], axis=-1
        )
        masked_inputs = keras.layers.Masking(mask_value=0.0)(concat_seq_inputs)

        x = masked_inputs
        for idx, units in enumerate(settings.seq_dense_layers):
            x = keras.layers.Dense(units, activation="relu")(x)

        x = TransformerBlock(
            x.shape[-1],
            settings.num_heads,
            settings.ff_dim,
            settings.output_dim,
            settings.dropout_rate,
        )(x)

        x = keras.layers.Concatenate(axis=-1)([x, global_outputs])

        for idx, units in enumerate(settings.dense_layers):
            x = keras.layers.Dense(units, activation="relu")(x)
            if idx < len(settings.dense_dropouts) - 1 and settings.dense_dropouts[idx] > 0.0:
                x = keras.layers.Dropout(settings.dense_dropouts[idx])(x)

        # The same order as the model inputs should be maintained, else Keras 3 breaks
        inputs = (
            global_categorical_inputs
            + [global_numerical_inputs]
            + sequential_categorical_inputs
            + [sequential_numerical_inputs]
        )
        outputs = keras.layers.Dense(output_shape[-1], activation="relu")(x)

        super().__init__(inputs=inputs, outputs=outputs, name=name)
        super().build([None] + list(global_input_shape) + list(sequential_input_shape))

        self.settings = settings

        self.global_input_keys = {
            "categorical": [layer.name for layer in global_categorical_inputs],
            "numerical": global_numerical_inputs.name,
        }
        self.global_input_shape = global_input_shape
        self.global_categorical_maps = global_categorical_maps

        self.sequential_input_keys = {
            "categorical": [layer.name for layer in sequential_categorical_inputs],
            "numerical": sequential_numerical_inputs.name,
        }
        self.sequential_input_shape = sequential_input_shape
        self.sequential_categorical_maps = sequential_categorical_maps

    def to_config(self):
        return {
            "name": self.name,
            "settings": self.settings.to_config(),
            "global_input_shape": self.global_input_shape,
            "sequential_input_shape": self.sequential_input_shape,
            "output_shape": self.output_shape,
            "global_categorical_maps": self.global_categorical_maps,
            "sequential_categorical_maps": self.sequential_categorical_maps,
        }

    @classmethod
    def from_config(cls, config):
        config["settings"] = TransformerSettings.from_config(config["settings"])
        return cls(**config)


class TorchMLP(torch.nn.Module):
    def __init__(
        self, settings: MLPSettings, input_shape, output_shape, categorical_maps, name="TorchMLP"
    ):
        if torch is None:
            raise ImportError("Torch is not available. Please install torch to use this class.")

        super().__init__()

        self.embeddings = torch.nn.ModuleDict()
        for idx, key in enumerate(categorical_maps):
            self.embeddings[f"g{idx}_{key}_embedding"] = torch.nn.Embedding(
                num_embeddings=len(categorical_maps[key]) + 1,
                embedding_dim=settings.embedding_outputs[idx],
            )

        self.numerical_layers = torch.nn.ModuleDict()
        in_dim = input_shape[-1] - len(categorical_maps)
        for idx, units in enumerate(settings.numerical_dense_layers):
            self.numerical_layers[f"g{idx + len(categorical_maps)}_numerical_layer"] = (
                torch.nn.Linear(in_dim, units, bias=False)
            )
            in_dim = units

        concat_dim = sum(settings.embedding_outputs) + in_dim
        self.dense_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for idx, units in enumerate(settings.dense_layers):
            self.dense_layers.append(torch.nn.Linear(concat_dim, units))
            if idx < len(settings.dense_dropouts) - 1 and settings.dense_dropouts[idx] > 0.0:
                self.dropouts.append(torch.nn.Dropout(settings.dense_dropouts[idx]))
            else:
                self.dropouts.append(None)

            concat_dim = units

        self.output_layer = torch.nn.Linear(concat_dim, output_shape[-1])

        self.name = name
        self.settings = settings

        self.global_input_keys = {
            "categorical": [name for name in self.embeddings.keys()],
            "numerical": f"g{len(categorical_maps)}_numerical_layer",
        }
        self.global_input_shape = input_shape
        self.global_categorical_maps = categorical_maps
        self.output_shape = output_shape

    def forward(self, inputs):
        x_categorical = []
        for idx, key in enumerate(self.embeddings.keys()):
            x_categorical.append(self.embeddings[key](inputs[key]).squeeze(1))

        x_numerical = inputs[self.global_input_keys["numerical"]]
        for idx, key in enumerate(self.numerical_layers.keys()):
            x_numerical = self.numerical_layers[key](x_numerical)

        x = torch.cat(x_categorical + [x_numerical], dim=-1)
        for dense, dropout in zip(self.dense_layers, self.dropouts):
            x = torch.nn.functional.relu(dense(x))
            if dropout:
                x = dropout(x)

        outputs = torch.nn.functional.relu(self.output_layer(x))
        return outputs

    def to_config(self):
        return {
            "name": self.name,
            "settings": self.settings.to_config(),
            "input_shape": self.global_input_shape,
            "output_shape": self.output_shape,
            "categorical_maps": self.global_categorical_maps,
        }

    @classmethod
    def from_config(cls, config):
        config["settings"] = MLPSettings.from_config(config["settings"])
        return cls(**config)


class TorchGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        pass

    def to_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerBlock(keras.layers.Layer):
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
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

        self.global_avg_pooling = keras.layers.GlobalAveragePooling1D()
        self.out_dense = keras.layers.Dense(output_dim)

        self.supports_masking = True  # needed to pass the mask to downstream layers

    def call(self, inputs, training=False, mask=None, global_pool="last"):
        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)

        attn_output = self.att(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        outputs = self.layernorm2(out1 + ffn_output)

        if global_pool == "avg":
            outputs = self.global_avg_pooling(outputs)
        elif global_pool == "last":
            outputs = outputs[:, -1, :]  # last hidden state
        else:
            raise ValueError(
                f"global_pool must be one of [\"avg\", \"last\"], but got \"{global_pool}\""
            )

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
