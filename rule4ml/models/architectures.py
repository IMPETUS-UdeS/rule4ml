import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch_geometric
from packaging import version

try:
    import keras
    import tensorflow as tf
except ModuleNotFoundError as error:
    if error.name not in ("keras", "tensorflow"):
        raise
    keras = None

if keras is not None and version.parse(keras.__version__).major >= 3:
    from keras import ops as kops
else:
    kops = None


@dataclass
class MLPSettings:
    embedding_layers: list = field(default_factory=lambda: [16, 16, 16, 16])
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
    linear_output: bool = False  # for centered targets (Log1pScaler)

    def to_config(self):
        return {
            "embedding_layers": self.embedding_layers,
            "numerical_dense_layers": self.numerical_dense_layers,
            "dense_layers": self.dense_layers,
            "dense_dropouts": self.dense_dropouts,
            "linear_output": self.linear_output,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@dataclass
class GNNSettings:
    numerical_dense_layers: list = field(default_factory=lambda: [32])
    gconv_layers: list = field(default_factory=lambda: [128, 64])

    global_embedding_layers: list = field(default_factory=lambda: [16, 16, 16, 16])
    seq_embedding_layers: list = field(default_factory=lambda: [16, 16, 16, 16])

    dense_layers: list = field(default_factory=lambda: [128, 128, 64])
    bayesian_dense: list = field(default_factory=lambda: [False, False, False])
    dense_dropouts: list = field(
        default_factory=lambda: [
            0.2,
            0.2,
        ]
    )
    bayesian_output: bool = False
    linear_output: bool = False  # for centered targets (Log1pScaler)

    def to_config(self):
        return {
            "numerical_dense_layers": self.numerical_dense_layers,
            "gconv_layers": self.gconv_layers,
            "global_embedding_layers": self.global_embedding_layers,
            "seq_embedding_layers": self.seq_embedding_layers,
            "dense_layers": self.dense_layers,
            "bayesian_dense": self.bayesian_dense,
            "dense_dropouts": self.dense_dropouts,
            "bayesian_output": self.bayesian_output,
            "linear_output": self.linear_output,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if keras is not None:

    class KerasMLP(keras.Model):
        def __init__(
            self,
            settings: MLPSettings,
            input_shape,
            output_shape,
            categorical_maps,
            name="KerasMLP",
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
                        output_dim=settings.embedding_layers[idx],
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
                if (
                    idx < len(settings.dense_dropouts)
                    and settings.dense_dropouts[idx] > 0.0
                ):
                    x = KerasMCDropout(settings.dense_dropouts[idx])(x)

            model_inputs = categorical_inputs + [numerical_inputs]
            model_outputs = keras.layers.Dense(
                output_shape[-1], activation=None if settings.linear_output else "relu"
            )(x)

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

    class KerasMCDropout(keras.layers.Dropout):
        def __init__(self, rate, **kwargs):
            """
            Expose an additional attribute to better control the call method's
            'training' argument for MC Dropout at inference time.
            """

            self.enable_mc = kwargs.pop("enable_mc", False)
            super().__init__(rate, **kwargs)

        def call(self, inputs, training=None):
            # Normal behavior during training:
            #   - when training=True (from model.fit)
            #   - when training=False, dropout would normally be off (assuming enable_mc=False).
            # During inference, if enable_mc is True, we force training=True to enable MC dropout.

            if not training and self.enable_mc:
                training = True
            return super().call(inputs, training=training)


class TorchMLP(torch.nn.Module):
    def __init__(
        self,
        settings: MLPSettings,
        input_shape,
        output_shape,
        categorical_maps,
        name="TorchMLP",
        device=None,
    ):
        super().__init__()

        self.embeddings = torch.nn.ModuleDict()
        for idx, key in enumerate(categorical_maps):
            self.embeddings[f"g{idx}_{key}_embedding"] = torch.nn.Embedding(
                num_embeddings=len(categorical_maps[key]) + 1,
                embedding_dim=settings.embedding_layers[idx],
            )

        self.numerical_layers = torch.nn.ModuleDict()
        in_dim = input_shape[-1] - len(categorical_maps)
        if in_dim > 0:
            for idx, units in enumerate(settings.numerical_dense_layers):
                self.numerical_layers[
                    f"g{idx + len(categorical_maps)}_numerical_layer"
                ] = torch.nn.Linear(in_dim, units, bias=False)
                in_dim = units

        concat_dim = sum(settings.embedding_layers[: len(categorical_maps)]) + in_dim
        self.dense_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for idx, units in enumerate(settings.dense_layers):
            self.dense_layers.append(torch.nn.Linear(concat_dim, units))
            if (
                idx < len(settings.dense_dropouts)
                and settings.dense_dropouts[idx] > 0.0
            ):
                self.dropouts.append(torch.nn.Dropout(settings.dense_dropouts[idx]))
            else:
                self.dropouts.append(None)

            concat_dim = units

        self.output_layer = torch.nn.Linear(concat_dim, output_shape[-1])

        self.name = name
        self.settings = settings

        self.global_input_keys = {
            "categorical": [name for name in self.embeddings],
            "numerical": f"g{len(categorical_maps)}_numerical_layer",
        }
        self.global_input_shape = input_shape
        self.global_categorical_maps = categorical_maps

        self.output_shape = output_shape

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device
        self.to(self.device)

    def forward(self, inputs):
        x_categorical = []
        for idx, key in enumerate(self.embeddings.keys()):
            x_categorical.append(self.embeddings[key](inputs[key]).flatten(1))

        x_numerical = inputs[self.global_input_keys["numerical"]]
        for idx, key in enumerate(self.numerical_layers.keys()):
            x_numerical = self.numerical_layers[key](x_numerical)

        x = torch.cat(x_categorical + [x_numerical], dim=-1)
        for dense, dropout in zip(self.dense_layers, self.dropouts):
            x = torch.nn.functional.relu(dense(x))
            if dropout:
                x = dropout(x)

        outputs = self.output_layer(x)
        if not self.settings.linear_output:
            outputs = torch.nn.functional.softplus(outputs)

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
    def __init__(
        self,
        settings: GNNSettings,
        global_input_shape,
        sequential_input_shape,
        output_shape,
        global_categorical_maps,
        sequential_categorical_maps,
        name="TorchGNN",
        device=None,
    ):

        super().__init__()

        self.global_embeddings = torch.nn.ModuleDict()
        for idx, key in enumerate(global_categorical_maps):
            self.global_embeddings[f"g{idx}_{key}_embedding"] = torch.nn.Embedding(
                num_embeddings=len(global_categorical_maps[key]) + 1,
                embedding_dim=settings.global_embedding_layers[idx],
            )

        self.sequential_embeddings = torch.nn.ModuleDict()
        for idx, key in enumerate(sequential_categorical_maps):
            self.sequential_embeddings[f"s{idx}_{key}_embedding"] = torch.nn.Embedding(
                num_embeddings=len(sequential_categorical_maps[key]) + 1,
                embedding_dim=settings.seq_embedding_layers[idx],
            )

        self.numerical_layers = torch.nn.ModuleDict()
        g_in_dim = max(0, global_input_shape[-1] - len(global_categorical_maps))
        if g_in_dim > 0:
            for idx, units in enumerate(settings.numerical_dense_layers):
                self.numerical_layers[
                    f"g{idx + len(global_categorical_maps)}_numerical_layer"
                ] = torch.nn.Linear(g_in_dim, units, bias=False)
                g_in_dim = units

        seq_numerical_dim = sequential_input_shape[-1] - len(
            sequential_categorical_maps
        )
        seq_embedding_dim = sum(
            settings.seq_embedding_layers[: len(sequential_categorical_maps)]
        )
        gnn_dim = seq_embedding_dim + seq_numerical_dim

        global_embedding_dim = sum(
            settings.global_embedding_layers[: len(global_categorical_maps)]
        )
        # unused, kept so v2 state dicts load
        g_token_dim = global_embedding_dim + g_in_dim
        self.g_projection = torch.nn.Sequential(
            torch.nn.Linear(g_token_dim, gnn_dim),
            torch.nn.ReLU(),
        )

        self.gconvs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for idx, units in enumerate(settings.gconv_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(gnn_dim, units),
                torch.nn.ReLU(),
                torch.nn.Linear(units, units),
            )
            self.gconvs.append(torch_geometric.nn.GINConv(nn=mlp))
            self.norms.append(torch_geometric.nn.GraphNorm(in_channels=units))
            gnn_dim = units

        self.jumping_knowledge = torch_geometric.nn.JumpingKnowledge(mode="cat")

        self.attention_pool = torch_geometric.nn.AttentionalAggregation(
            gate_nn=torch.nn.Sequential(torch.nn.Linear(sum(settings.gconv_layers), 1))
        )

        concat_dim = global_embedding_dim + g_in_dim + sum(settings.gconv_layers)
        self.dense_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for idx, units in enumerate(settings.dense_layers):
            dense_layer = (
                TorchBayesianLinear(concat_dim, units)
                if (idx < len(settings.bayesian_dense) and settings.bayesian_dense[idx])
                else torch.nn.Linear(concat_dim, units)
            )
            self.dense_layers.append(dense_layer)
            if (
                idx < len(settings.dense_dropouts)
                and settings.dense_dropouts[idx] > 0.0
            ):
                self.dropouts.append(torch.nn.Dropout(settings.dense_dropouts[idx]))
            else:
                self.dropouts.append(None)

            concat_dim = units

        self.output_layer = (
            TorchBayesianLinear(concat_dim, output_shape[-1])
            if settings.bayesian_output
            else torch.nn.Linear(concat_dim, output_shape[-1])
        )

        self.name = name
        self.settings = settings

        self.global_input_keys = {
            "categorical": [name for name in self.global_embeddings],
            "numerical": f"g{len(global_categorical_maps)}_numerical_layer",
        }
        self.global_input_shape = global_input_shape
        self.global_categorical_maps = global_categorical_maps

        self.sequential_input_keys = {
            "categorical": [name for name in self.sequential_embeddings],
            "numerical": f"s{len(sequential_categorical_maps)}_numerical_layer",
        }
        self.sequential_input_shape = sequential_input_shape
        self.sequential_categorical_maps = sequential_categorical_maps

        self.output_shape = output_shape
        self.kl_loss = 0.0

        if device is None:
            if torch.cuda.is_available() and os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ) not in [
                "",
                "-1",
            ]:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device
        self.to(self.device)

    def forward(self, inputs):
        x_global_categorical = []
        for idx, key in enumerate(self.global_embeddings.keys()):
            x_global_categorical.append(
                self.global_embeddings[key](inputs[key]).flatten(1)
            )

        x_global_numerical = inputs[self.global_input_keys["numerical"]]
        for idx, key in enumerate(self.numerical_layers.keys()):
            x_global_numerical = self.numerical_layers[key](x_global_numerical)

        seq_numerical_inputs = inputs[
            self.sequential_input_keys["numerical"]
        ]  # shape: (batch, seq_len, seq_features)
        mask = ~(seq_numerical_inputs == 0.0).all(
            dim=-1
        )  # mask to eliminate zero padding

        x_sequential_categorical = []
        for idx, key in enumerate(self.sequential_embeddings.keys()):
            x_sequential_categorical.append(
                self.sequential_embeddings[key](inputs[key]).flatten(2)
            )

        node_features = []
        edge_indices = []
        batch_vector = []
        ptr = 0
        for i in range(seq_numerical_inputs.shape[0]):
            valid_seq_numerical = seq_numerical_inputs[i][mask[i]]
            if valid_seq_numerical.size(0) == 0:
                raise ValueError(f"Record {i} of the batch has no layers")

            valid_seq_embeddings = [
                embedding[i][mask[i]] for embedding in x_sequential_categorical
            ]
            valid_features = torch.cat(
                [*valid_seq_embeddings, valid_seq_numerical], dim=-1
            )
            node_features.append(valid_features.to(self.device))

            num_nodes = valid_features.size(0)
            if num_nodes > 1:
                row = torch.arange(num_nodes - 1, dtype=torch.long)
                edge = torch.stack([row, row + 1], dim=0) + ptr
                edge_indices.append(edge.to(self.device))

            batch_vector.append(
                torch.full((num_nodes,), i, dtype=torch.long).to(self.device)
            )
            ptr += num_nodes

        x_seq = torch.cat(node_features, dim=0)
        edge_indices = (
            torch.cat(edge_indices, dim=1)
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        )
        batch = torch.cat(batch_vector, dim=0)

        if edge_indices.numel() > 0:
            edge_indices = torch_geometric.utils.to_undirected(
                edge_indices, num_nodes=ptr
            )
        edge_indices, _ = torch_geometric.utils.add_self_loops(
            edge_indices, num_nodes=ptr
        )

        xs = []
        for gconv, norm in zip(self.gconvs, self.norms):
            x_seq = gconv(x_seq, edge_index=edge_indices)
            x_seq = norm(x_seq, batch)
            x_seq = torch.nn.functional.relu(x_seq)
            xs.append(x_seq)

        x_seq = self.jumping_knowledge(xs)

        pooled_seq = self.attention_pool(x_seq, index=batch)

        self.kl_loss = 0.0
        x = torch.cat([*x_global_categorical, x_global_numerical, pooled_seq], dim=-1)
        for idx, dense in enumerate(self.dense_layers):
            x = torch.nn.functional.relu(dense(x))
            if hasattr(dense, "kl_divergence"):
                self.kl_loss += dense.kl_divergence()
            if self.dropouts[idx] is not None:
                x = self.dropouts[idx](x)

        outputs = self.output_layer(x)
        if hasattr(self.output_layer, "kl_divergence"):
            self.kl_loss += self.output_layer.kl_divergence()

        if not self.settings.linear_output:
            outputs = torch.nn.functional.softplus(outputs)

        return outputs

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
        config["settings"] = GNNSettings.from_config(config["settings"])
        return cls(**config)


class TorchBayesianLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = torch.nn.Parameter(torch.Tensor(out_features))

        # Prior N(0, prior_std^2)
        self.prior_std = prior_std
        self.prior_logvar = 2 * torch.log(torch.tensor(prior_std))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5))
        torch.nn.init.constant_(self.weight_logvar, -5)  # small initial variance

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias_mu, -bound, bound)
        torch.nn.init.constant_(self.bias_logvar, -5)

    def forward(self, input):
        # Sample weights using reparameterization trick
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps

        return torch.nn.functional.linear(input, weight, bias)

    def kl_divergence(self):
        """KL(q(w|θ) || p(w)) for both weights and biases, assuming Gaussian prior."""

        # posterior and prior variances
        w_var = torch.exp(self.weight_logvar)
        b_var = torch.exp(self.bias_logvar)
        prior_var = torch.exp(self.prior_logvar)

        # KL for each Gaussian param: 0.5 * [ (σ² + μ²)/prior_var - 1 - log(σ²/prior_var) ]
        kl_weight = (
            0.5
            * (
                (w_var + self.weight_mu**2) / prior_var
                - (self.weight_logvar - self.prior_logvar)
                - 1.0
            ).sum()
        )
        kl_bias = (
            0.5
            * (
                (b_var + self.bias_mu**2) / prior_var
                - (self.bias_logvar - self.prior_logvar)
                - 1.0
            ).sum()
        )

        return kl_weight + kl_bias
