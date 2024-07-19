from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Dense, Dropout, Input
from keras.models import Model
from utils import FloatRange, IntRange, Power2Range


@dataclass
class GeneratorSettings:
    """
    Data class that holds parameter ranges for NNs generation (only supports FCNN so far).
    Helper Range classes are used for easy random (uniform) sampling.

    Args: (all optional, default values inside data class)
        input_range (Power2Range): Input size range.
        layer_range (IntRange): Number of layers range.
        neuron_range (Power2Range): Number of neurons range.
        output_range (IntRange): Output size range.
        dropout_range (FloatRange): Dropout strength range.

        activations (list): List of included activation functions.
        output_softmax_only (bool): Forces softmax to only be used as the last layer of the NN.

        parameter_limit (int): Hard limit to the number of parameters in each layer of the NN, regardless of neuron_range. Unlimited if <= 0.

        global_bias_probability (float): Probability to use any bias within an NN.
        global_bn_probability (float): Same for batch normalization.
        global_dropout_probability (float): Same for dropout.
        global_skip_probability (float): Same for skip connection (Add/Concatenate).

        bias_probability_func (lambda): Probability function for bias, using the current layer index.
        bn_probability_func (lambda): Ditto for batch normalization.
        dropout_probability_func (lambda): Ditto for dropout.
        skip_probability_func (lambda): Ditto for skip connection.
    """

    # Default settings, used for original dataset

    input_range: Power2Range = Power2Range(16, 1024)
    layer_range: IntRange = IntRange(2, 20)
    neuron_range: Power2Range = Power2Range(2, 4096)
    output_range: IntRange = IntRange(1, 1000)
    dropout_range: FloatRange = FloatRange(0.1, 0.8)

    activations: list = field(default_factory=lambda: [None, "relu", "tanh", "sigmoid", "softmax"])
    output_softmax_only: bool = False

    parameter_limit: int = 4096

    # Global probabilities
    global_bias_probability: float = 0.9
    global_bn_probability: float = 0.2
    global_dropout_probability: float = 0.4
    global_skip_probability: float = 0.15

    def default_bias_prob(x):
        return 0.9

    def default_bn_prob(x):
        return max(0, 0.8 - 2 ** (-x / 5))

    def default_dropout_prob(x):
        return max(0, 0.8 - 2 ** (-x / 5))

    def default_skip_prob(x):
        return max(0, 0.8 - 2 ** (-x / 5))

    # Layer-wise probablities
    bias_probability_func: Callable[[float], float] = default_bias_prob
    bn_probability_func: Callable[[float], float] = default_bn_prob
    dropout_probability_func: Callable[[float], float] = default_dropout_prob
    skip_probability_func: Callable[[float], float] = default_skip_prob


def generate_fc_layer(
    inputs,
    units,
    activation=None,
    use_bias=True,
    use_bn=False,
    dropout_rate=0.0,
    skip_inputs=None,
):
    """
    Generates a Keras Dense layer with possible activation, batchnorm, dropout and add/concatenate layers.

    Args:
        inputs (Layer): Previous layer of the network.
        units (int): Number of units in the Dense layer.
        activation (str, optional): Name of the Keras activation. Defaults to None.
        use_bias (bool, optional): Whether to use bias. Defaults to True.
        use_bn (bool, optional): Whether to add batchnorm. Defaults to False.
        dropout_rate (float, optional): Strength of dropout. No dropout is applied if 0. Defaults to 0.
        skip_inputs (Layer, optional): Layer to add/concatenate with the current Dense. Defaults to None.

    Returns:
        list: Generated layers in order starting with the Dense layer.
    """

    out_list = []
    out_list.append(Dense(units, use_bias=use_bias)(inputs))

    if skip_inputs is not None:
        skip_layer = Add() if out_list[-1].shape[1] == skip_inputs.shape[1] else Concatenate()
        out_list.append(skip_layer([skip_inputs, out_list[-1]]))

    if activation is not None:
        out_list.append(Activation(activation)(out_list[-1]))

    if use_bn:
        out_list.append(BatchNormalization()(out_list[-1]))

    if dropout_rate > 0.0:
        out_list.append(Dropout(dropout_rate)(out_list[-1]))

    return out_list


def generate_fc_network(settings: GeneratorSettings, rng=None, verbose=0):
    """
    Generates a fully connected network based on Keras functional API.
    Generation parameters are randomly sampled from the ranges in the GeneratorSettings instance.

    Args:
        settings (GeneratorSettings): Holds parameter ranges for NN generation.
        rng (Random Generator, optional): Takes a Numpy Random Generator instance. Defaults to None.
        verbose (int): Output verbosity. Defaults to 0.

    Returns:
        Model: Keras model of the fully connected NN.
    """

    if rng is None:
        rng = np.random.default_rng()

    input_list = []  # keeps track of NN layers
    input_shape = (settings.input_range.random_in_range(rng),)
    input_layer = Input(shape=input_shape)
    input_list.append(input_layer)

    # Network-level decisions
    model_bias = rng.random() < settings.global_bias_probability
    model_bn = rng.random() < settings.global_bn_probability
    model_dropout = rng.random() < settings.global_dropout_probability
    model_skip = rng.random() < settings.global_skip_probability

    # Layers generation
    n_layers = settings.layer_range.random_in_range(rng)
    for i in range(n_layers):
        x = input_list[-1]  # Get the previous layer

        use_bias = model_bias and rng.random() < settings.bias_probability_func(
            i + 1
        )  # Whether the new Dense layer uses bias
        use_bn = (
            model_bn
            and i < n_layers - 1
            and rng.random() < settings.bn_probability_func(i + 1)
            and (settings.parameter_limit <= 0 or x.shape[1] <= settings.parameter_limit // 4)
        )  # Whether BatchNorm is added after the new Dense layer

        dropout_rate = 0.0
        if (
            model_dropout
            and not use_bn
            and i < n_layers - 1
            and rng.random() < settings.dropout_probability_func(i + 1)
        ):  # Same for dropout
            dropout_rate = settings.dropout_range.random_in_range(rng)

        skip_inputs = None
        if (
            model_skip and i > 0 and rng.random() < settings.skip_probability_func(i + 1)
        ):  # Same for skip connections
            skip_inputs = input_list[rng.integers(0, high=len(input_list) - 1)]

        unit_range = settings.neuron_range if i < n_layers - 1 else settings.output_range
        range_class_name = unit_range.__class__.__name__
        units = unit_range.random_in_range(rng)  # Dense layer units
        if (
            settings.parameter_limit > 0
        ):  # Number of units might change depending on the hard parameter limit
            max_units = min(units, settings.parameter_limit // x.shape[1])
            if use_bias:  # Considering bias usage
                bias_limit = (settings.parameter_limit - max_units) // x.shape[1]
                if bias_limit <= 0:
                    max_units = 1
                else:
                    if range_class_name == Power2Range.__name__:
                        bias_limit = 2 ** int(np.log2(bias_limit))
                    max_units = min(max_units, bias_limit)
            if use_bn:  # Considering batchnorm (parameters = 4 * output_shape of Dense)
                bn_limit = (settings.parameter_limit // 4) // x.shape[1]
                if bn_limit <= 0:
                    max_units = 1
                else:
                    if range_class_name == Power2Range.__name__:
                        bn_limit = 2 ** int(np.log2(bn_limit))
                    max_units = min(max_units, bn_limit)
            if max_units != units:
                range_class = globals()[unit_range.__class__.__name__]
                units = range_class(settings.neuron_range.min, max_units).random_in_range(
                    rng
                )  # Forcing down units to a range that respects parameter_limit

        activation_choices = settings.activations
        if (
            i < n_layers - 1 and settings.output_softmax_only
        ):  # Whether to limit softmax as final output only
            activation_choices = [
                act for act in activation_choices if (act is None or act.lower() != "softmax")
            ]
        activation = rng.choice(activation_choices)  # Random activation

        # Layer gen
        input_list += generate_fc_layer(
            x, units, activation, use_bias, use_bn, dropout_rate, skip_inputs
        )

    model = Model(inputs=input_layer, outputs=input_list[-1])
    model.build([None] + list(input_shape))

    if verbose > 0:
        model.summary()

    return model


def generate_ae_network(settings: GeneratorSettings, rng=None, verbose=0):
    """
    Generates a simple autoencoder network based on Keras functional API.
    Generation parameters are randomly sampled from the ranges in the GeneratorSettings instance.

    Args:
        settings (GeneratorSettings): Holds parameter ranges for NN generation.
        rng (Random Generator, optional): Takes a Numpy Random Generator instance. Defaults to None.
        verbose (int): 0: Output verbosity. Defaults to 0.

    Returns:
        Model: Keras model of the autoencoder NN.
    """

    if rng is None:
        rng = np.random.default_rng()

    input_list = []  # keeps track of autoencoder layers
    input_shape = (settings.inout_range.random_in_range(rng),)  # output shape is the same
    dense_units = [input_shape[0]]

    input_layer = Input(shape=input_shape)
    input_list.append(input_layer)

    model_bias = rng.random() < settings.global_bias_probability

    n_enc_layers = settings.layer_range.random_in_range(rng)
    n_dec_layers = n_enc_layers  # same number of decoder layers

    for i in range(n_enc_layers):
        x = input_list[-1]

        use_bias = model_bias and rng.random() < settings.bias_probability_func(i + 1)

        unit_range = Power2Range(settings.min_neurons, min(dense_units[-1], settings.max_neurons))
        units = unit_range.random_in_range(rng)

        if settings.parameter_limit > 0:
            max_units = min(units, settings.parameter_limit // x.shape[1])
            if use_bias:
                bias_limit = (settings.parameter_limit - max_units) // x.shape[1]
                if bias_limit <= 0:
                    max_units = 1
                else:
                    bias_limit = 2 ** int(np.log2(bias_limit))
                    max_units = min(max_units, bias_limit)

            if max_units != units:
                units = Power2Range(settings.min_neurons, max_units).random_in_range(rng)

        if i < n_enc_layers - 1:
            dense_units.append(units)

        activation = rng.choice(settings.activations)
        input_list += generate_fc_layer(x, units, activation, use_bias)

    # Assume encoder/decoder symmetry
    for i in range(n_dec_layers):
        x = input_list[-1]

        units = dense_units[len(dense_units) - i - 1]
        if i < n_dec_layers - 1:
            activation = rng.choice(settings.activations)
        else:
            activation = 'softmax'

        use_bias = model_bias and rng.random() < settings.bias_probability_func(i + 1)

        input_list += generate_fc_layer(x, units, activation, use_bias)

    model = Model(inputs=input_layer, outputs=input_list[-1])
    model.build([None] + list(input_shape))

    if verbose > 0:
        model.summary()

    return model


def get_submodels(base_model, verbose=0):
    """
    Splits a Keras Model into submodels with an increasing number of layers, starting from 2 layers (input -> dense).

    Args:
        base_model (Model): Keras Model to split.
        verbose (int, optional): Output verbosity. Defaults to 0.

    Returns:
        list: All submodels in an increasing order of number of layers.
    """

    submodels = []
    sublayers = []

    base_input = base_model.input
    input_shape = base_model.input_shape
    for layer in base_model.layers[1:]:
        sublayers.append(layer.output)

        model = Model(inputs=base_input, outputs=sublayers[-1])
        model.build([None] + list(input_shape))

        if verbose > 0:
            model.summary()

        submodels.append(model)

    return submodels


if __name__ == "__main__":
    rng = np.random.default_rng(seed=None)
    nn = generate_fc_network(GeneratorSettings(output_softmax_only=True), rng, verbose=1)

    # get_submodels(nn, verbose=0)
