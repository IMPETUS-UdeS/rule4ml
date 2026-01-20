from collections import OrderedDict

import keras
import numpy as np
import tensorflow as tf

from rule4ml.parsers.utils import camel_keys_to_snake, get_closest_reuse_factor

try:
    import torch
except ImportError:
    torch = None

try:
    import onnx
except ImportError:
    onnx = None


def config_from_keras_model(model, hls_config):
    """
    _summary_

    Args:
        model (_type_): _description_
        hls_config (_type_): _description_

    Returns:
        _type_: _description_
    """

    if hasattr(model, "_build_input_shape") and model._build_input_shape is not None:  # Keras 2
        model_input_shape = model._build_input_shape
    elif hasattr(model, "_build_shapes_dict") and model._build_shapes_dict is not None:  # Keras 3
        model_input_shape = list(model._build_shapes_dict.values())[0]
    else:
        raise AttributeError(
            "Could not get model input shape. Make sure model.build() was called previously."
        )

    dummy_input = tf.random.uniform((1, *model_input_shape[1:]))
    _ = model(dummy_input)

    hls_config = camel_keys_to_snake(hls_config)

    layers_data = []
    for idx, layer in enumerate(model.layers):
        nested_activation = False

        class_name = layer.__class__.__name__
        layer_name = layer.name

        layer_config = layer.get_config()
        layer_weights = layer.get_weights()

        layer_dict = {}
        layer_dict["class_name"] = class_name
        layer_dict["name"] = layer_name

        if class_name.lower() == "activation":
            if idx == 0:
                input_shape = model_input_shape
            else:
                input_shape = layers_data[-1]["output_shape"]

        elif hasattr(
            layer, "input_shape"
        ):  # Keras 2, Sequential and Functional APIs (not including subclassing)
            input_shape = layer.input_shape
        elif hasattr(layer, "_build_input_shape"):  # Keras 2, Subclassed from keras.Model
            input_shape = layer._build_input_shape

        elif hasattr(layer, "batch_shape"):  # Keras 3, InputLayer
            input_shape = layer.batch_shape
        elif (  # Keras 3, other layers (best way we found, not documented)
            hasattr(layer, "_build_shapes_dict") and layer._build_shapes_dict is not None
        ):
            input_shape = list(layer._build_shapes_dict.values())[0]

        else:
            raise AttributeError(
                f"Could not get the input shape for layer {layer.name}. Make sure model.build() was called previously."
            )

        input_shape = np.asarray(input_shape).flatten()
        input_shape = tuple(input_shape)
        layer_dict["input_shape"] = input_shape

        if class_name == "InputLayer":  # Same input and output shape for InputLayer
            output_shape = input_shape
        elif hasattr(layer, "output_shape"):  # Keras 2
            output_shape = layer.output_shape
        else:  # Keras 3, layers other than InputLayer
            output_shape = layer.compute_output_shape(input_shape)

        output_shape = tuple(output_shape)
        layer_dict["output_shape"] = tuple(output_shape)

        # Tracking inbound layers can be useful for add/concatenate layers
        if hasattr(layer, "inbound_nodes"):
            inbound_nodes = layer.inbound_nodes
            inbound_layers = []
            for node in inbound_nodes:
                if not isinstance(node.inbound_layers, (list, tuple)):
                    inbound_layers.append(node.inbound_layers)
                else:
                    inbound_layers += node.inbound_layers
            layer_dict["inbound_layers"] = [layer.name for layer in inbound_layers]

        parameter_count = 0
        for weight_group in layer_weights:
            parameter_count += np.size(weight_group)

        parameter_count = int(parameter_count)
        layer_dict["parameters"] = parameter_count

        trainable_parameter_count = 0
        for var_group in layer.trainable_variables:
            trainable_parameter_count += np.size(var_group)

        trainable_parameter_count = int(trainable_parameter_count)
        layer_dict["trainable_parameters"] = trainable_parameter_count

        if class_name in ["Dense", "QDense"]:
            layer_dict["neurons"] = int(layer_config["units"])
            layer_dict["use_bias"] = layer_config["use_bias"]

        elif class_name in ["Conv1D", "Conv2D", "QConv1D", "QConv2D"]:
            layer_dict["channels"] = int(input_shape[-1])
            layer_dict["filters"] = int(layer_config["filters"])
            layer_dict["kernel_size"] = tuple([int(x) for x in layer_config["kernel_size"]])
            layer_dict["strides"] = tuple([int(x) for x in layer_config["strides"]])
            layer_dict["padding"] = layer_config["padding"]
            layer_dict["use_bias"] = layer_config["use_bias"]

        elif class_name == "Dropout":
            layer_dict["dropout_rate"] = layer_config["rate"]

        if "activation" in layer_config and layer_config["activation"] != "linear":
            if class_name in ["Activation", "QActivation"]:
                layer_dict["activation"] = layer_config["activation"]
            else:
                nested_activation = True

        dtype = layer_config["dtype"]
        if isinstance(dtype, dict):
            dtype = dtype["config"]["name"]
        layer_dict["dtype"] = dtype

        reuse_factor = hls_config["model"]["reuse_factor"]
        if "layer_type" in hls_config and class_name in hls_config["layer_type"]:
            reuse_factor = hls_config["layer_type"][class_name].get("reuse_factor", reuse_factor)
        if "layer_name" in hls_config and layer.name in hls_config["layer_name"]:
            reuse_factor = hls_config["layer_name"][layer.name].get("reuse_factor", reuse_factor)

        layer_dict["reuse_factor"] = reuse_factor
        if class_name in ["Dense", "QDense"]:
            n_in = np.prod([x for x in input_shape if x is not None])
            n_out = np.prod([x for x in output_shape if x is not None])
            layer_dict["reuse_factor"] = get_closest_reuse_factor(n_in, n_out, reuse_factor)
        elif class_name in ["Conv1D", "Conv2D", "QConv1D", "QConv2D"]:
            n_in = layer_dict["channels"] * np.prod(layer_dict["kernel_size"])
            n_out = layer_dict["filters"]
            layer_dict["reuse_factor"] = get_closest_reuse_factor(n_in, n_out, reuse_factor)

        layers_data.append(layer_dict)

        if (
            nested_activation
        ):  # activation function wrapped in a layer other than "(Q)Activation", example: Dense(units=32, activation="relu")
            activation_dict = {}
            activation_dict["class_name"] = "Activation"

            activation_dict["input_shape"] = output_shape
            activation_dict["output_shape"] = output_shape

            activation_dict["activation"] = layer_config["activation"]

            activation_dict["parameters"] = 0
            activation_dict["trainable_parameters"] = 0

            activation_dict["dtype"] = layer_config["dtype"]
            activation_dict["reuse_factor"] = reuse_factor

            layers_data.append(activation_dict)

    return layers_data


def keras_model_from_config(model_config):
    layer_dict = {}

    x = inputs = None
    for layer_config in model_config:
        class_name = layer_config["class_name"]
        input_shape = layer_config.get("input_shape", [])[1:]

        if class_name == "InputLayer":
            x = inputs = keras.layers.Input(shape=input_shape)

        elif class_name == "Dense":
            dense_layer = keras.layers.Dense(
                units=layer_config["neurons"],
                use_bias=layer_config["use_bias"],
                dtype=layer_config["dtype"],
            )
            layer_dict[dense_layer.name] = dense_layer
            x = dense_layer(x)

        elif class_name == "Activation":
            activation = layer_config["activation"]
            activation_layer = keras.layers.Activation(activation)
            layer_dict[activation_layer.name] = activation_layer
            x = activation_layer(x)

        elif class_name == "Dropout":
            dropout_rate = layer_config.get(
                "dropout_rate", np.random.default_rng().uniform(0.1, 0.8)
            )
            dropout_layer = keras.layers.Dropout(dropout_rate)
            layer_dict[dropout_layer.name] = dropout_layer
            x = dropout_layer(x)

        elif class_name in ["Add", "Concatenate"]:
            inbound_names = layer_config["inbound_layers"]
            inbound_layers = [layer_dict[name] for name in inbound_names]

            skip_layer = getattr(keras.layers, class_name)()
            layer_dict[skip_layer.name] = skip_layer
            x = skip_layer(inbound_layers)

        else:
            keras_layer = getattr(keras.layers, class_name)()
            layer_dict[keras_layer.name] = keras_layer
            x = keras_layer(x)

    model = keras.Model(inputs=inputs, outputs=x)
    return model


def config_from_torch_model(model, hls_config):
    """
    _summary_

    Args:
        model (_type_): _description_
        hls_config (_type_): _description_

    Returns:
        _type_: _description_
    """

    if torch is None:
        raise ImportError("Failed to import module torch.")

    torch_layer_mapping = {
        "Linear": "Dense",
        "Conv2d": "Conv2D",
        "add": "Add",
        "cat": "Concatenate",
        "BatchNorm1d": "BatchNormalization",
        "BatchNorm2d": "BatchNormalization",
    }

    hls_config = camel_keys_to_snake(hls_config)

    layers_data = OrderedDict()
    model_layers = get_torch_layers(model)
    traced_model = torch.fx.symbolic_trace(model)
    module_count = 0
    for idx, node in enumerate(traced_model.graph.nodes):
        if node.op == "output":
            continue

        layer_dict = {}

        if node.op == "placeholder":
            class_name = "InputLayer"

            reuse_factor = hls_config["model"]["reuse_factor"]
            if "layer_type" in hls_config and class_name in hls_config["layer_type"]:
                reuse_factor = hls_config["layer_type"][class_name].get(
                    "reuse_factor", reuse_factor
                )
            if "layer_name" in hls_config and node.name in hls_config["layer_name"]:
                reuse_factor = hls_config["layer_name"][node.name].get("reuse_factor", reuse_factor)

            layer_dict = {
                "class_name": class_name,
                "input_shape": (None,),
                "output_shape": (None,),
                "parameters": 0,
                "trainable_parameters": 0,
                "dtype": "float32",
                "reuse_factor": reuse_factor,
            }

        elif node.op == "call_module":
            layer = model_layers[module_count]

            class_name = layer.__class__.__name__
            mapped_class_name = torch_layer_mapping.get(class_name, class_name)
            layer_dict["class_name"] = mapped_class_name

            if mapped_class_name in ["Dense"]:
                layer_dict["input_shape"] = (None, layer.in_features)
                layer_dict["output_shape"] = (None, layer.out_features)

                use_bias = layer.bias is not None
                layer_dict["parameters"] = (layer.in_features + int(use_bias)) * layer.out_features
                layer_dict["trainable_parameters"] = layer_dict["parameters"]

                layer_dict["neurons"] = layer_dict["output_shape"][-1]
                layer_dict["use_bias"] = use_bias

                layer_dict["dtype"] = ".".join(str(layer.weight.dtype).split(".")[1:])

            elif module_count > 0:
                layers_data_values = list(layers_data.values())

                layer_dict["input_shape"] = layers_data_values[idx - 1]["output_shape"]
                layer_dict["output_shape"] = layer_dict["input_shape"]

                if mapped_class_name in ["BatchNormalization"]:
                    input_size = np.prod([x for x in layer_dict["input_shape"] if x is not None])
                    layer_dict["parameters"] = 4 * input_size
                    layer_dict["trainable_parameters"] = 2 * input_size
                else:
                    layer_dict["parameters"] = 0
                    layer_dict["trainable_parameters"] = 0

                layer_dict["dtype"] = layers_data_values[idx - 1]["dtype"]

            else:
                raise TypeError(f"Model has unexpected first layer of type {class_name}.")

            reuse_factor = hls_config["model"]["reuse_factor"]
            if "layer_type" in hls_config and mapped_class_name in hls_config["layer_type"]:
                reuse_factor = hls_config["layer_type"][mapped_class_name].get(
                    "reuse_factor", reuse_factor
                )
            if "layer_name" in hls_config and node.name in hls_config["layer_name"]:
                reuse_factor = hls_config["layer_name"][node.name].get("reuse_factor", reuse_factor)

            layer_dict["reuse_factor"] = reuse_factor
            if mapped_class_name in ["Dense"]:
                n_in = np.prod([x for x in layer_dict["input_shape"] if x is not None])
                n_out = np.prod([x for x in layer_dict["output_shape"] if x is not None])

                layer_dict["reuse_factor"] = get_closest_reuse_factor(n_in, n_out, reuse_factor)

            module_count += 1

        elif node.op == "call_function":
            class_name = node.name.split("_")
            if class_name[-1].isdigit():
                class_name = class_name[:-1]
            class_name = "_".join(class_name)
            layer_dict["class_name"] = class_name

            layers_data_values = list(layers_data.values())

            input_shapes = [
                layers_data[input_node.name]["output_shape"] for input_node in node.all_input_nodes
            ]
            layer_dict["input_shape"] = input_shapes

            if node.name == "cat":
                dim = node.kwargs["dim"]
                output_shape = list(input_shapes[0])
                output_shape[dim] = sum([x[dim] for x in input_shapes])

                layer_dict["output_shape"] = tuple(output_shape)
            else:
                layer_dict["output_shape"] = input_shapes[0]

            layer_dict["parameters"] = 0
            layer_dict["trainable_parameters"] = 0

            layer_dict["dtype"] = layers_data_values[idx - 1]["dtype"]
            layer_dict["reuse_factor"] = reuse_factor

            layer_dict["class_name"] = torch_layer_mapping.get(class_name, class_name)

        layers_data[node.name] = layer_dict

    layers_data = list(layers_data.values())

    model_input_shape = layers_data[1]["input_shape"]
    layers_data[0]["input_shape"] = model_input_shape
    layers_data[0]["output_shape"] = model_input_shape
    layers_data[0]["dtype"] = layers_data[1]["dtype"]

    return layers_data


def config_from_onnx_model(model, hls_config):
    """_summary_

    Args:
        model (_type_): _description_
        hls_config (_type_): _description_

    Raises:
        ImportError: _description_
    """

    if onnx is None:
        raise ImportError("Failed to import module onnx.")

    layers_data = []
    return layers_data


def get_torch_layers(model):
    layers = []

    model_children = list(model.children())
    if model_children == []:
        return [model]

    for child in model_children:
        layers += get_torch_layers(child)

    return layers
