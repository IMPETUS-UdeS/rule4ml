from collections import OrderedDict

import numpy as np
import tensorflow as tf

from rule4ml.parsers.utils import get_closest_reuse_factor

try:
    import torch
except ImportError:
    torch = None

try:
    import onnx
except ImportError:
    onnx = None


def config_from_keras_model(model, reuse_factor):
    """
    _summary_

    Args:
        model (_type_): _description_
        reuse_factor (_type_): _description_

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

    layers_data = []
    for layer in model.layers:
        nested_activation = False

        class_name = layer.__class__.__name__
        layer_config = layer.get_config()
        layer_weights = layer.get_weights()

        layer_dict = {}
        layer_dict["class_name"] = class_name

        if hasattr(
            layer, "input_shape"
        ):  # Keras 2, Sequential and Functional APIs (not including subclassing)
            input_shape = layer.input_shape
        elif hasattr(layer, "_build_input_shape"):  # Keras 2, Subclassed from keras.Model
            input_shape = layer._build_input_shape

        elif hasattr(layer, "batch_shape"):  # Keras 3, InputLayer
            input_shape = layer.batch_shape
        elif hasattr(
            layer, "_build_shapes_dict"
        ):  # Keras 3, other layers (best way we found, not documented)
            input_shape = list(layer._build_shapes_dict.values())[0]

        else:
            raise AttributeError(
                f"Could not get the input shape for layer {layer.name}. Make sure model.build() was called previously."
            )

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

        if class_name == "Dense":
            layer_dict["neurons"] = int(layer_config["units"])
            layer_dict["use_bias"] = layer_config["use_bias"]

        elif class_name == "Conv2D":
            layer_dict["n_channel"] = int(input_shape[-1])
            layer_dict["n_filter"] = int(layer_config["filters"])
            layer_dict["kernel_size"] = tuple([int(x) for x in layer_config["kernel_size"]])
            layer_dict["padding"] = layer_config["padding"]
            layer_dict["use_bias"] = layer_config["use_bias"]

        if "activation" in layer_config:
            if class_name == "Activation":
                layer_dict["activation"] = layer_config["activation"]
            else:
                nested_activation = True

        dtype = layer_config["dtype"]
        if isinstance(dtype, dict):
            dtype = dtype["config"]["name"]
        layer_dict["dtype"] = dtype

        layer_dict["reuse_factor"] = reuse_factor
        if class_name in ["Dense", "Conv2D"]:
            layer_dict["reuse_factor"] = get_closest_reuse_factor(
                np.prod([x for x in input_shape if x is not None]),
                np.prod([x for x in output_shape if x is not None]),
                reuse_factor,
            )

        layers_data.append(layer_dict)

        if (
            nested_activation
        ):  # activation function wrapped in a layer other than "Activation", example: Dense(units=32, activation="relu")
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


def config_from_torch_model(model, reuse_factor):
    """
    _summary_

    Args:
        model (_type_): _description_
        reuse_factor (_type_): _description_

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
            layer_dict["class_name"] = class_name

            if class_name in ["Linear"]:
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

                if class_name in ["BatchNorm1d", "BatchNorm2d"]:
                    input_size = np.prod([x for x in layer_dict["input_shape"] if x is not None])
                    layer_dict["parameters"] = 4 * input_size
                    layer_dict["trainable_parameters"] = 2 * input_size
                else:
                    layer_dict["parameters"] = 0
                    layer_dict["trainable_parameters"] = 0

                layer_dict["dtype"] = layers_data_values[idx - 1]["dtype"]

            else:
                raise TypeError(f"Model has unexpected first layer of type {class_name}.")

            layer_dict["reuse_factor"] = reuse_factor
            if class_name in ["Linear"]:
                layer_dict["reuse_factor"] = get_closest_reuse_factor(
                    np.prod([x for x in layer_dict["input_shape"] if x is not None]),
                    np.prod([x for x in layer_dict["output_shape"] if x is not None]),
                    reuse_factor,
                )

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


def config_from_onnx_model(model, reuse_factor):
    """_summary_

    Args:
        model (_type_): _description_
        reuse_factor (_type_): _description_

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
