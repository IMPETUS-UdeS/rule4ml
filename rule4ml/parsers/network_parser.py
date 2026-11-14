import numpy as np
import torch

from rule4ml.parsers.utils import (
    camel_keys_to_snake,
    ensure_input_layer,
    get_closest_reuse_factor,
    get_reuse_factor,
)

try:
    import onnx
except ImportError:
    onnx = None


def config_from_keras_model(model, hls_config):
    """Parse a Keras model into per-layer configs with hls4ml reuse factors."""

    import tensorflow as tf

    if getattr(model, "_build_input_shape", None) is not None:  # Keras 2
        model_input_shape = model._build_input_shape
    else:  # Keras 3
        model_input_shape = _input_shape(model)

    if model_input_shape is None:
        raise AttributeError(
            "Could not get model input shape. Make sure model.build() was called previously."
        )
    if any(isinstance(dim, (list, tuple)) for dim in model_input_shape):
        raise NotImplementedError("Models with multiple inputs are not supported.")

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
        elif hasattr(
            layer, "_build_input_shape"
        ):  # Keras 2, Subclassed from keras.Model
            input_shape = layer._build_input_shape

        elif hasattr(layer, "batch_shape"):  # Keras 3, InputLayer
            input_shape = layer.batch_shape
        else:  # Keras 3, other layers
            input_shape = _input_shape(layer)
            if input_shape is None:
                raise AttributeError(
                    f"Could not get the input shape for layer {layer.name}. Make sure model.build() was called previously."
                )

        layer_dict["input_shape"] = tuple(np.asarray(input_shape).flatten())

        if class_name == "InputLayer":  # Same input and output shape for InputLayer
            output_shape = layer_dict["input_shape"]
        elif hasattr(layer, "output_shape"):  # Keras 2
            output_shape = layer.output_shape
        else:  # Keras 3, merge layers take the list of input shapes
            output_shape = layer.compute_output_shape(input_shape)

        input_shape = layer_dict["input_shape"]
        output_shape = tuple(output_shape)
        layer_dict["output_shape"] = output_shape

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
            layer_dict["kernel_size"] = tuple(
                [int(x) for x in layer_config["kernel_size"]]
            )
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

        reuse_factor = get_reuse_factor(hls_config, layer_dict, layer.name)
        layer_dict["reuse_factor"] = reuse_factor

        if class_name in ["Dense", "QDense"]:
            n_in = np.prod([x for x in input_shape if x is not None])
            n_out = np.prod([x for x in output_shape if x is not None])
            layer_dict["reuse_factor"] = get_closest_reuse_factor(
                n_in, n_out, reuse_factor
            )
        elif class_name in ["Conv1D", "Conv2D", "QConv1D", "QConv2D"]:
            n_in = layer_dict["channels"] * np.prod(layer_dict["kernel_size"])
            n_out = layer_dict["filters"]
            layer_dict["reuse_factor"] = get_closest_reuse_factor(
                n_in, n_out, reuse_factor
            )

        layers_data.append(layer_dict)

        # Activation function wrapped in a layer other than (Q)Activation, example: Dense(units=32, activation="relu")
        if nested_activation:
            activation_dict = {}
            activation_dict["class_name"] = "Activation"
            activation_dict["name"] = f"{layer_name}_{layer_config['activation']}"

            activation_dict["input_shape"] = output_shape
            activation_dict["output_shape"] = output_shape

            activation_dict["activation"] = layer_config["activation"]

            activation_dict["parameters"] = 0
            activation_dict["trainable_parameters"] = 0

            activation_dict["dtype"] = layer_config["dtype"]
            activation_dict["reuse_factor"] = get_reuse_factor(
                hls_config, activation_dict, activation_dict["name"]
            )

            layers_data.append(activation_dict)

    # Only Sequential configs name the InputLayer missing from model.layers
    if isinstance(model, tf.keras.Sequential):
        input_name = model.get_config()["layers"][0]["config"]["name"]
        return ensure_input_layer(layers_data, hls_config, input_name)

    return ensure_input_layer(layers_data, hls_config)


def _input_shape(layer):
    import tensorflow as tf

    try:  # Symbolic graph
        inputs = tf.nest.flatten(layer.input)
    except (AttributeError, ValueError):  # No graph, shapes recorded at build
        build_shapes = getattr(layer, "_build_shapes_dict", None) or {}
        return next(iter(build_shapes.values()), None)

    shapes = [tuple(tensor.shape) for tensor in inputs]
    return shapes[0] if len(shapes) == 1 else shapes


def keras_model_from_config(model_config):
    import keras

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
    """Parse a Torch model into per-layer configs with hls4ml reuse factors."""

    hls_config = camel_keys_to_snake(hls_config)
    traced_model = torch.fx.symbolic_trace(model)

    layers_data = {}
    for node in traced_model.graph.nodes:
        if node.op == "output":
            continue

        module, operation = None, None
        if node.op == "call_module":
            module = traced_model.get_submodule(node.target)
            operation = module.__class__.__name__
        elif node.op == "call_function":
            operation = getattr(node.target, "__name__", str(node.target))
        elif node.op == "call_method":
            operation = node.target
            supported = ["relu", "sigmoid", "tanh", "softmax", "flatten", "view"]
            if operation not in supported:
                raise ValueError(f"Unsupported Torch method: {operation}")
        elif node.op != "placeholder":
            raise ValueError(f"Unsupported Torch operation: {node.op} {node.target}")

        if node.op == "placeholder":
            layer_dict = {
                "class_name": "InputLayer",
                "input_shape": (None,),
                "output_shape": (None,),
                "parameters": 0,
                "trainable_parameters": 0,
                "dtype": "float32",
            }

        elif operation == "Linear":
            use_bias = module.bias is not None
            parameters = (module.in_features + int(use_bias)) * module.out_features
            layer_dict = {
                "class_name": "Dense",
                "input_shape": (None, module.in_features),
                "output_shape": (None, module.out_features),
                "parameters": parameters,
                "trainable_parameters": parameters,
                "neurons": module.out_features,
                "use_bias": use_bias,
                "dtype": ".".join(str(module.weight.dtype).split(".")[1:]),
            }

        else:
            if operation in ["cat", "concat", "concatenate"]:
                inputs = node.args[0] if node.args else node.kwargs["tensors"]
            else:
                inputs = node.all_input_nodes

            if any(input_node.op == "placeholder" for input_node in inputs):
                raise TypeError(
                    f"Model has unexpected first layer of type {operation}."
                )

            input_shapes = [
                layers_data[input_node.name]["output_shape"] for input_node in inputs
            ]
            layer_dict = {
                "input_shape": input_shapes[0],
                "output_shape": input_shapes[0],
                "parameters": 0,
                "trainable_parameters": 0,
                "dtype": layers_data[inputs[0].name]["dtype"],
            }

            if operation.lower() in ["relu", "sigmoid", "tanh", "softmax"]:
                if operation.lower() == "softmax":
                    if module is not None:
                        dim = module.dim
                    else:
                        dim = node.kwargs.get(
                            "dim", node.args[1] if len(node.args) > 1 else None
                        )

                    rank = len(input_shapes[0])
                    if dim is None:  # Torch's implicit softmax dimension
                        dim = 0 if rank == 3 else 1
                    if dim % rank == 0:
                        raise ValueError(
                            "Softmax along the batch dimension is not supported."
                        )

                layer_dict["class_name"] = "Activation"
                layer_dict["activation"] = operation.lower()

            elif operation in ["cat", "concat", "concatenate"]:
                dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
                dim %= len(input_shapes[0])
                if dim == 0:
                    raise ValueError(
                        "Concatenation along the batch dimension is not supported."
                    )
                if len(input_shapes) != 2:
                    raise ValueError("hls4ml concatenates exactly two tensors.")

                output_shape = list(input_shapes[0])
                output_shape[dim] = sum(shape[dim] for shape in input_shapes)

                layer_dict["class_name"] = "Concatenate"
                layer_dict["input_shape"] = input_shapes
                layer_dict["output_shape"] = tuple(output_shape)

            elif operation in ["Flatten", "flatten", "view"]:
                input_shape = input_shapes[0]
                if operation == "view":
                    view_shape = node.args[1:]
                    if isinstance(view_shape[0], (list, tuple)):
                        view_shape = view_shape[0]

                    size = int(np.prod(input_shape[1:]))
                    if list(view_shape) != [-1, size]:
                        raise ValueError(
                            f"Unsupported view {tuple(view_shape)}, "
                            f"only view(-1, {size}) is."
                        )

                    output_shape = (None, size)

                else:
                    if module is not None:
                        start, end = module.start_dim, module.end_dim
                    else:
                        args = node.args[1:]
                        start = node.kwargs.get("start_dim", args[0] if args else 0)
                        end = node.kwargs.get(
                            "end_dim", args[1] if len(args) > 1 else -1
                        )

                    start, end = start % len(input_shape), end % len(input_shape)
                    if start == 0:
                        raise ValueError(
                            "Flattening the batch dimension is not supported."
                        )

                    size = int(np.prod(input_shape[start : end + 1]))
                    output_shape = (*input_shape[:start], size, *input_shape[end + 1 :])

                layer_dict["class_name"] = "Flatten"
                layer_dict["output_shape"] = output_shape

            elif operation == "add":
                layer_dict["class_name"] = "Add"
                layer_dict["input_shape"] = input_shapes

            elif operation in ["BatchNorm1d", "BatchNorm2d"]:
                input_size = np.prod([x for x in input_shapes[0] if x is not None])

                layer_dict["class_name"] = "BatchNormalization"
                layer_dict["parameters"] = 4 * input_size
                layer_dict["trainable_parameters"] = 2 * input_size

            else:
                layer_dict["class_name"] = {"Conv2d": "Conv2D"}.get(
                    operation, operation
                )

        reuse_factor = get_reuse_factor(hls_config, layer_dict, node.name)
        if layer_dict["class_name"] == "Dense":
            n_in, n_out = layer_dict["input_shape"][-1], layer_dict["output_shape"][-1]
            reuse_factor = get_closest_reuse_factor(n_in, n_out, reuse_factor)

        layer_dict["reuse_factor"] = reuse_factor
        layers_data[node.name] = layer_dict

    layers_data = list(layers_data.values())

    model_input_shape = layers_data[1]["input_shape"]
    layers_data[0]["input_shape"] = model_input_shape
    layers_data[0]["output_shape"] = model_input_shape
    layers_data[0]["dtype"] = layers_data[1]["dtype"]

    return layers_data


def config_from_onnx_model(model, hls_config):
    """Placeholder, ONNX parsing is not implemented."""

    if onnx is None:
        raise ImportError("Failed to import module onnx.")

    layers_data = []
    return layers_data
