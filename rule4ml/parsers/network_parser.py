import math

import numpy as np
import torch

from rule4ml.parsers.utils import (
    camel_keys_to_snake,
    ensure_input_layer,
    get_closest_reuse_factor,
    get_reuse_factor,
    get_weight_bits,
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

        quantizer = layer_config.get("kernel_quantizer")  # hls4ml prefers it to config
        if quantizer is None:
            weight_bits = get_weight_bits(hls_config, layer_dict, layer.name)
        elif quantizer["class_name"] in ("quantized_bits", "quantized_po2"):
            weight_bits = int(quantizer["config"]["bits"])
        elif "binary" in quantizer["class_name"]:
            weight_bits = 1
        elif "ternary" in quantizer["class_name"]:
            weight_bits = 2
        else:  # hls4ml's width for unsupported quantizers
            weight_bits = 16
        if weight_bits is not None:
            layer_dict["weight_bits"] = weight_bits

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


def config_from_torch_model(model, hls_config, input_shape=None):
    """Parse a Torch model into per-layer configs with hls4ml reuse factors.

    Args:
        model (torch.nn.Module): Model to parse.
        hls_config (dict): HLS config.
        input_shape (tuple, optional): Input shape without the batch dimension, in
            Torch's channels-first layout. Models that start with a layer other than
            Linear need it, such as CNNs. Shapes are stored channels-last, as hls4ml
            converts them.

    Returns:
        list: Layer configs in graph order.
    """

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
            shape = (None,)
            if input_shape is not None:
                shape = (None, *input_shape[1:], input_shape[0])

            layer_dict = {
                "class_name": "InputLayer",
                "input_shape": shape,
                "output_shape": shape,
                "parameters": 0,
                "trainable_parameters": 0,
                "dtype": "float32",
            }

        elif operation == "Linear":
            source_shape = layers_data[node.all_input_nodes[0].name]["output_shape"]
            if len(source_shape) > 2:
                raise NotImplementedError(
                    "Linear on inputs with more than one feature axis is not supported."
                )
            if source_shape[-1] not in [None, module.in_features]:
                raise ValueError(
                    f"{node.name} expects {module.in_features} features, "
                    f"got input shape {source_shape}."
                )

            parameters, trainable_parameters = _torch_parameter_counts(module)
            layer_dict = {
                "class_name": "Dense",
                "input_shape": (None, module.in_features),
                "output_shape": (None, module.out_features),
                "parameters": parameters,
                "trainable_parameters": trainable_parameters,
                "neurons": module.out_features,
                "use_bias": module.bias is not None,
                "dtype": ".".join(str(module.weight.dtype).split(".")[1:]),
            }

        else:
            if operation in ["cat", "concat", "concatenate"]:
                inputs = node.args[0] if node.args else node.kwargs["tensors"]
            else:
                inputs = node.all_input_nodes

            if input_shape is None and any(
                input_node.op == "placeholder" for input_node in inputs
            ):
                raise TypeError(
                    f"A model that starts with {operation} needs an input shape, "
                    "passed as (model, input_shape)."
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
                rank = len(input_shapes[0])
                dim %= rank
                if dim == 0:
                    raise ValueError(
                        "Concatenation along the batch dimension is not supported."
                    )
                if len(input_shapes) != 2:
                    raise ValueError("hls4ml concatenates exactly two tensors.")

                dim = rank - 1 if dim == 1 else dim - 1  # Torch channels go last

                output_shape = list(input_shapes[0])
                output_shape[dim] = sum(shape[dim] for shape in input_shapes)

                layer_dict["class_name"] = "Concatenate"
                layer_dict["input_shape"] = input_shapes
                layer_dict["output_shape"] = tuple(output_shape)

            elif operation in ["Flatten", "flatten", "view"]:
                tensor_shape = input_shapes[0]
                if operation == "view":
                    view_shape = node.args[1:]
                    if isinstance(view_shape[0], (list, tuple)):
                        view_shape = view_shape[0]

                    size = int(np.prod(tensor_shape[1:]))
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

                    rank = len(tensor_shape)
                    start, end = start % rank, end % rank
                    if start == 0:
                        raise ValueError(
                            "Flattening the batch dimension is not supported."
                        )
                    if rank > 2 and (start, end) != (1, rank - 1):
                        raise ValueError(
                            "Channels-last tensors only flatten from dimension 1."
                        )

                    size = int(np.prod(tensor_shape[start : end + 1]))
                    output_shape = (
                        *tensor_shape[:start],
                        size,
                        *tensor_shape[end + 1 :],
                    )

                layer_dict["class_name"] = "Flatten"
                layer_dict["output_shape"] = output_shape

            elif operation == "add":
                if len(set(input_shapes)) > 1:  # hls4ml merges equal shapes only
                    raise ValueError(f"Add of different shapes {input_shapes}.")

                layer_dict["class_name"] = "Add"
                layer_dict["input_shape"] = input_shapes

            elif operation in ["BatchNorm1d", "BatchNorm2d"]:
                parameters, trainable_parameters = _torch_parameter_counts(module)
                layer_dict["class_name"] = "BatchNormalization"
                layer_dict["parameters"] = parameters
                layer_dict["trainable_parameters"] = trainable_parameters

            elif operation in ["Conv1d", "Conv2d"]:
                if module.groups != 1:  # hls4ml reads them as depthwise
                    raise NotImplementedError("Grouped convolutions are not supported.")
                if input_shapes[0][-1] != module.in_channels:
                    raise ValueError(
                        f"{node.name} expects {module.in_channels} channels, "
                        f"got channels-last input shape {input_shapes[0]}."
                    )

                parameters, trainable_parameters = _torch_parameter_counts(module)
                padding = module.padding
                no_padding = padding == "valid" or (
                    padding != "same" and not any(padding)
                )
                spatial_shape = _spatial_output_shape(module, input_shapes[0])

                layer_dict.update(
                    {
                        "class_name": f"Conv{len(module.kernel_size)}D",
                        "output_shape": (None, *spatial_shape, module.out_channels),
                        "parameters": parameters,
                        "trainable_parameters": trainable_parameters,
                        "channels": module.in_channels,
                        "filters": module.out_channels,
                        "kernel_size": tuple(module.kernel_size),
                        "strides": tuple(module.stride),
                        "padding": "valid" if no_padding else "same",  # as hls4ml
                        "use_bias": module.bias is not None,
                    }
                )

            elif operation in ["MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d"]:
                pooling = "Max" if operation.startswith("Max") else "Average"
                spatial_shape = _spatial_output_shape(module, input_shapes[0])

                layer_dict["class_name"] = f"{pooling}Pooling{operation[-2]}D"
                layer_dict["output_shape"] = (None, *spatial_shape, input_shapes[0][-1])

            elif operation in ["Dropout", "dropout"]:
                layer_dict["class_name"] = "Dropout"

            else:
                raise ValueError(f"Unsupported Torch operation: {operation}")

        reuse_factor = get_reuse_factor(hls_config, layer_dict, node.name)
        if layer_dict["class_name"] == "Dense":
            n_in, n_out = layer_dict["input_shape"][-1], layer_dict["output_shape"][-1]
            reuse_factor = get_closest_reuse_factor(n_in, n_out, reuse_factor)
        elif layer_dict["class_name"] in ["Conv1D", "Conv2D"]:
            n_in = layer_dict["channels"] * np.prod(layer_dict["kernel_size"])
            reuse_factor = get_closest_reuse_factor(
                n_in, layer_dict["filters"], reuse_factor
            )

        layer_dict["reuse_factor"] = reuse_factor
        weight_bits = get_weight_bits(hls_config, layer_dict, node.name)
        if weight_bits is not None:
            layer_dict["weight_bits"] = weight_bits
        layers_data[node.name] = layer_dict

    layers_data = list(layers_data.values())

    if input_shape is None:  # Only Linear layers can start the model
        model_input_shape = layers_data[1]["input_shape"]
        layers_data[0]["input_shape"] = model_input_shape
        layers_data[0]["output_shape"] = model_input_shape
    layers_data[0]["dtype"] = layers_data[1]["dtype"]

    return layers_data


def _torch_parameter_counts(module):
    """Total and trainable parameters of a Torch layer, with running statistics."""

    trainable = sum(
        param.numel() for param in module.parameters() if param.requires_grad
    )
    statistics = [buffer for buffer in module.buffers() if buffer.is_floating_point()]
    total = sum(tensor.numel() for tensor in [*module.parameters(), *statistics])

    return total, trainable


def _spatial_output_shape(module, input_shape):
    """Spatial output shape of a Torch conv or pooling layer on a channels-last input."""

    rank = int(type(module).__name__[-2])  # Conv1d, MaxPool2d and so on
    if len(input_shape) != rank + 2:
        raise ValueError(
            f"{type(module).__name__} expects {rank} spatial dimensions, got "
            f"channels-last input shape {input_shape}. input_shape takes Torch's "
            "channels-first layout without the batch dimension."
        )

    spatial_shape = input_shape[1:-1]
    if module.padding == "same":
        return spatial_shape

    ceil_mode = getattr(module, "ceil_mode", False)
    padding = 0 if module.padding == "valid" else module.padding
    windows = [
        value if isinstance(value, (list, tuple)) else (value,) * rank
        for value in [
            module.kernel_size,
            module.stride,
            padding,
            getattr(module, "dilation", 1),  # Average pooling has no dilation
        ]
    ]

    output_shape = []
    for size, kernel, stride, pad, dilation in zip(
        spatial_shape, *windows, strict=True
    ):
        span = (size + 2 * pad - dilation * (kernel - 1) - 1) / stride
        length = (math.ceil(span) if ceil_mode else math.floor(span)) + 1
        if ceil_mode and (length - 1) * stride >= size + pad:  # Starts in padding
            length -= 1
        output_shape.append(length)

    return tuple(output_shape)


def config_from_onnx_model(model, hls_config):
    """Placeholder, ONNX parsing is not implemented."""

    if onnx is None:
        raise ImportError("Failed to import module onnx.")

    layers_data = []
    return layers_data
