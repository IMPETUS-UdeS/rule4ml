import torch


def torch_weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def get_optimizer_from_str(optimizer_name, backend, **kwargs):
    """
    Get an optimizer from a string name.

    Args:
        optimizer_name (str): The name of the optimizer.
        backend (str): The backend to use ('keras' or 'torch').
        **kwargs: Additional arguments to pass to the optimizer.

    Returns:
        Optimizer: The optimizer instance.
    """

    optimizer_name = optimizer_name.lower()

    if backend == "keras":
        import keras

        optimizers = {
            "adam": keras.optimizers.Adam,
            "sgd": keras.optimizers.SGD,
            "rmsprop": keras.optimizers.RMSprop,
            "adagrad": keras.optimizers.Adagrad,
            "adadelta": keras.optimizers.Adadelta,
            "adamax": keras.optimizers.Adamax,
            "nadam": keras.optimizers.Nadam,
        }
    elif backend == "torch":
        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
            "adadelta": torch.optim.Adadelta,
            "adamax": torch.optim.Adamax,
            "nadam": torch.optim.NAdam,
        }
    else:
        raise ValueError(
            f"Unsupported backend \"{backend}\". Supported backends are 'keras' and 'torch'."
        )

    if optimizer_name not in optimizers:
        raise ValueError(
            f'Unsupported optimizer "{optimizer_name}". Supported optimizers are {list(optimizers.keys())}.'
        )

    return optimizers[optimizer_name](**kwargs)
