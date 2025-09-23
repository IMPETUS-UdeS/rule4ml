import keras

try:
    import torch
except ImportError:
    torch = None


def torch_weights_init(m):
    if torch is None:
        raise ImportError(
            "PyTorch is not installed. Please install PyTorch to use this functionality."
        )

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
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Please install PyTorch to use this backend."
            )

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
            f"Unsupported optimizer \"{optimizer_name}\". Supported optimizers are {list(optimizers.keys())}."
        )

    return optimizers[optimizer_name](**kwargs)


def get_loss_from_str(loss_name, backend, **kwargs):
    """
    Get a loss function from a string name.

    Args:
        loss_name (str): The name of the loss function.
        backend (str): The backend to use ('keras' or 'torch').

    Returns:
        Loss function: The loss function instance.
    """

    loss_name = loss_name.lower()
    if backend == "keras":
        losses = {
            "mse": keras.losses.MeanSquaredError,
            "msle": keras.losses.MeanSquaredLogarithmicError,
            "mae": keras.losses.MeanAbsoluteError,
            "binary_crossentropy": keras.losses.BinaryCrossentropy,
            "categorical_crossentropy": keras.losses.CategoricalCrossentropy,
            "sparse_categorical_crossentropy": keras.losses.SparseCategoricalCrossentropy,
        }
    elif backend == "torch":
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Please install PyTorch to use this backend."
            )

        losses = {
            "mse": torch.nn.MSELoss,
            "msle": MSLELoss,
            "mae": torch.nn.L1Loss,
            "binary_crossentropy": torch.nn.BCELoss,
            "categorical_crossentropy": torch.nn.CrossEntropyLoss,
            "sparse_categorical_crossentropy": torch.nn.CrossEntropyLoss,
            "tweedie": TweedieLoss,
        }
    else:
        raise ValueError(
            f"Unsupported backend \"{backend}\". Supported backends are 'keras' and 'torch'."
        )

    if loss_name not in losses:
        raise ValueError(
            f"Unsupported loss \"{loss_name}\". Supported losses are {list(losses.keys())}."
        )

    return losses[loss_name](**kwargs)


class MSLELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        loss = (torch.log1p(y_pred) - torch.log1p(y_true)) ** 2

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # no reduction
            return loss


class TweedieLoss(torch.nn.Module):
    def __init__(self, p=1.5, reduction="mean", eps=1e-9):
        super().__init__()
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_true, y_pred):
        y_true = y_true.clamp_min(0.0)
        y_pred = y_pred.clamp_min(self.eps)
        if self.p == 1.0:  # Poisson limit
            loss = 2.0 * (y_true * torch.log((y_true + self.eps) / y_pred) - (y_true - y_pred))
        elif self.p == 2.0:  # Gamma limit
            loss = 2.0 * (torch.log(y_pred / (y_true + self.eps)) + (y_true - y_pred) / y_pred)
        else:
            loss = 2.0 * (
                (y_true * y_pred.pow(1.0 - self.p)) / (1.0 - self.p)
                - (y_pred.pow(2.0 - self.p)) / (2.0 - self.p)
            )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
