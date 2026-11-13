import torch


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
        import keras

        losses = {
            "mse": keras.losses.MeanSquaredError,
            "msle": keras.losses.MeanSquaredLogarithmicError,
            "mae": keras.losses.MeanAbsoluteError,
            "binary_crossentropy": keras.losses.BinaryCrossentropy,
            "categorical_crossentropy": keras.losses.CategoricalCrossentropy,
            "sparse_categorical_crossentropy": keras.losses.SparseCategoricalCrossentropy,
        }
    elif backend == "torch":
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
            f'Unsupported loss "{loss_name}". Supported losses are {list(losses.keys())}.'
        )

    return losses[loss_name](**kwargs)


class MSLELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
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

    def forward(self, y_pred, y_true):
        y_true = y_true.clamp_min(0.0)
        y_pred = y_pred.clamp_min(self.eps)

        if self.p == 1.0:  # Poisson limit
            loss = 2.0 * (
                y_true * torch.log((y_true + self.eps) / y_pred) - (y_true - y_pred)
            )
        elif self.p == 2.0:  # Gamma limit
            loss = 2.0 * (
                torch.log(y_pred / (y_true + self.eps)) + (y_true - y_pred) / y_pred
            )
        else:
            loss = 2.0 * (
                y_true.pow(2.0 - self.p) / ((1.0 - self.p) * (2.0 - self.p))
                - y_true * y_pred.pow(1.0 - self.p) / (1.0 - self.p)
                + y_pred.pow(2.0 - self.p) / (2.0 - self.p)
            )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
