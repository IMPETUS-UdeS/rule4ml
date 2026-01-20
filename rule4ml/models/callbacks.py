import math
import os

try:
    import torch
except ImportError:
    torch = None


class ReduceLROnPlateau:
    def __init__(
        self, monitor="val_loss", mode="min", factor=0.5, patience=5, min_delta=0.0, min_lr=1e-6
    ):
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr

        self._best = math.inf if mode == "min" else -math.inf
        self._current_lrs = []
        self._num_bad = 0

    def _has_improved(self, value) -> bool:
        if self.mode == "min":
            return value < (self._best - self.min_delta)
        else:
            return value > (self._best + self.min_delta)

    def step(self, value, wrapper, verbose=0):
        if not self._current_lrs:
            self._current_lrs = [
                param_group['lr'] for param_group in wrapper.optimizer.param_groups
            ]

        if self._has_improved(value):
            self._best = float(value)
            self._num_bad = 0
            return

        self._num_bad += 1
        if self._num_bad >= self.patience:
            for idx in range(len(wrapper.optimizer.param_groups)):
                old_lr = wrapper.optimizer.param_groups[idx]['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                if old_lr > self.min_lr:
                    wrapper.optimizer.param_groups[idx]['lr'] = new_lr
                    self._current_lrs[idx] = new_lr
                    if verbose > 0:
                        print(
                            f"[{self.__class__.__name__}] Reducing learning rate "
                            + f"from {old_lr:.6f} to {new_lr:.6f}."
                        )
            self._num_bad = 0


class EarlyStopping:
    def __init__(
        self, monitor="val_loss", mode="min", patience=10, min_delta=0.0, restore_best=True
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self._best = math.inf if mode == "min" else -math.inf
        self._num_bad = 0
        self._best_state = None

    def _has_improved(self, value) -> bool:
        if self.mode == "min":
            return value < (self._best - self.min_delta)
        else:
            return value > (self._best + self.min_delta)

    def step(self, value, wrapper, verbose=0) -> bool:
        """Returns True if training should stop."""
        if self._has_improved(value):
            self._best = float(value)
            self._num_bad = 0
            if self.restore_best:
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in wrapper.model.state_dict().items()
                }
            return False

        self._num_bad += 1
        if self._num_bad >= self.patience:
            if verbose > 0:
                print(
                    f"[{self.__class__.__name__}] Early stopping triggered. "
                    + f"Best {self.monitor}: {self._best:.6f}"
                )
            if self.restore_best and self._best_state is not None:
                wrapper.model.load_state_dict(self._best_state)
                if verbose > 0:
                    print(f"[{self.__class__.__name__}] Restored best model state.")
            return True
        return False


class ModelCheckpoint:
    def __init__(self, dirpath, monitor="val_loss", mode="min"):
        if torch is None:
            raise ImportError(
                "torch is not installed. Please install torch to use ModelCheckpoint."
            )
        assert os.path.isdir(dirpath), f"{dirpath} is not a valid directory"
        assert mode in ["min", "max"], "mode should be 'min' or 'max'"

        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode

        self._best = math.inf if mode == "min" else -math.inf

    def _has_improved(self, value) -> bool:
        if self.mode == "min":
            return value < self._best
        else:
            return value > self._best

    def step(self, value, wrapper, verbose=0):
        if self._has_improved(value):
            self._best = float(value)
            payload = {
                "best_value": self._best,
                "monitor": self.monitor,
                "model": wrapper.model.state_dict(),
                "optimizer": wrapper.optimizer.state_dict(),
            }

            os.makedirs(self.dirpath, exist_ok=True)
            filepath = os.path.join(self.dirpath, f"{wrapper.model.name}_best.pth")
            torch.save(payload, filepath)
            wrapper.save(self.dirpath)  # save weights and config separately too

            if verbose > 0:
                print(
                    f"[{self.__class__.__name__}] {self.monitor} improved to {self._best:.6f}. "
                    + f"Saved to {self.dirpath}"
                )
