import keras
import numpy as np
import tensorflow as tf
from packaging import version

try:
    import torch
except ImportError:
    torch = None

# Check Keras version
if version.parse(keras.__version__).major >= 3:
    from keras import ops as kops
else:
    kops = None


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    return 200 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


class KerasParametricMAPE(tf.keras.metrics.Metric):
    def __init__(self, y_index, name=None, eps=1e-6, **kwargs):
        if name is None:
            name = f"mape_{y_index}"

        super().__init__(name=name, **kwargs)
        self.y_index = y_index
        self.eps = eps
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_i = y_true[..., self.y_index]
        y_pred_i = y_pred[..., self.y_index]

        if kops is None:
            diff = tf.abs(y_true_i - y_pred_i) / tf.maximum(tf.abs(y_true_i), self.eps)
            mape = 100.0 * diff
            self.total.assign_add(tf.reduce_sum(mape))
            self.count.assign_add(tf.cast(tf.size(mape), tf.float32))
        else:
            diff = kops.absolute(y_true_i - y_pred_i) / kops.maximum(
                kops.absolute(y_true_i), self.eps
            )
            mape = 100.0 * diff
            self.total.assign_add(kops.sum(mape))
            self.count.assign_add(kops.cast(kops.size(mape), tf.float32))

    def result(self):
        return self.total / self.count


class KerasParametricSMAPE(tf.keras.metrics.Metric):
    def __init__(self, y_index, name=None, eps=1e-6, **kwargs):
        if name is None:
            name = f"smape_{y_index}"

        super().__init__(name=name, **kwargs)
        self.y_index = y_index
        self.eps = eps
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_i = y_true[..., self.y_index]
        y_pred_i = y_pred[..., self.y_index]

        if kops is None:
            numerator = tf.abs(y_true_i - y_pred_i)
            denominator = tf.maximum(tf.abs(y_true_i) + tf.abs(y_pred_i), self.eps)
            smape = 100.0 * numerator / denominator
            self.total.assign_add(tf.reduce_sum(smape))
            self.count.assign_add(tf.cast(tf.size(smape), tf.float32))
        else:
            numerator = kops.absolute(y_true_i - y_pred_i)
            denominator = kops.maximum(kops.absolute(y_true_i) + kops.absolute(y_pred_i), self.eps)
            smape = 100.0 * numerator / denominator
            self.total.assign_add(kops.sum(smape))
            self.count.assign_add(kops.cast(kops.size(smape), tf.float32))

    def result(self):
        return self.total / self.count


class KerasParametricR2(tf.keras.metrics.Metric):
    def __init__(self, y_index, name=None, eps=1e-6, **kwargs):
        if name is None:
            name = f"r2_{y_index}"

        super().__init__(name=name, **kwargs)
        self.y_index = y_index
        self.eps = eps
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_i = y_true[..., self.y_index]
        y_pred_i = y_pred[..., self.y_index]

        if kops is None:
            mean_true = tf.reduce_mean(y_true_i, axis=-1, keepdims=True)
            ss_res = tf.reduce_sum(tf.square(y_true_i - y_pred_i), axis=-1)
            ss_tot = tf.reduce_sum(tf.square(y_true_i - mean_true), axis=-1)
            ss_tot = tf.maximum(ss_tot, self.eps)
            r2_score = 1.0 - ss_res / ss_tot
            self.total.assign_add(tf.reduce_sum(r2_score))
            self.count.assign_add(tf.cast(tf.size(r2_score), tf.float32))
        else:
            mean_true = kops.mean(y_true_i, axis=-1, keepdims=True)
            ss_res = kops.sum(kops.square(y_true_i - y_pred_i), axis=-1)
            ss_tot = kops.sum(kops.square(y_true_i - mean_true), axis=-1)
            ss_tot = kops.maximum(ss_tot, self.eps)
            r2_score = 1.0 - ss_res / ss_tot
            self.total.assign_add(kops.sum(r2_score))
            self.count.assign_add(kops.cast(kops.size(r2_score), tf.float32))

    def result(self):
        return self.total / self.count


class TorchParametricMAPE(torch.nn.Module):
    def __init__(self, y_index, name="", eps=1e-6, scaler=None, device=None):
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Please install PyTorch to use this backend."
            )

        super().__init__()
        self.y_index = y_index
        self.eps = eps
        self.name = name if name else f"mape_{y_index}"
        self.scaler = scaler

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        self.total = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.count = torch.zeros(1, dtype=torch.float32, device=self.device)

    def update(self, y_true, y_pred, keys=None):
        y_true_i = y_true[..., self.y_index]
        y_pred_i = y_pred[..., self.y_index]

        if self.scaler and keys:
            key = keys[self.y_index]
            y_true_i = self.scaler.inverse({key: y_true_i})[key]
            y_pred_i = self.scaler.inverse({key: y_pred_i})[key]

        y_true_i = y_true_i.reshape(-1)
        y_pred_i = y_pred_i.reshape(-1)

        diff = torch.abs(y_true_i - y_pred_i) / torch.clamp(torch.abs(y_true_i), min=self.eps)
        mape = 100.0 * torch.mean(diff, dim=-1)

        self.total += torch.sum(mape)
        self.count += torch.tensor(mape.numel(), dtype=torch.float32, device=self.device)

    def result(self):
        if self.count.item() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return self.total / self.count

    def reset(self):
        self.total.fill_(0)
        self.count.fill_(0)


class TorchParametricSMAPE(torch.nn.Module):
    def __init__(self, y_index, name="", eps=1e-6, scaler=None, device=None):
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Please install PyTorch to use this backend."
            )

        super().__init__()
        self.y_index = y_index
        self.eps = eps
        self.name = name if name else f"smape_{y_index}"
        self.scaler = scaler

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        self.total = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.count = torch.zeros(1, dtype=torch.float32, device=self.device)

    def update(self, y_true, y_pred, keys=None):
        y_true_i = y_true[..., self.y_index]
        y_pred_i = y_pred[..., self.y_index]

        if self.scaler and keys:
            key = keys[self.y_index]
            y_true_i = self.scaler.inverse({key: y_true_i})[key]
            y_pred_i = self.scaler.inverse({key: y_pred_i})[key]

        y_true_i = y_true_i.reshape(-1)
        y_pred_i = y_pred_i.reshape(-1)

        numerator = torch.abs(y_true_i - y_pred_i)
        denominator = torch.clamp(torch.abs(y_true_i) + torch.abs(y_pred_i), min=self.eps)
        smape = 100.0 * numerator / denominator

        self.total += torch.sum(smape)
        self.count += torch.tensor(smape.numel(), dtype=torch.float32, device=self.device)
    
    def result(self):
        if self.count.item() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return self.total / self.count

    def reset(self):
        self.total.fill_(0)
        self.count.fill_(0)


class TorchParametricR2(torch.nn.Module):
    def __init__(self, y_index, name="", eps=1e-6, scaler=None, device=None):
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Please install PyTorch to use this backend."
            )

        super().__init__()
        self.y_index = y_index
        self.eps = eps
        self.name = name if name else f"r2_{y_index}"
        self.scaler = scaler

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        self.reset()

    def update(self, y_true, y_pred, keys=None):
        y_true_i = y_true[..., self.y_index]
        y_pred_i = y_pred[..., self.y_index]

        if self.scaler and keys:
            key = keys[self.y_index]
            y_true_i = self.scaler.inverse({key: y_true_i})[key]
            y_pred_i = self.scaler.inverse({key: y_pred_i})[key]

        y_true_i = y_true_i.reshape(-1)
        y_pred_i = y_pred_i.reshape(-1)

        self.y_true_all.append(y_true_i)
        self.y_pred_all.append(y_pred_i)

    def result(self):
        y_true = torch.cat(self.y_true_all)
        y_pred = torch.cat(self.y_pred_all)

        mean_true = torch.mean(y_true)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - mean_true) ** 2)
        ss_tot = torch.clamp(ss_tot, min=self.eps)

        r2 = 1.0 - ss_res / ss_tot

        return r2

    def reset(self):
        self.y_true_all = []
        self.y_pred_all = []
