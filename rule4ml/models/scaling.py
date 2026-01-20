import math

import numpy as np

try:
    import torch
except Exception:
    torch = None


def _is_torch(x):
    return (torch is not None) and isinstance(x, torch.Tensor)


class BaseScaler:
    def transform(self, target_row, input_row):
        raise NotImplementedError()

    def inverse(self, target_row, input_row):
        raise NotImplementedError()

    def to_config(self):
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError()


class BoardScaler(BaseScaler):
    def __init__(self, available_resources: dict, board_map: dict[str, int]):
        self.available_resources = {}
        self.board_map = board_map
        for board, resources in available_resources.items():
            board = self._get_board_name(board)
            self.available_resources[board] = {}
            for name, value in resources.items():
                self.available_resources[board][f"{name.lower()}"] = float(value)

    def _get_board_name(self, board):
        if isinstance(board, (int, float)):
            board = int(board)
            for name, idx in self.board_map.items():
                if idx == board:
                    return name.lower()
        elif isinstance(board, str):
            return board.lower()
        return board

    def transform(self, target_row, input_row):
        out = {}
        board_name = input_row.get("board")
        board_name = self._get_board_name(board_name)
        if board_name in self.available_resources:
            available = self.available_resources[board_name]
            for k, v in target_row.items():
                out[k] = v / available.get(k.lower(), 1.0)
        return out

    def inverse(self, target_row, input_row):
        out = {}
        board_name = input_row.get("board")
        board_name = self._get_board_name(board_name)
        if board_name in self.available_resources:
            available = self.available_resources[board_name]
            for k, z in target_row.items():
                out[k] = z * available.get(k.lower(), 1.0)
        return out

    def to_config(self):
        return {"available_resources": self.available_resources, "board_map": self.board_map}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BoardPercentScaler(BaseScaler):
    def __init__(self, available_resources: dict, board_map: dict[str, int]):
        self.available_resources = {}
        self.board_map = board_map
        for board, resources in available_resources.items():
            board = self._get_board_name(board)
            self.available_resources[board] = {}
            for name, value in resources.items():
                self.available_resources[board][f"{name.lower()}"] = float(value)

    def _get_board_name(self, board):
        try:
            board = int(board)
            for name, idx in self.board_map.items():
                if idx == board:
                    return name.lower()
        except Exception:
            if isinstance(board, str):
                return board.lower()
            return board

    def transform(self, target_row, input_row):
        out = {}
        board_name = input_row.get("board")
        board_name = self._get_board_name(board_name)
        if board_name in self.available_resources:
            available = self.available_resources[board_name]
            for k, v in target_row.items():
                if k.lower() in available:
                    out[k] = v / available[k.lower()] * 100.0
        return out

    def inverse(self, target_row, input_row):
        out = {}
        board_name = input_row.get("board")
        board_name = self._get_board_name(board_name)
        if board_name in self.available_resources:
            available = self.available_resources[board_name]
            for k, z in target_row.items():
                if k.lower() in available:
                    out[k] = z * available[k.lower()] / 100.0
        else:
            print("Board not found in available resources:", board_name)
        return out

    def to_config(self):
        return {"available_resources": self.available_resources, "board_map": self.board_map}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Log1pScaler(BaseScaler):
    def __init__(self, mu=None, sigma=None):
        self.mu = {} if mu is None else mu
        self.sigma = {} if sigma is None else sigma

    def _fwd(self, y):
        if _is_torch(y):
            return torch.log1p(torch.clamp(y, min=0.0))
        elif isinstance(y, np.ndarray):
            return np.log1p(np.maximum(y, 0.0))
        else:
            return math.log1p(max(0.0, float(y)))

    def _inv(self, z, mu, sig):
        if _is_torch(z):
            mu_t = torch.as_tensor(mu, dtype=z.dtype, device=z.device)
            sig_t = torch.as_tensor(max(sig, 1e-8), dtype=z.dtype, device=z.device)
            return torch.expm1(z * sig_t + mu_t)
        elif isinstance(z, np.ndarray):
            return np.expm1(z * sig + mu)
        else:
            return math.expm1(float(z) * sig + mu)

    def fit(self, dict_list):
        from collections import defaultdict

        by_key = defaultdict(list)
        for d in dict_list:
            for k, v in d.items():
                by_key[k].append(self._fwd(v))
        for k, arr in by_key.items():
            arr = np.asarray(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            self.mu[k] = 0.0 if arr.size == 0 else float(arr.mean())
            self.sigma[k] = 1.0 if arr.size == 0 else float(arr.std() + 1e-8)

    def transform(self, target_row, input_row):
        out = {}
        for k, v in target_row.items():
            z = self._fwd(v)
            mu = self.mu[k]
            sig = self.sigma[k]
            if _is_torch(z):
                mu_t = torch.as_tensor(mu, dtype=z.dtype, device=z.device)
                sig_t = torch.as_tensor(max(sig, 1e-8), dtype=z.dtype, device=z.device)
                out[k] = (z - mu_t) / sig_t
            elif isinstance(z, np.ndarray):
                out[k] = (z - mu) / sig
            else:
                out[k] = (z - mu) / sig
        return out

    def inverse(self, target_row, input_row):
        out = {}
        for k, z in target_row.items():
            mu = self.mu[k]
            sig = self.sigma[k]
            if z is None:
                out[k] = np.nan
            else:
                out[k] = self._inv(z, mu, sig)
        return out

    def to_config(self):
        return {"mu": self.mu, "sigma": self.sigma}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
