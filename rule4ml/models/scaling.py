import math
import numpy as np
try:
    import torch
except Exception:
    torch = None

def _is_torch(x): return (torch is not None) and isinstance(x, torch.Tensor)
def _is_np(x):    return isinstance(x, np.ndarray)

class Log1pScaler:
    def __init__(self, mu=None, sigma=None):
        self.mu = {} if mu is None else mu
        self.sigma = {} if sigma is None else sigma

    def _fwd(self, y):
        if _is_torch(y):
            return torch.log1p(torch.clamp(y, min=0.0))
        elif _is_np(y):
            return np.log1p(np.maximum(y, 0.0))
        else:
            return math.log1p(max(0.0, float(y)))

    def _inv(self, z, mu, sig):
        if _is_torch(z):
            mu_t  = torch.as_tensor(mu,  dtype=z.dtype, device=z.device)
            sig_t = torch.as_tensor(max(sig, 1e-8), dtype=z.dtype, device=z.device)
            return torch.expm1(z * sig_t + mu_t)
        elif _is_np(z):
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

    def transform(self, d):
        out = {}
        for k, v in d.items():
            z  = self._fwd(v)
            mu = self.mu[k]
            sig = self.sigma[k]
            if _is_torch(z):
                mu_t  = torch.as_tensor(mu,  dtype=z.dtype, device=z.device)
                sig_t = torch.as_tensor(max(sig, 1e-8), dtype=z.dtype, device=z.device)
                out[k] = (z - mu_t) / sig_t
            elif _is_np(z):
                out[k] = (z - mu) / sig
            else:
                out[k] = (z - mu) / sig
        return out

    def inverse(self, d):
        out = {}
        for k, z in d.items():
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
        return cls(mu=config.get("mu", {}), sigma=config.get("sigma", {}))
