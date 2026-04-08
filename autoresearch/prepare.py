import hashlib
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from rule4ml.models.wrappers import BaseModelWrapper, TorchModelWrapper
from rule4ml.parsers.data_parser import json_to_df, read_from_json

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "datasets", "huggingface", "wa-hls4ml")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
HLS4ML_DIR = os.path.join(ROOT_DIR, "hls4ml")
HLS4ML_TAGS = {"0.8.1": "v0.8.1", "1.1.0": "v1.1.0"}

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TIME_BUDGET = 3600  # wall-clock training seconds (excludes startup and evaluation)
ALL_TARGETS = ["bram", "dsp", "ff", "lut", "cycles", "interval"]

EVERYTHING_SEED = 42
JSON_BATCH_SIZE = 256
JSON_MAX_WORKERS = 8

FORCE_CPU = False

def set_seed(seed: int = EVERYTHING_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def get_split_json_patterns(split: str) -> list[str]:
    split_dir = os.path.join(DATA_DIR, split)
    return [
        os.path.join(split_dir, f"{split}_2_20_merged.json"),
        os.path.join(split_dir, f"{split}_2layer_merged.json"),
        os.path.join(split_dir, f"{split}_3layer_merged.json"),
        os.path.join(split_dir, f"{split}_conv1d_merged.json"),
        os.path.join(split_dir, f"{split}_conv2d_merged.json"),
        os.path.join(split_dir, f"{split}_latency_merged.json"),
        os.path.join(split_dir, f"{split}_resource_merged.json"),
        os.path.join(split_dir, "*exemplar_models.json"),
    ]

def load_split_from_json(
    split: str,
    global_categorical_maps: dict,
    sequential_categorical_maps: dict,
    normalize: bool,
) -> pd.DataFrame:
    json_data = read_from_json(
        get_split_json_patterns(split),
        batch_size=JSON_BATCH_SIZE,
        max_workers=JSON_MAX_WORKERS,
    )
    df = json_to_df(
        json_data,
        global_categorical_maps,
        sequential_categorical_maps,
        normalize=normalize,
        max_workers=JSON_MAX_WORKERS,
    )
    df[ALL_TARGETS] = df[ALL_TARGETS].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=ALL_TARGETS).reset_index(drop=True)

def load_tensor_cache(cache_path: str) -> dict | None:
    if not os.path.exists(cache_path):
        return None

    cached = torch.load(cache_path, weights_only=False, map_location="cpu")
    required_input_splits = ("train", "val")
    required_target_splits = ("train", "val")
    if (
        "inputs" not in cached
        or "targets" not in cached
        or any(split not in cached["inputs"] for split in required_input_splits)
        or any(split not in cached["targets"] for split in required_target_splits)
    ):
        print("Tensor cache is missing required splits/targets; rebuilding from JSON.")
        return None

    split_lengths = {
        split: len(next(iter(cached["inputs"][split].values())))
        for split in required_input_splits
    }
    cached["meta"] = cached.get("meta", {})
    cached["meta"]["split_lengths"] = split_lengths
    return cached

def build_inputs_df(
    df: pd.DataFrame,
    global_labels: List[str],
    sequential_labels: List[str]
) -> pd.DataFrame:
    inputs_df = df[global_labels].copy()
    inputs_df["sequential_inputs"] = df["sequential_inputs"].apply(
        lambda x: x[sequential_labels]
    )
    return inputs_df

def build_input_tensors(wrapper: TorchModelWrapper, inputs_df: pd.DataFrame, device: torch.device) -> dict:
    input_dict = wrapper.build_inputs(inputs_df)
    cat_keys = set(
        wrapper.global_input_keys["categorical"]
        + wrapper.sequential_input_keys["categorical"]
    )
    return {
        k: torch.tensor(v, dtype=torch.long if k in cat_keys else torch.float32).to(device)
        for k, v in input_dict.items()
    }

def make_dataloader(
    input_tensors: dict,
    target_tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> torch.utils.data.DataLoader:
    key_list = list(input_tensors.keys())
    dataset = torch.utils.data.TensorDataset(
        *[input_tensors[k] for k in key_list], target_tensor
    )

    def collate_fn(batch):
        batch = list(zip(*batch))
        inputs = {key: torch.stack(batch[i]) for i, key in enumerate(key_list)}
        targets = torch.stack(batch[-1])
        return inputs, targets

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, pin_memory=pin_memory,
    )

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(
        np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
    )

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(r2_score(y_true, y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def tensor_cache_key(feature_cols: list, global_cat_keys, seq_cat_keys, normalize) -> str:
    """
    Compute the 12-hex cache key shared by predict() and train.py's cache writer.
    Defining it once here ensures both sides always produce the same filename.
    """
    h = hashlib.sha256(repr(
        list(feature_cols)
        + sorted(global_cat_keys)
        + sorted(seq_cat_keys)
        + [normalize]
    ).encode())
    return h.hexdigest()[:12]

def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    best_loss: float
):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "best_loss": best_loss,
    }, path)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        if checkpoint.get("optimizer_state_dict") is None:
            raise ValueError("Checkpoint is missing optimizer state dict.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if lr_scheduler is not None:
        if checkpoint.get("lr_scheduler_state_dict") is None:
            raise ValueError("Checkpoint is missing lr_scheduler state dict.")
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["best_loss"]

# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

@torch.inference_mode()
def predict(
    wrapper: BaseModelWrapper,
    inputs_df: pd.DataFrame,
) -> np.ndarray:
    """
    Inference helper. Strips any leaked target columns from inputs_df before
    running the model. Returns a (N, output_size) numpy array.

    Available to the agent for use during training (e.g. validation monitoring),
    but also called internally by evaluate() so the agent cannot override how
    the final test-set predictions are produced.
    """
    safe_inputs = inputs_df.drop(
        columns=[c for c in ALL_TARGETS if c in inputs_df.columns]
    )

    wrapper.model.eval()
    return wrapper.predict_from_df(safe_inputs)

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(
    wrappers: List[Tuple],
    test_inputs_df: pd.DataFrame,
    test_targets_df: pd.DataFrame,
) -> dict:
    """
    Locked evaluation. Calls predict() internally for each trained model group
    with use_cache=False, so final test predictions are always rebuilt from the
    provided test features.

    Args:
        wrappers: List of (wrapper, group_targets) tuples, one per trained group.
            group_targets is a list of target name strings (e.g. ['bram', 'dsp']).
        test_inputs_df: Feature DataFrame for the test split (targets stripped automatically).
        test_targets_df: DataFrame with one column per target in ALL_TARGETS.
        device:         Torch device for inference.

    Returns:
        Flat dict with keys:
            mean_smape            — mean SMAPE across all six targets (primary metric)
            mean_r2               — mean R² across all six targets
            smape_<target>        — per-target SMAPE
            r2_<target>           — per-target R²
            rmse_<target>         — per-target RMSE
    """

    predictions = {}
    for wrapper, group_targets in wrappers:
        preds = predict(wrapper, test_inputs_df)
        for i, target in enumerate(group_targets):
            if target in predictions:
                raise ValueError(f"Duplicate prediction target during evaluation: {target}")
            predictions[target] = preds[:, i]

    missing_targets = [t for t in ALL_TARGETS if t not in predictions]
    if missing_targets:
        raise ValueError(
            "Evaluation requires predictions for all targets. Missing: "
            + ", ".join(missing_targets)
        )

    trained = list(ALL_TARGETS)
    smape_vals, r2_vals, rmse_vals = {}, {}, {}
    for t in trained:
        y_true = test_targets_df[t].values
        y_pred = np.asarray(predictions[t]).ravel()
        smape_vals[t] = smape(y_true, y_pred)
        r2_vals[t] = r2(y_true, y_pred)
        rmse_vals[t] = rmse(y_true, y_pred)

    metrics = {
        "mean_smape": float(np.mean(list(smape_vals.values()))),
        "mean_r2": float(np.mean(list(r2_vals.values()))),
    }
    for t in ALL_TARGETS:
        metrics[f"smape_{t}"] = smape_vals[t]
        metrics[f"r2_{t}"] = r2_vals[t]
        metrics[f"rmse_{t}"] = rmse_vals[t]

    return metrics

# --------------------------------------------------------------------------
# Summary printer
# --------------------------------------------------------------------------

def print_summary(
    metrics: dict,
    num_epochs: Optional[dict] = None,
    training_seconds: Optional[float] = None,
    total_seconds: Optional[float] = None,
    peak_vram_mb: Optional[float] = None,
    platform: Optional[str] = None,
    col_width: int = 23,
) -> None:
    """
    Print the experiment summary.
    """

    def line(key: str, val: str) -> str:
        label = f"{key}:"
        return f"{label:<{col_width}}{val}"

    lines = ["---"]
    lines.append(line("mean_smape", f"{metrics['mean_smape']:.4f}"))
    for t in ALL_TARGETS:
        v = metrics.get(f"smape_{t}")
        if v is not None:
            lines.append(line(f"smape_{t}", f"{v:.4f}"))
    lines.append(line("mean_r2", f"{metrics['mean_r2']:.4f}"))
    for t in ALL_TARGETS:
        v = metrics.get(f"r2_{t}")
        if v is not None:
            lines.append(line(f"r2_{t}", f"{v:.4f}"))
    for t in ALL_TARGETS:
        v = metrics.get(f"rmse_{t}")
        if v is not None:
            lines.append(line(f"rmse_{t}", f"{v:.2f}"))

    if num_epochs is not None:
        lines.append(line("num_epochs", json.dumps(num_epochs)))
    if training_seconds is not None:
        lines.append(line("training_seconds", f"{training_seconds:.2f}"))
    if total_seconds is not None:
        lines.append(line("total_seconds", f"{total_seconds:.2f}"))
    if peak_vram_mb is not None:
        lines.append(line("peak_vram_mb", f"{peak_vram_mb:.2f}"))
    if platform is not None:
        lines.append(line("platform", platform))

    print("\n".join(lines), flush=True)
