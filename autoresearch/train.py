import argparse
import os
import time

import torch

from autoresearch.prepare import (ALL_TARGETS, CACHE_DIR, EVERYTHING_SEED,
                                  FORCE_CPU, TIME_BUDGET, build_input_tensors,
                                  build_inputs_df, evaluate,
                                  load_split_from_json, load_tensor_cache,
                                  make_dataloader, print_summary, set_seed,
                                  tensor_cache_key)
from rule4ml.models.architectures import GNNSettings, TorchGNN
from rule4ml.models.wrappers import TorchModelWrapper

set_seed(EVERYTHING_SEED)

# --------------------------------------------------------------------------
# Categorical maps
# --------------------------------------------------------------------------

GLOBAL_CATEGORICAL_MAPS = {
    "strategy": {"latency": 1, "resource": 2},
    "board": {"pynq-z2": 1, "zcu102": 2, "alveo-u200": 3, "alveo-u250": 4},
    "hls4ml_version": {"0.8.1": 1, "1.1.0": 2},
    "vivado_version": {
        "2019.1": 1, "2019.2": 2, "2020.1": 3, "2020.2": 4,
        "2021.1": 5, "2021.2": 6, "2022.1": 7, "2022.2": 8,
        "2023.1": 9, "2023.2": 10, "2024.1": 11, "2024.2": 12,
    },
}

SEQUENTIAL_CATEGORICAL_MAPS = {
    "layer_type": {
        "inputlayer": 1,
        "dense": 2,
        "conv1d": 3,
        "conv2d": 4,
        "maxpooling1d": 5,
        "averagepooling1d": 6,
        "maxpooling2d": 7,
        "averagepooling2d": 8,
        "relu": 9,
        "sigmoid": 10,
        "tanh": 11,
        "softmax": 12,
        "batchnormalization": 13,
        "add": 14,
        "concatenate": 15,
        "dropout": 16,
        "flatten": 17,
    }
}

# --------------------------------------------------------------------------
# Feature labels
# --------------------------------------------------------------------------

GLOBAL_FEATURE_LABELS = [
    # Categorical (must appear first to match global_input_shape computation)
    "strategy", "board", "hls4ml_version", "vivado_version",
    # Numerical
    "bit_width", "reuse_mean",
    "dense_inputs_mean", "dense_outputs_mean", "dense_parameters_mean",
    "dense_reuse_mean", "dense_count",
    "conv1d_inputs_mean", "conv1d_outputs_mean", "conv1d_parameters_mean",
    "conv1d_filters_mean", "conv1d_kernel_size_mean", "conv1d_strides_mean",
    "conv1d_reuse_mean", "conv1d_count",
    "conv2d_inputs_mean", "conv2d_outputs_mean", "conv2d_parameters_mean",
    "conv2d_filters_mean", "conv2d_kernel_size_mean", "conv2d_strides_mean",
    "conv2d_reuse_mean", "conv2d_count",
    "batchnormalization_inputs_mean", "batchnormalization_outputs_mean",
    "batchnormalization_parameters_mean", "batchnormalization_count",
    "add_count", "concatenate_count", "dropout_count",
    "relu_count", "sigmoid_count", "tanh_count",
    "softmax_inputs_mean", "softmax_outputs_mean", "softmax_count",
    "total_add", "total_mult", "total_lookup", "total_logical",
]

SEQUENTIAL_FEATURE_LABELS = [
    # Categorical
    "layer_type",
    # Numerical
    "layer_input_size",
    "layer_output_size",
    "layer_parameter_count",
    "layer_trainable_parameter_count",
    "layer_filters",
    "layer_kernel_height",
    "layer_kernel_width",
    "layer_stride_height",
    "layer_stride_width",
    "layer_reuse",
    "layer_op_add",
    "layer_op_mult",
    "layer_op_lookup",
    "layer_op_logical",
]

# --------------------------------------------------------------------------
# Targets
# Can be separate predictor for each target,
# or groups of targets as desired.
# --------------------------------------------------------------------------

# Joint model: all 6 targets in a single predictor.
# This gives the full 3600s budget to one model (~36 epochs vs 6 in baseline).
TARGET_GROUPS = {"all": ALL_TARGETS}

# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------

BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# --------------------------------------------------------------------------
# Models factories
# --------------------------------------------------------------------------

def make_gnn(output_size: int, device: torch.device, name: str = "GNN") -> TorchGNN:
    return TorchGNN(
        settings=GNNSettings(
            global_embedding_layers=[16, 16, 16, 16],  # one per global categorical map
            seq_embedding_layers=[16],  # one per sequential categorical map
            numerical_dense_layers=[32],
            gconv_layers=[128, 64],
            dense_layers=[128, 128, 64],
            dense_dropouts=[0.2, 0.2],
        ),
        global_input_shape=(None, len(GLOBAL_FEATURE_LABELS)),
        sequential_input_shape=(None, len(SEQUENTIAL_FEATURE_LABELS)),
        output_shape=(None, output_size),
        global_categorical_maps=GLOBAL_CATEGORICAL_MAPS,
        sequential_categorical_maps=SEQUENTIAL_CATEGORICAL_MAPS,
        name=name,
        device=device,
    )

# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def log_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    L1 (MAE) in log1p space, averaged per target then across targets.
    More robust to extreme outliers than MSLE (L2):
      - MSLE weight ratio outlier/typical ≈ 1500x
      - log-MAE weight ratio           ≈ 39x
    This prevents cycles/interval extreme values from dominating,
    while preserving meaningful gradient signal for large predictions.
    """
    log_pred = torch.log1p(torch.clamp(y_pred, min=0.0))
    log_true = torch.log1p(torch.clamp(y_true, min=0.0))
    return torch.mean(torch.mean(torch.abs(log_pred - log_true), dim=0))

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_predictor(
    group_name: str,
    group_targets: list,
    input_tensors_splits: dict,  # {"train": {key: Tensor}, "val": {key: Tensor}}
    target_tensors: dict,  # {"train": Tensor(N, len(group_targets)), "val": Tensor}
    time_budget: float,
    device: torch.device,
) -> tuple:
    predictor = make_gnn(
        output_size=len(group_targets),
        device=device,
        name=f"{'-'.join(t.upper() for t in group_targets)}_GNN",
    )
    wrapper = TorchModelWrapper()
    wrapper.set_model(predictor)

    pin_memory = device.type == "cuda"
    train_loader = make_dataloader(
        input_tensors_splits["train"],
        target_tensors["train"],
        BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        input_tensors_splits["val"],
        target_tensors["val"],
        BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Cosine annealing: T_max is estimated as total_budget / (one epoch cost).
    # We use a generous T_max so the LR decays slowly. Restarts every ~20 epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    n_epochs = 0

    progress = 0.0
    training_time = 0.0
    while True:
        predictor.train()
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        for inputs, targets in train_loader:
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = log_mae_loss(predictor(inputs), targets)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.time()
        training_time += t1 - t0
        progress = min(1.0, training_time / time_budget)

        predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                targets = targets.to(device, non_blocking=True)
                val_loss += log_mae_loss(predictor(inputs), targets).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in predictor.state_dict().items()
            }

        scheduler.step()

        if progress < 1.0:
            n_epochs += 1
        print(
            f"Epoch {n_epochs} done, Overall progress: {progress:.2%}"
            f", Training time: {training_time/60:.1f} min",
            flush=True
        )

        if progress >= 1.0:
            break

    if best_state is not None:
        predictor.load_state_dict(best_state)

    return wrapper, n_epochs, training_time

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    total_start = time.time()

    arg_parser = argparse.ArgumentParser(description="Train FPGA resource predictors")
    arg_parser.add_argument("--branch-name", type=str, help="Git branch name")
    arg_parser.add_argument("--commit-hash", type=str, help="Git commit hash")
    cli_args = arg_parser.parse_args()

    use_gpu = torch.cuda.is_available() and not FORCE_CPU
    current_device = torch.cuda.current_device() if use_gpu else ""
    device = torch.device(f"cuda:{current_device}" if use_gpu else "cpu")
    device_name = torch.cuda.get_device_name(current_device) if use_gpu else "CPU"

    # On cache hit: skip all raw-data loading and reuse prebuilt tensors.
    # On cache miss: load all three splits from raw JSON, tensorize once, write to disk.
    # The cache key is aligned with the hash prepare.predict() computes at lookup time.
    feature_hash = tensor_cache_key(
        GLOBAL_FEATURE_LABELS + ["sequential_inputs"],
        GLOBAL_CATEGORICAL_MAPS.keys(),
        SEQUENTIAL_CATEGORICAL_MAPS.keys(),
    )
    cache_path = os.path.join(CACHE_DIR, f"tensors_{feature_hash}.pt")

    cached = load_tensor_cache(cache_path)
    if cached is not None:
        print(f"Loading tensor cache: {cache_path}", flush=True)
        input_tensors_splits = cached["inputs"]  # {split: {key: Tensor}}
        target_tensors_all = cached["targets"]
        n_train = cached["meta"]["split_lengths"]["train"]
        n_val = cached["meta"]["split_lengths"]["val"]

        raw_test_df = load_split_from_json(
            "test",
            GLOBAL_CATEGORICAL_MAPS,
            SEQUENTIAL_CATEGORICAL_MAPS,
        )
        test_inputs_df = build_inputs_df(
            raw_test_df,
            GLOBAL_FEATURE_LABELS,
            SEQUENTIAL_FEATURE_LABELS,
        )
        test_targets_df = raw_test_df[ALL_TARGETS].copy()
        n_test = len(raw_test_df)

        print(
            f"  train: {n_train} | val: {n_val}"
            f" | test: {n_test} samples (from cache)",
            flush=True)
    else:
        print(
            "Tensor cache not found, rebuilding splits from raw JSON...",
            flush=True
        )
        raw_splits = {
            s: load_split_from_json(
                s,
                GLOBAL_CATEGORICAL_MAPS,
                SEQUENTIAL_CATEGORICAL_MAPS,
            )
            for s in ("train", "val", "test")
        }
        for s, df in raw_splits.items():
            print(f"  {s}: {len(df)} samples", flush=True)

        inputs_df_splits_full = {
            s: build_inputs_df(df, GLOBAL_FEATURE_LABELS, SEQUENTIAL_FEATURE_LABELS)
            for s, df in raw_splits.items()
        }

        # Use a throw-away wrapper/GNN just to define input structure for build_inputs()
        cpu_dev = torch.device("cpu")
        _ref_gnn = make_gnn(output_size=1, device=cpu_dev, name="ref")
        _ref_wrapper = TorchModelWrapper()
        _ref_wrapper.set_model(_ref_gnn)

        print("Tensorizing splits (cached after the first run)...", flush=True)
        input_tensors_splits = {
            s: build_input_tensors(_ref_wrapper, df, device=cpu_dev)
            for s, df in inputs_df_splits_full.items()
        }
        target_tensors_all = {
            s: torch.tensor(raw_splits[s][ALL_TARGETS].values, dtype=torch.float32)
            for s in ("train", "val")
        }

        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save(
            {
                "inputs": input_tensors_splits,
                "targets": target_tensors_all,
                "meta": {
                    "feature_columns": GLOBAL_FEATURE_LABELS + ["sequential_inputs"],
                    "target_columns": ALL_TARGETS,
                    "source": "json_to_df",
                    "split_lengths": {
                        s: len(next(iter(input_tensors_splits[s].values())))
                        for s in ("train", "val", "test")
                    },
                },
            },
            cache_path,
        )
        print(f"  Saved tensor cache: {cache_path}", flush=True)

        test_inputs_df = inputs_df_splits_full["test"]
        test_targets_df = raw_splits["test"][ALL_TARGETS]

    n_groups = len(TARGET_GROUPS)
    per_group_budget = TIME_BUDGET / n_groups

    trained_wrappers = []
    num_epochs = {}
    training_seconds = 0.0

    for group_name, group_targets in TARGET_GROUPS.items():
        print(
            f"\nTraining '{group_name}': targets={group_targets}"
            f", budget={per_group_budget:.0f}s",
            flush=True
        )
        group_idx = [ALL_TARGETS.index(t) for t in group_targets]
        group_target_tensors = {
            s: target_tensors_all[s][:, group_idx] for s in ("train", "val")
        }
        wrapper, epochs, training_time = train_predictor(
            group_name=group_name,
            group_targets=group_targets,
            input_tensors_splits=input_tensors_splits,
            target_tensors=group_target_tensors,
            time_budget=per_group_budget,
            device=device,
        )
        num_epochs[group_name] = epochs
        trained_wrappers.append((wrapper, group_targets))
        training_seconds += training_time

    total_seconds = time.time() - total_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if use_gpu else 0.0
    )

    metrics = evaluate(
        trained_wrappers,
        test_inputs_df,
        test_targets_df,
        device,
    )
    print_summary(
        metrics=metrics,
        num_epochs=num_epochs,
        training_seconds=training_seconds,
        total_seconds=total_seconds,
        peak_vram_mb=peak_vram_mb,
        platform=device_name,
    )
