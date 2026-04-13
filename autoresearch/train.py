import argparse
import os
import time

os.environ.setdefault(
    "HSA_ENABLE_SDMA", "0"
)  # Set before torch/HSA runtime initializes to prevent ROCm GPU hangs on RDNA3
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow verbose logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from autoresearch.prepare import (
    get_architecture_name,
    ALL_TARGETS,
    CACHE_DIR,
    EVERYTHING_SEED,
    FORCE_CPU,
    TIME_BUDGET,
    build_input_tensors,
    build_inputs_df,
    evaluate,
    load_checkpoint,
    load_split_from_json,
    load_tensor_cache,
    make_dataloader,
    print_summary,
    save_checkpoint,
    set_seed,
    tensor_cache_key,
)
from rule4ml.models.architectures import GNNSettings, TorchTransformerHybridPredictorGated
from rule4ml.models.wrappers import TorchModelWrapper

from autoresearch.wrappers_ext import MCDropoutWrapper

set_seed(EVERYTHING_SEED)

# --------------------------------------------------------------------------
# Categorical maps
# --------------------------------------------------------------------------

GLOBAL_CATEGORICAL_MAPS = {
    "strategy": {"latency": 1, "resource": 2},
    "board": {"pynq-z2": 1, "zcu102": 2, "alveo-u200": 3, "alveo-u250": 4},
    "hls4ml_version": {"0.8.1": 1, "1.1.0": 2},
    "vivado_version": {
        "2019.1": 1,
        "2019.2": 2,
        "2020.1": 3,
        "2020.2": 4,
        "2021.1": 5,
        "2021.2": 6,
        "2022.1": 7,
        "2022.2": 8,
        "2023.1": 9,
        "2023.2": 10,
        "2024.1": 11,
        "2024.2": 12,
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
    "strategy",
    "board",
    "hls4ml_version",
    "vivado_version",
    # Numerical
    "bit_width",
    "reuse_mean",
    "reuse_max",
    "weight_bits_min",
    "weight_bits_max",
    "weight_bits_mean",
    "total_table_size",
    "dense_inputs_mean",
    "dense_outputs_mean",
    "dense_parameters_mean",
    "dense_reuse_mean",
    "dense_reuse_max",
    "dense_count",
    "conv1d_inputs_mean",
    "conv1d_outputs_mean",
    "conv1d_parameters_mean",
    "conv1d_filters_mean",
    "conv1d_kernel_size_mean",
    "conv1d_strides_mean",
    "conv1d_reuse_mean",
    "conv1d_reuse_max",
    "conv1d_count",
    "conv2d_inputs_mean",
    "conv2d_outputs_mean",
    "conv2d_parameters_mean",
    "conv2d_filters_mean",
    "conv2d_kernel_size_mean",
    "conv2d_strides_mean",
    "conv2d_reuse_mean",
    "conv2d_reuse_max",
    "conv2d_count",
    "batchnormalization_inputs_mean",
    "batchnormalization_outputs_mean",
    "batchnormalization_parameters_mean",
    "batchnormalization_count",
    "add_count",
    "concatenate_count",
    "dropout_count",
    "relu_count",
    "sigmoid_count",
    "tanh_count",
    "softmax_inputs_mean",
    "softmax_outputs_mean",
    "softmax_count",
    "total_add",
    "total_mult",
    "total_lookup",
    "total_logical",
]

SEQUENTIAL_FEATURE_LABELS = [
    # Categorical
    "layer_type",
    # Numerical (original 14)
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
    # From data_parser (exp20)
    "layer_weight_bits",  # per-layer weight bit width (from hls_config LayerName)
    # HLS-derived analytical features (exp9, 6 new)
    "layer_multiplier",  # ceil(params / reuse) — proxy for DSP count per layer
    "layer_uses_lut_table",  # 1 if sigmoid/tanh/softmax (lookup-table activation)
    "layer_pipelined",  # 1 if reuse > 1 (pipelined, drives CYCLES/INTERVAL)
    # DSP routing features (exp20)
    "layer_dsp_eligible",  # 1 if layer_weight_bits <= 18 (fits in DSP48E2 B input)
    "layer_dsp_multiplier",  # DSP-routed multiplications per layer
    # Analytical initiation interval (exp32, idea-027)
    "layer_analytical_ii",  # params / max(reuse, 1) — directly encodes II formula
]

# --------------------------------------------------------------------------
# Targets
# - Can be separate predictor for each target, or groups of targets as desired.
# - Can be normalized with board maximum available (percentage) or left as raw values.
# --------------------------------------------------------------------------

TARGET_GROUPS = {"all": ALL_TARGETS}
NORMALIZE_TARGETS = False

# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------

BATCH_SIZE = 256  # 512 regressed, reverting
LEARNING_RATE = 1e-3  # best: 1e-3

# --------------------------------------------------------------------------
# Models factories
# --------------------------------------------------------------------------


GATE_BCE_WEIGHT = 0.05  # BCE weight for gate supervision (light — regression dominates)


def make_predictor(
    output_size: int, device: torch.device, name: str
) -> TorchTransformerHybridPredictorGated:
    return TorchTransformerHybridPredictorGated(
        settings=GNNSettings(
            global_embedding_layers=[16, 16, 16, 16],  # one per global categorical map
            seq_embedding_layers=[16],  # one per sequential categorical map
            numerical_dense_layers=[32],
            gconv_layers=[
                128,
                64,
            ],  # unused by Transformer but kept for GNNSettings compat
            dense_layers=[128, 64],  # Transformer output head: d_model→128→64→output
            dense_dropouts=[],
        ),
        global_input_shape=(None, len(GLOBAL_FEATURE_LABELS)),
        sequential_input_shape=(None, len(SEQUENTIAL_FEATURE_LABELS)),
        output_shape=(None, output_size),
        global_categorical_maps=GLOBAL_CATEGORICAL_MAPS,
        sequential_categorical_maps=SEQUENTIAL_CATEGORICAL_MAPS,
        name=name,
        device=device,
        d_model=128,
        nhead=8,  # 4→8 heads (head_dim 32→16): more diverse attention, same FLOPS, no epoch penalty
        num_layers=4,
        dim_feedforward=512,  # 256→512: more capacity in FF layers; log1p preproc enables better use
        dropout=0.15,  # exp33: 0.1→0.15 for better MC-Dropout uncertainty signal
    )


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def log_mse_loss(y_pred: torch.Tensor, y_true_log: torch.Tensor) -> torch.Tensor:
    """
    MSE loss for log-transformed targets.
    y_true_log is already log1p(raw_target) — model predicts directly in log space.
    Unlike MSLE (which double-logs), this cleanly maps Softplus outputs to log-scale
    targets, giving uniform gradient scale across the full target range.
    """
    return torch.mean(torch.mean((y_pred - y_true_log) ** 2, dim=0))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train_predictor(
    group_name: str,
    group_targets: list,
    input_tensors_splits: dict,  # {"train": {key: Tensor}, "val": {key: Tensor}}
    target_tensors: dict,  # {"train": Tensor(N, len(group_targets)), "val": Tensor}
    time_budget: float,
    base_log_dir: str,
    device: torch.device,
) -> tuple:
    predictor = make_predictor(
        output_size=len(group_targets),
        device=device,
        name=f"{'-'.join(t.upper() for t in group_targets)}_Transformer",
    )
    wrapper = MCDropoutWrapper()
    wrapper.set_model(predictor)
    wrapper.set_input_labels(
        [f for f in GLOBAL_FEATURE_LABELS if f not in GLOBAL_CATEGORICAL_MAPS],
        [f for f in SEQUENTIAL_FEATURE_LABELS if f not in SEQUENTIAL_CATEGORICAL_MAPS],
    )
    wrapper.set_output_labels(group_targets)

    pin_memory = device.type == "cuda"
    # Transform targets to log space: model learns to predict log1p(raw_target)
    log_target_tensors = {s: torch.log1p(target_tensors[s]) for s in target_tensors}
    train_loader = make_dataloader(
        input_tensors_splits["train"],
        log_target_tensors["train"],
        BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        input_tensors_splits["val"],
        log_target_tensors["val"],
        BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,  # SGD failed badly, reverting
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=1,
        eta_min=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=1,
        eta_min=1e-6,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=5,  # exp38: 5 epoch LR warmup
    )

    log_dir = os.path.join(base_log_dir, group_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    n_epochs = 0
    progress = 0.0
    training_time = 0.0
    best_val_loss = float("inf")
    bce_fn = torch.nn.BCELoss()

    while True:
        running_loss = 0.0
        predictor.train()

        t0 = time.time()
        for inputs, targets in train_loader:
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds, gates = predictor.forward_with_gates(inputs)
            mse = log_mse_loss(preds, targets)
            # BCE on resource gates: binary label = (raw_target > 0)
            # targets are already log1p-transformed; expm1 recovers raw values
            raw_resource = torch.expm1(targets[:, : predictor._N_RESOURCE])
            binary_targets = (raw_resource > 0).float()
            bce = bce_fn(gates, binary_targets)
            loss = mse + GATE_BCE_WEIGHT * bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                predictor.parameters(), max_norm=1.0
            )  # 0.5 regressed
            optimizer.step()

            running_loss += loss.item()

        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.time()
        training_time += t1 - t0
        progress = min(1.0, training_time / time_budget)

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/Train", epoch_loss, n_epochs)

        predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                targets = targets.to(device, non_blocking=True)
                preds, gates = predictor.forward_with_gates(inputs)
                mse = log_mse_loss(preds, targets)
                raw_resource = torch.expm1(targets[:, : predictor._N_RESOURCE])
                binary_targets = (raw_resource > 0).float()
                bce = bce_fn(gates, binary_targets)
                val_loss += (mse + GATE_BCE_WEIGHT * bce).item()
        val_loss /= max(len(val_loader), 1)
        writer.add_scalar("Loss/Val", val_loss, n_epochs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(checkpoints_dir, f"{group_name}_best.pt"),
                n_epochs,
                predictor,
                optimizer,
                scheduler,
                best_val_loss,
            )

        save_checkpoint(
            os.path.join(checkpoints_dir, f"{group_name}_latest.pt"),
            n_epochs,
            predictor,
            optimizer,
            scheduler,
            best_val_loss,
        )

        scheduler.step()

        # Periodically free fragmented GPU memory to reduce chance of ROCm driver hang
        if device.type == "cuda":
            torch.cuda.empty_cache()

        n_epochs += 1
        print(
            f"Epoch {n_epochs} done, Overall progress: {progress:.2%}"
            f", Training time: {training_time / 60:.1f} min",
            flush=True,
        )

        if progress >= 1.0:
            break

    load_checkpoint(
        os.path.join(checkpoints_dir, f"{group_name}_best.pt"),
        predictor,
        optimizer=None,
        lr_scheduler=None,
        device=device,
    )
    wrapper.save(log_dir)

    return wrapper, n_epochs, training_time


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main():
    total_start = time.time()

    arg_parser = argparse.ArgumentParser(description="Train FPGA resource predictors")
    arg_parser.add_argument(
        "--branch-name", required=True, type=str, help="Git branch name"
    )
    arg_parser.add_argument(
        "--commit-hash", required=True, type=str, help="Git commit hash"
    )
    cli_args = arg_parser.parse_args()

    use_gpu = torch.cuda.is_available() and not FORCE_CPU
    current_device = torch.cuda.current_device() if use_gpu else ""
    device = torch.device(f"cuda:{current_device}" if use_gpu else "cpu")
    device_name = torch.cuda.get_device_name(current_device) if use_gpu else "CPU"

    # On cache hit: skip all raw-data loading and reuse prebuilt tensors.
    # On cache miss: load all three splits from raw JSON, tensorize once, write to disk.
    # The cache key is aligned with the hash prepare.predict() computes at lookup time.
    # Include SEQUENTIAL_FEATURE_LABELS in the cache key so adding new sequential
    # features automatically invalidates and rebuilds the cache.
    feature_hash = tensor_cache_key(
        GLOBAL_FEATURE_LABELS + ["sequential_inputs"] + SEQUENTIAL_FEATURE_LABELS,
        GLOBAL_CATEGORICAL_MAPS.keys(),
        SEQUENTIAL_CATEGORICAL_MAPS.keys(),
        NORMALIZE_TARGETS,
    )
    cache_path = os.path.join(CACHE_DIR, f"{feature_hash}_train_val.pt")

    cached = load_tensor_cache(cache_path)
    if cached is not None:
        print(f"Loading tensor cache: {cache_path}", flush=True)
        input_tensors_splits = cached["inputs"]
        target_tensors_all = cached["targets"]
        n_train = cached["meta"]["split_lengths"]["train"]
        n_val = cached["meta"]["split_lengths"]["val"]

        test_raw_df = load_split_from_json(
            "test",
            GLOBAL_CATEGORICAL_MAPS,
            SEQUENTIAL_CATEGORICAL_MAPS,
            normalize=NORMALIZE_TARGETS,
        )
        n_test = len(test_raw_df)

        print(
            f"  train: {n_train} | val: {n_val} | test: {n_test} samples (from cache)",
            flush=True,
        )
    else:
        print("Tensor cache not found, rebuilding splits from raw JSON...", flush=True)
        raw_splits = {
            s: load_split_from_json(
                s,
                GLOBAL_CATEGORICAL_MAPS,
                SEQUENTIAL_CATEGORICAL_MAPS,
                normalize=NORMALIZE_TARGETS,
            )
            for s in ("train", "val", "test")
        }
        for s, df in raw_splits.items():
            print(f"  {s}: {len(df)} samples", flush=True)

        inputs_df_splits_full = {
            s: build_inputs_df(raw_splits[s], GLOBAL_FEATURE_LABELS, SEQUENTIAL_FEATURE_LABELS)
            for s in ("train", "val")
        }

        # Use a throw-away wrapper/GNN just to define input structure for build_inputs()
        # Use the full output size (6) so hybrid predictors don't get a negative n_timing dim.
        cpu_dev = torch.device("cpu")
        _ref_gnn = make_predictor(
            output_size=len(ALL_TARGETS), device=cpu_dev, name="ref"
        )
        _ref_wrapper = TorchModelWrapper()
        _ref_wrapper.set_model(_ref_gnn)
        _ref_wrapper.set_input_labels(
            [f for f in GLOBAL_FEATURE_LABELS if f not in GLOBAL_CATEGORICAL_MAPS],
            [
                f
                for f in SEQUENTIAL_FEATURE_LABELS
                if f not in SEQUENTIAL_CATEGORICAL_MAPS
            ],
        )

        print("Tensorizing splits (cached after the first run)...", flush=True)
        input_tensors_splits = {
            s: build_input_tensors(
                _ref_wrapper, inputs_df_splits_full[s], device=cpu_dev
            )
            for s in ("train", "val")
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
                        for s in ("train", "val")
                    },
                },
            },
            cache_path,
        )
        print(f"  Saved tensor cache: {cache_path}", flush=True)

        test_raw_df = raw_splits["test"]

    branch_name = cli_args.branch_name
    commit_hash = cli_args.commit_hash
    base_log_dir = os.path.join(
        os.path.dirname(__file__), "runs", branch_name, commit_hash
    )
    os.makedirs(base_log_dir, exist_ok=True)

    n_groups = len(TARGET_GROUPS)
    per_group_budget = TIME_BUDGET / n_groups

    trained_wrappers = []
    num_epochs = {}
    training_seconds = 0.0

    for group_name, group_targets in TARGET_GROUPS.items():
        print(
            f"\nTraining '{group_name}': targets={group_targets}"
            f", budget={per_group_budget:.0f}s",
            flush=True,
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
            base_log_dir=base_log_dir,
            device=device,
        )
        num_epochs[group_name] = epochs
        trained_wrappers.append((wrapper, group_targets))
        training_seconds += training_time

    metrics = evaluate(
        trained_wrappers,
        test_raw_df,
        GLOBAL_FEATURE_LABELS,
        SEQUENTIAL_FEATURE_LABELS,
    )
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if use_gpu else 0.0
    total_seconds = time.time() - total_start

    print_summary(
        metrics=metrics,
        num_epochs=num_epochs,
        training_seconds=training_seconds,
        total_seconds=total_seconds,
        peak_vram_mb=peak_vram_mb,
        platform=device_name,
    )

    # ── MC-Dropout uncertainty analysis ──────────────────────────────────────
    test_raw_df["_architecture"] = test_raw_df["model_name"].apply(get_architecture_name)
    unique_archs = sorted(test_raw_df["_architecture"].unique())

    for wrapper, group_targets in trained_wrappers:
        if not hasattr(wrapper, "predict_uncertainty_from_df"):
            continue
        test_inputs_df = build_inputs_df(
            test_raw_df, GLOBAL_FEATURE_LABELS, SEQUENTIAL_FEATURE_LABELS
        )
        safe_inputs = test_inputs_df.drop(
            columns=[c for c in ALL_TARGETS if c in test_inputs_df.columns]
        )
        _, variance = wrapper.predict_uncertainty_from_df(safe_inputs)

        for i, target in enumerate(group_targets):
            by_arch = {
                arch: float(np.mean(variance[test_raw_df["_architecture"].values == arch, i]))
                for arch in unique_archs
            }
            by_arch["all architectures"] = float(np.mean(variance[:, i]))
            print(f"uncertainty_mc_variance_{target}: {by_arch}", flush=True)
