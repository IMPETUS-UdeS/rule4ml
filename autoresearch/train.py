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
from rule4ml.models.architectures import GNNSettings, TorchTransformerHybridPredictor
from rule4ml.models.wrappers import BaseModelWrapper, TorchModelWrapper

from autoresearch.wrappers_ext import MCDropoutWrapper, SeparateHeadWrapper

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
PHASE2_BUDGET = 300
UNCERTAINTY_ENSEMBLE_SIZE = 4
UNCERTAINTY_RANK_LAMBDA = 0.02
UNCERTAINTY_PAIR_SUBSAMPLE = 64

# --------------------------------------------------------------------------
# Models factories
# --------------------------------------------------------------------------


GATE_BCE_WEIGHT = 0.05  # BCE weight for gate supervision (light — regression dominates)


def make_predictor(
    output_size: int, device: torch.device, name: str
) -> TorchTransformerHybridPredictor:
    return TorchTransformerHybridPredictor(
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


def fit_monotonic_uncertainty_calibration(
    raw_uncertainty: np.ndarray, true_log_abs_error: np.ndarray
) -> list[dict]:
    """
    Fit per-target monotonic piecewise-linear maps from predicted uncertainty to
    observed log-space absolute error using isotonic regression via PAV.
    """

    def pav_isotonic(x_sorted, y_sorted):
        blocks = []
        for xi, yi in zip(x_sorted, y_sorted):
            blocks.append(
                {
                    "start": float(xi),
                    "end": float(xi),
                    "sum": float(yi),
                    "count": 1,
                }
            )
            while len(blocks) >= 2:
                last = blocks[-1]
                prev = blocks[-2]
                if prev["sum"] / prev["count"] <= last["sum"] / last["count"]:
                    break
                merged = {
                    "start": prev["start"],
                    "end": last["end"],
                    "sum": prev["sum"] + last["sum"],
                    "count": prev["count"] + last["count"],
                }
                blocks[-2:] = [merged]
        xs = []
        ys = []
        for block in blocks:
            mean_y = block["sum"] / block["count"]
            xs.append(block["start"])
            ys.append(mean_y)
            if block["end"] > block["start"]:
                xs.append(block["end"])
                ys.append(mean_y)
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    calibrators = []
    for i in range(raw_uncertainty.shape[1]):
        x = np.asarray(raw_uncertainty[:, i], dtype=np.float64)
        y = np.asarray(true_log_abs_error[:, i], dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        y_sorted = y[order]
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        y_means = np.bincount(inverse, weights=y_sorted) / np.maximum(
            np.bincount(inverse), 1
        )
        iso_x, iso_y = pav_isotonic(unique_x, y_means)
        calibrators.append({"x": iso_x.tolist(), "y": iso_y.tolist()})
    return calibrators


def build_uncertainty_calibration(
    wrapper,
    val_inputs,
    val_log_targets: torch.Tensor,
):
    predictor = wrapper.model
    predictor.eval()
    predictor._store_repr = True
    with torch.no_grad():
        unc_inputs = {k: v.to(predictor.device) for k, v in val_inputs.items()}
        preds = predictor(unc_inputs)
        true_log_abs_error = torch.log1p(
            torch.abs(preds - val_log_targets.to(predictor.device))
        )
        if predictor.uncertainty_resource_ensemble_heads is not None:
            mean_uncertainty, std_uncertainty = predictor.forward_uncertainty_stats(
                unc_inputs
            )
            wrapper.uncertainty_spread_weight = fit_spread_weights(
                mean_uncertainty.cpu().numpy(),
                std_uncertainty.cpu().numpy(),
                true_log_abs_error.cpu().numpy(),
            )
            spread_weight = torch.tensor(
                wrapper.uncertainty_spread_weight,
                dtype=mean_uncertainty.dtype,
                device=predictor.device,
            )
            raw_uncertainty = mean_uncertainty + std_uncertainty * spread_weight
        else:
            raw_uncertainty = predictor.forward_uncertainty(unc_inputs)
    predictor._store_repr = False
    wrapper.uncertainty_calibration = fit_monotonic_uncertainty_calibration(
        raw_uncertainty.cpu().numpy(),
        true_log_abs_error.cpu().numpy(),
    )


def fit_spread_weights(
    mean_uncertainty: np.ndarray,
    std_uncertainty: np.ndarray,
    true_log_abs_error: np.ndarray,
) -> list[float]:
    alphas = np.linspace(0.0, 3.0, 31)
    weights = []

    def safe_corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    for i in range(mean_uncertainty.shape[1]):
        best_alpha = 0.0
        best_corr = safe_corr(mean_uncertainty[:, i], true_log_abs_error[:, i])
        for alpha in alphas[1:]:
            fused = mean_uncertainty[:, i] + alpha * std_uncertainty[:, i]
            corr = safe_corr(fused, true_log_abs_error[:, i])
            if corr > best_corr:
                best_corr = corr
                best_alpha = float(alpha)
        weights.append(best_alpha)
    return weights


def pairwise_ranking_loss(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
    max_pairs: int = UNCERTAINTY_PAIR_SUBSAMPLE,
) -> torch.Tensor:
    batch_size = pred_scores.shape[0]
    if batch_size < 2:
        return pred_scores.new_tensor(0.0)
    pair_count = min(max_pairs, batch_size // 2)
    perm_a = torch.randperm(batch_size, device=pred_scores.device)[:pair_count]
    perm_b = torch.randperm(batch_size, device=pred_scores.device)[:pair_count]
    valid = perm_a != perm_b
    if not torch.any(valid):
        return pred_scores.new_tensor(0.0)
    a_idx = perm_a[valid]
    b_idx = perm_b[valid]
    true_diff = true_scores[a_idx] - true_scores[b_idx]
    pred_diff = pred_scores[a_idx] - pred_scores[b_idx]
    sign = torch.sign(true_diff)
    informative = sign != 0
    if not torch.any(informative):
        return pred_scores.new_tensor(0.0)
    return torch.nn.functional.softplus(-sign[informative] * pred_diff[informative]).mean()


def train_uncertainty_heads(
    predictor: TorchTransformerHybridPredictor,
    val_inputs: dict,
    val_log_targets: torch.Tensor,
    time_budget: float,
    device: torch.device,
) -> tuple[int, float]:
    predictor.add_uncertainty_modulated_ensemble_heads(
        d_hidden=32,
        ensemble_size=UNCERTAINTY_ENSEMBLE_SIZE,
    )

    for param in predictor.parameters():
        param.requires_grad_(False)
    for param in predictor.uncertainty_resource_ensemble_heads.parameters():
        param.requires_grad_(True)
    for param in predictor.uncertainty_timing_ensemble_heads.parameters():
        param.requires_grad_(True)
    for param in predictor.uncertainty_resource_modulators.parameters():
        param.requires_grad_(True)
    for param in predictor.uncertainty_timing_modulators.parameters():
        param.requires_grad_(True)

    unc_params = (
        list(predictor.uncertainty_resource_ensemble_heads.parameters())
        + list(predictor.uncertainty_timing_ensemble_heads.parameters())
        + list(predictor.uncertainty_resource_modulators.parameters())
        + list(predictor.uncertainty_timing_modulators.parameters())
    )
    optimizer = torch.optim.Adam(unc_params, lr=1e-3)
    pin_memory = device.type == "cuda"
    val_loader = make_dataloader(
        val_inputs, val_log_targets, BATCH_SIZE, shuffle=True, pin_memory=pin_memory
    )

    t_start = time.time()
    n_epochs = 0
    final_loss = float("nan")
    while True:
        total_loss = 0.0
        n_batches = 0
        for inputs, log_targets in val_loader:
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            log_targets = log_targets.to(device, non_blocking=True)

            predictor.eval()
            predictor._store_repr = True
            with torch.no_grad():
                preds = predictor(inputs)
            predictor._store_repr = False

            log_residuals = torch.log1p(torch.abs(preds - log_targets))

            predictor.uncertainty_resource_ensemble_heads.train()
            predictor.uncertainty_timing_ensemble_heads.train()
            predictor.uncertainty_resource_modulators.train()
            predictor.uncertainty_timing_modulators.train()
            member_losses = []
            for member_idx, (resource_heads, timing_head) in enumerate(
                zip(
                    predictor.uncertainty_resource_ensemble_heads,
                    predictor.uncertainty_timing_ensemble_heads,
                )
            ):
                r_unc = torch.cat(
                    [
                        resource_heads[i](
                            predictor._last_r_ctx[:, i, :]
                            * predictor.uncertainty_resource_modulators[member_idx][i](
                                predictor._last_r_ctx[:, i, :]
                            )
                        )
                        for i in range(predictor._N_RESOURCE)
                    ],
                    dim=-1,
                )
                t_unc = timing_head(
                    predictor._last_cls_out
                    * predictor.uncertainty_timing_modulators[member_idx](
                        predictor._last_cls_out
                    )
                )
                unc_pred = torch.cat([r_unc, t_unc], dim=-1)
                mse_loss = torch.mean((unc_pred - log_residuals) ** 2)
                rank_loss = torch.stack(
                    [
                        pairwise_ranking_loss(unc_pred[:, i], log_residuals[:, i])
                        for i in range(log_residuals.shape[1])
                    ]
                ).mean()
                member_losses.append(mse_loss + UNCERTAINTY_RANK_LAMBDA * rank_loss)

            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(member_losses).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        n_epochs += 1
        final_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t_start
        print(
            f"Phase 2 epoch {n_epochs} done, loss={final_loss:.4f}, elapsed={elapsed:.0f}s",
            flush=True,
        )
        if elapsed >= time_budget:
            break

    for param in predictor.parameters():
        param.requires_grad_(True)

    return n_epochs, time.time() - t_start


def uncertainty_calibration_report(
    wrapper,
    eval_df,
    feature_labels,
    sequential_labels,
    target_labels,
):
    test_inputs_df = build_inputs_df(eval_df, feature_labels, sequential_labels)
    safe_inputs = test_inputs_df.drop(
        columns=[c for c in ALL_TARGETS if c in test_inputs_df.columns],
        errors="ignore",
    )
    point_estimates, predicted_unc = wrapper.predict_uncertainty_from_df(safe_inputs)

    y_true = eval_df[target_labels].to_numpy(dtype=float)
    abs_error_raw = np.abs(point_estimates - y_true)
    pred_log = BaseModelWrapper.predict_from_df(wrapper, safe_inputs.copy())
    true_log = np.log1p(y_true)
    abs_error_log = np.abs(pred_log - true_log)
    archs = eval_df["model_name"].apply(get_architecture_name).to_numpy()
    unique_archs = sorted(np.unique(archs))

    def safe_corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    for i, target in enumerate(target_labels):
        rmse_log = float(
            np.sqrt(np.mean((predicted_unc[:, i] - abs_error_log[:, i]) ** 2))
        )
        print(
            f"uncertainty_corr_logabs_{target}: {safe_corr(predicted_unc[:, i], abs_error_log[:, i]):.4f}",
            flush=True,
        )
        print(
            f"uncertainty_corr_rawabs_{target}: {safe_corr(np.expm1(predicted_unc[:, i]), abs_error_raw[:, i]):.4f}",
            flush=True,
        )
        print(f"uncertainty_rmse_logabs_{target}: {rmse_log:.4f}", flush=True)
        by_arch_unc = {
            arch: float(np.mean(predicted_unc[archs == arch, i])) for arch in unique_archs
        }
        by_arch_err = {
            arch: float(np.mean(abs_error_log[archs == arch, i])) for arch in unique_archs
        }
        by_arch_unc["all architectures"] = float(np.mean(predicted_unc[:, i]))
        by_arch_err["all architectures"] = float(np.mean(abs_error_log[:, i]))
        print(f"uncertainty_mean_{target}: {by_arch_unc}", flush=True)
        print(f"uncertainty_true_logabs_{target}: {by_arch_err}", flush=True)

    if hasattr(wrapper, "predict_uncertainty_interval_from_df") and (
        wrapper.model.uncertainty_resource_quantile_heads is not None
    ):
        _, lower_quant, upper_quant = wrapper.predict_uncertainty_interval_from_df(
            safe_inputs
        )
        interval_hit = (abs_error_log >= lower_quant) & (abs_error_log <= upper_quant)
        upper_hit = abs_error_log <= upper_quant
        for i, target in enumerate(target_labels):
            print(
                f"uncertainty_interval_coverage_{target}: {float(np.mean(interval_hit[:, i])):.4f}",
                flush=True,
            )
            print(
                f"uncertainty_upper_coverage_{target}: {float(np.mean(upper_hit[:, i])):.4f}",
                flush=True,
            )


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
    arg_parser.add_argument(
        "--backbone-run-dir",
        type=str,
        default=None,
        help="Existing run directory with saved config/weights to use as frozen backbone",
    )
    arg_parser.add_argument(
        "--uncertainty-budget",
        type=float,
        default=PHASE2_BUDGET,
        help="Phase-2 training budget in seconds for post-hoc uncertainty heads",
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

    if cli_args.backbone_run_dir:
        print(
            f"\nTraining uncertainty heads from frozen backbone: {cli_args.backbone_run_dir}",
            flush=True,
        )
        group_name = "all"
        group_targets = ALL_TARGETS
        group_log_dir = os.path.join(base_log_dir, group_name)
        os.makedirs(group_log_dir, exist_ok=True)

        config_path = os.path.join(
            cli_args.backbone_run_dir,
            "BRAM-DSP-FF-LUT-CYCLES-INTERVAL_Transformer.config.json",
        )
        weights_path = os.path.join(
            cli_args.backbone_run_dir,
            "BRAM-DSP-FF-LUT-CYCLES-INTERVAL_Transformer.weights.pt",
        )
        wrapper = SeparateHeadWrapper()
        wrapper.load(config_path, weights_path)
        wrapper.model.to(device)

        val_log_targets = torch.log1p(target_tensors_all["val"])
        unc_epochs, unc_seconds = train_uncertainty_heads(
            predictor=wrapper.model,
            val_inputs=input_tensors_splits["val"],
            val_log_targets=val_log_targets,
            time_budget=cli_args.uncertainty_budget,
            device=device,
        )
        build_uncertainty_calibration(
            wrapper,
            input_tensors_splits["val"],
            val_log_targets,
        )
        wrapper.save(group_log_dir)

        metrics = evaluate(
            [(wrapper, group_targets)],
            test_raw_df,
            GLOBAL_FEATURE_LABELS,
            SEQUENTIAL_FEATURE_LABELS,
        )
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if use_gpu else 0.0
        )
        total_seconds = time.time() - total_start
        print_summary(
            metrics=metrics,
            num_epochs={"all": "frozen_backbone", "uncertainty_heads": unc_epochs},
            training_seconds=unc_seconds,
            total_seconds=total_seconds,
            peak_vram_mb=peak_vram_mb,
            platform=device_name,
        )
        uncertainty_calibration_report(
            wrapper,
            test_raw_df,
            GLOBAL_FEATURE_LABELS,
            SEQUENTIAL_FEATURE_LABELS,
            group_targets,
        )
        return

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
