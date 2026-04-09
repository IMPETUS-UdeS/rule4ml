import argparse
import glob
import os
from typing import Sequence

os.environ.setdefault(
    "HSA_ENABLE_SDMA", "0"
)  # Set before torch/HSA runtime initializes to prevent ROCm GPU hangs on RDNA3
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow verbose logging

import numpy as np
import torch

from autoresearch.prepare import (
    ALL_TARGETS,
    FORCE_CPU,
    JSON_BATCH_SIZE,
    JSON_MAX_WORKERS,
    build_inputs_df,
    get_split_json_patterns,
    predict,
    print_summary,
    r2,
    rmse,
    smape,
)
from rule4ml.models.wrappers import (
    BaseModelWrapper,
    KerasModelWrapper,
    TorchModelWrapper,
)
from rule4ml.parsers.data_parser import (
    get_global_data,
    get_sequential_data,
    read_from_json,
    to_dataframe,
)


def print_arch_summary(
    arch_name: str,
    smape_vals: dict,
    r2_vals: dict,
    rmse_vals: dict,
    col_width: int = 23,
):
    """Print metrics for a single architecture."""
    print(f"{'=' * col_width * 4}", flush=True)
    print(f"{'Architecture: ' + arch_name:^{col_width * 4}}", flush=True)
    print(f"{'-' * col_width * 4}", flush=True)

    header = f"{'Target':<{col_width}}{'SMAPE':>{col_width}}{'R2':>{col_width}}{'RMSE':>{col_width}}"
    print(header, flush=True)
    print(f"{'-' * col_width * 4}", flush=True)

    for t in ALL_TARGETS:
        line = (
            f"{t:<{col_width}}"
            f"{smape_vals.get(t, float('nan')):>{col_width}.4f}"
            f"{r2_vals.get(t, float('nan')):>{col_width}.4f}"
            f"{rmse_vals.get(t, float('nan')):>{col_width}.2f}"
        )
        print(line, flush=True)

    mean_smape = np.mean(list(smape_vals.values()))
    mean_r2 = np.mean(list(r2_vals.values()))
    mean_rmse = np.mean(list(rmse_vals.values()))
    print(f"{'-' * col_width * 4}", flush=True)
    line = (
        f"{'mean':<{col_width}}"
        f"{mean_smape:>{col_width}.4f}"
        f"{mean_r2:>{col_width}.4f}"
        f"{mean_rmse:>{col_width}.2f}"
    )
    print(line, flush=True)
    print(flush=True)


def get_architecture_name(model_name):
    architecture = str(model_name.split("_")[0].split("/")[-1])
    if architecture.lower() in ["model", "2layer", "3layer", "latency", "resource"]:
        architecture = "dense"
    return architecture


def load_wrappers(path: str, device: torch.device) -> Sequence[BaseModelWrapper]:
    """Scan directory for *.config.json and load matching wrappers."""
    wrappers = []
    for config_path in sorted(
        glob.glob(os.path.join(path, "**", "*.config.json"), recursive=True)
    ):
        h5_path = config_path.replace(".config.json", ".weights.h5")
        pt_path = config_path.replace(".config.json", ".weights.pt")

        if os.path.exists(pt_path):
            wrapper = TorchModelWrapper()
            wrapper.load(config_path, pt_path)
        elif os.path.exists(h5_path):
            wrapper = KerasModelWrapper()
            wrapper.load(config_path, h5_path)
        else:
            print(f"Warning: No weights found for {config_path}, skipping.", flush=True)
            continue

        wrapper.model.to(device)
        wrappers.append(wrapper)
    return wrappers


def main():
    arg_parser = argparse.ArgumentParser(
        description="Evaluate a saved wrapper on models"
    )
    arg_parser.add_argument(
        "wrappers_dir",
        type=str,
        help="Path to the wrappers directory containing the saved configs and weights",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "exemplar"],
        required=True,
        help="Which data split to evaluate on. One of (train, val, test, exemplar)",
    )
    cli_args = arg_parser.parse_args()

    use_gpu = torch.cuda.is_available() and not FORCE_CPU
    current_device = torch.cuda.current_device() if use_gpu else ""
    device = torch.device(f"cuda:{current_device}" if use_gpu else "cpu")
    device_name = torch.cuda.get_device_name(current_device) if use_gpu else "CPU"

    wrappers = load_wrappers(cli_args.wrappers_dir, device)
    if not wrappers:
        print("No wrappers found in the specified directory.", flush=True)
        return

    json_data = read_from_json(
        get_split_json_patterns(cli_args.split),
        batch_size=JSON_BATCH_SIZE,
        max_workers=JSON_MAX_WORKERS,
    )
    meta_data, global_inputs, targets = get_global_data(
        json_data, normalize=True, max_workers=JSON_MAX_WORKERS
    )
    sequential_inputs = get_sequential_data(json_data, max_workers=JSON_MAX_WORKERS)

    ref_wrapper = wrappers[0]
    global_feature_labels = ref_wrapper.global_numerical_labels + list(
        ref_wrapper.global_categorical_maps.keys()
    )
    sequential_feature_labels = ref_wrapper.sequential_numerical_labels + list(
        ref_wrapper.sequential_categorical_maps.keys()
    )
    global_maps = getattr(ref_wrapper, "global_categorical_maps", {})
    sequential_maps = getattr(ref_wrapper, "sequential_categorical_maps", {})

    df = to_dataframe(
        meta_data=meta_data,
        global_inputs=global_inputs,
        sequential_inputs=sequential_inputs,
        global_categorical_maps=global_maps,
        sequential_categorical_maps=sequential_maps,
        targets=targets,
        max_workers=JSON_MAX_WORKERS,
    )
    df["architecture"] = df["model_name"].apply(get_architecture_name)
    unique_architectures = sorted(df["architecture"].unique())

    inputs_df = build_inputs_df(df, global_feature_labels, sequential_feature_labels)
    targets_df = df[ALL_TARGETS]

    # Per-architecture predictions
    arch_data = {
        arch: {target: {"y_true": [], "y_pred": []} for target in ALL_TARGETS}
        for arch in unique_architectures
    }
    for wrapper in wrappers:
        target_labels = wrapper.output_labels
        preds = predict(wrapper, inputs_df)

        for arch in unique_architectures:
            arch_mask = df["architecture"] == arch
            if not arch_mask.any():
                continue

            arch_targets = targets_df.loc[arch_mask].values
            arch_preds = preds[arch_mask]

            for i, target in enumerate(target_labels):
                target_idx = ALL_TARGETS.index(target)
                arch_data[arch][target]["y_true"].append(arch_targets[:, target_idx])
                arch_data[arch][target]["y_pred"].append(arch_preds[:, i])

    # Per-architecture breakdown
    all_y_true = {t: [] for t in ALL_TARGETS}
    all_y_pred = {t: [] for t in ALL_TARGETS}
    for arch in unique_architectures:
        smape_arch, r2_arch, rmse_arch = {}, {}, {}

        for target in ALL_TARGETS:
            y_true = (
                np.concatenate(arch_data[arch][target]["y_true"])
                if arch_data[arch][target]["y_true"]
                else np.array([])
            )
            y_pred = (
                np.concatenate(arch_data[arch][target]["y_pred"])
                if arch_data[arch][target]["y_pred"]
                else np.array([])
            )

            all_y_true[target].append(y_true)
            all_y_pred[target].append(y_pred)

            valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if valid.sum() > 0:
                smape_arch[target] = smape(y_true[valid], y_pred[valid])
                r2_arch[target] = r2(y_true[valid], y_pred[valid])
                rmse_arch[target] = rmse(y_true[valid], y_pred[valid])

        print_arch_summary(arch, smape_arch, r2_arch, rmse_arch)

    # Overall metrics
    overall_smape, overall_r2, overall_rmse = {}, {}, {}
    for target in ALL_TARGETS:
        y_true = np.concatenate(all_y_true[target])
        y_pred = np.concatenate(all_y_pred[target])

        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if valid.sum() > 0:
            overall_smape[target] = smape(y_true[valid], y_pred[valid])
            overall_r2[target] = r2(y_true[valid], y_pred[valid])
            overall_rmse[target] = rmse(y_true[valid], y_pred[valid])

    metrics = {
        "mean_smape": float(np.mean(list(overall_smape.values()))),
        "mean_r2": float(np.mean(list(overall_r2.values()))),
    }
    for target in ALL_TARGETS:
        metrics[f"smape_{target}"] = overall_smape.get(target, float("nan"))
        metrics[f"r2_{target}"] = overall_r2.get(target, float("nan"))
        metrics[f"rmse_{target}"] = overall_rmse.get(target, float("nan"))

    print_summary(metrics=metrics, platform=device_name)
