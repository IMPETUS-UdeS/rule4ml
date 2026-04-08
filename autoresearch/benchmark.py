import argparse
import glob
import os
from typing import Sequence

os.environ.setdefault("HSA_ENABLE_SDMA", "0")  # Set before torch/HSA runtime initializes to prevent ROCm GPU hangs on RDNA3
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow verbose logging

import numpy as np
import torch

from autoresearch.prepare import (DATA_DIR, FORCE_CPU, JSON_BATCH_SIZE,
                                  JSON_MAX_WORKERS, build_inputs_df, predict,
                                  print_summary, r2, rmse, smape)
from rule4ml.models.wrappers import (BaseModelWrapper, KerasModelWrapper,
                                     TorchModelWrapper)
from rule4ml.parsers.data_parser import (get_global_data, get_sequential_data,
                                         read_from_json, to_dataframe)


def load_wrappers(path: str, device: torch.device) -> Sequence[BaseModelWrapper]:
    """Scan directory for *.config.json and load matching wrappers."""
    wrappers = []
    for config_path in sorted(glob.glob(os.path.join(path, "**", "*.config.json"), recursive=True)):
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
        description="Evaluate a saved checkpoint on exemplar models"
    )
    arg_parser.add_argument(
        "wrappers_dir", type=str,
        help="Path to the wrappers directory containing the saved configs and weights"
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
        os.path.join(
            DATA_DIR, "exemplar", "*exemplar_models.json"
        ),
        batch_size=JSON_BATCH_SIZE,
        max_workers=JSON_MAX_WORKERS,
    )
    meta_data, global_inputs, targets = get_global_data(
        json_data, normalize=True, max_workers=JSON_MAX_WORKERS
    )
    sequential_inputs = get_sequential_data(json_data, max_workers=JSON_MAX_WORKERS)

    smape_vals, r2_vals, rmse_vals = {}, {}, {}
    for wrapper in wrappers:
        global_feature_labels = wrapper.global_numerical_labels + list(wrapper.global_categorical_maps.keys())
        sequential_feature_labels = wrapper.sequential_numerical_labels + list(wrapper.sequential_categorical_maps.keys())
        target_labels = wrapper.output_labels
        global_maps = getattr(wrapper, "global_categorical_maps", {})
        sequential_maps = getattr(wrapper, "sequential_categorical_maps", {})

        df = to_dataframe(
            meta_data=meta_data,
            global_inputs=global_inputs,
            sequential_inputs=sequential_inputs,
            global_categorical_maps=global_maps,
            sequential_categorical_maps=sequential_maps,
            targets=targets,
            max_workers=JSON_MAX_WORKERS,
        )
        inputs_df = build_inputs_df(df, global_feature_labels, sequential_feature_labels)
        targets_df = df[target_labels]

        preds = predict(wrapper, inputs_df)
        for i, target in enumerate(target_labels):
            if target in targets_df:
                target_values = targets_df[target].values
                pred_values = preds[:, i]

                valid = ~np.isnan(target_values) & ~np.isnan(pred_values)
                if valid.sum() == 0:
                    print(f"Warning: no valid samples for target '{target}', skipping.", flush=True)
                    continue

                smape_vals[target] = smape(target_values[valid], pred_values[valid])
                r2_vals[target] = r2(target_values[valid], pred_values[valid])
                rmse_vals[target] = rmse(target_values[valid], pred_values[valid])

    metrics = {
        "mean_smape": float(np.mean(list(smape_vals.values()))),
        "mean_r2": float(np.mean(list(r2_vals.values()))),
    }
    for target in smape_vals:
        metrics[f"smape_{target}"] = float(smape_vals[target])
    for target in r2_vals:
        metrics[f"r2_{target}"] = float(r2_vals[target])
    for target in rmse_vals:
        metrics[f"rmse_{target}"] = float(rmse_vals[target])

    print_summary(metrics=metrics, platform=device_name)
