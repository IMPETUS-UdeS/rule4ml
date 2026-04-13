"""
Cache augmentation script for exp18: analytical BRAM estimation features.

Instead of rebuilding the tensor cache from raw JSON (~7h), load the exp8 cache
(c5f1757d199f_train_val.pt) and compute the two new features from existing tensors:
  - layer_estimated_bram_blocks = ceil(params * weight_bits / 18000)   [sequential col 20]
  - total_estimated_bram_log    = log1p(sum over layers of above)       [global col 48]

Then save under the exp18 cache key so train.py finds it immediately.
"""
import os
import sys
import hashlib

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Inline tensor_cache_key (matches prepare.py exactly)
# ---------------------------------------------------------------------------

def tensor_cache_key(feature_cols, global_cat_keys, seq_cat_keys, normalize):
    h = hashlib.sha256(repr(
        list(feature_cols)
        + sorted(global_cat_keys)
        + sorted(seq_cat_keys)
        + [normalize]
    ).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Exp18 feature labels (must match train.py exactly)
# ---------------------------------------------------------------------------

GLOBAL_FEATURE_LABELS = [
    "strategy", "board", "hls4ml_version", "vivado_version",
    "bit_width", "reuse_mean", "reuse_max", "weight_bits_min",
    "weight_bits_max", "weight_bits_mean", "total_table_size",
    "dense_inputs_mean", "dense_outputs_mean", "dense_parameters_mean",
    "dense_reuse_mean", "dense_reuse_max", "dense_count",
    "conv1d_inputs_mean", "conv1d_outputs_mean", "conv1d_parameters_mean",
    "conv1d_filters_mean", "conv1d_kernel_size_mean", "conv1d_strides_mean",
    "conv1d_reuse_mean", "conv1d_reuse_max", "conv1d_count",
    "conv2d_inputs_mean", "conv2d_outputs_mean", "conv2d_parameters_mean",
    "conv2d_filters_mean", "conv2d_kernel_size_mean", "conv2d_strides_mean",
    "conv2d_reuse_mean", "conv2d_reuse_max", "conv2d_count",
    "batchnormalization_inputs_mean", "batchnormalization_outputs_mean",
    "batchnormalization_parameters_mean", "batchnormalization_count",
    "add_count", "concatenate_count", "dropout_count",
    "relu_count", "sigmoid_count", "tanh_count",
    "softmax_inputs_mean", "softmax_outputs_mean", "softmax_count",
    "total_add", "total_mult", "total_lookup", "total_logical",
    "total_estimated_bram_log",
]

SEQUENTIAL_FEATURE_LABELS = [
    "layer_type",
    "layer_input_size", "layer_output_size", "layer_parameter_count",
    "layer_trainable_parameter_count", "layer_filters",
    "layer_kernel_height", "layer_kernel_width",
    "layer_stride_height", "layer_stride_width",
    "layer_reuse", "layer_op_add", "layer_op_mult",
    "layer_op_lookup", "layer_op_logical",
    "layer_weight_bits",
    "layer_multiplier", "layer_uses_lut_table", "layer_pipelined",
    "layer_dsp_eligible", "layer_dsp_multiplier",
    "layer_estimated_bram_blocks",
]

GLOBAL_CATEGORICAL_MAPS = {
    "strategy": None, "board": None, "hls4ml_version": None, "vivado_version": None,
}
SEQUENTIAL_CATEGORICAL_MAPS = {
    "layer_type": None,
}

NORMALIZE_TARGETS = False

_BRAM18_BITS = 18000.0

# Layer_parameter_count is column 2 in s1_numerical_layer (0-indexed, after removing categorical)
# layer_weight_bits is column 14
PARAMS_COL = 2
WEIGHT_BITS_COL = 14

CACHE_DIR = "datasets/huggingface/wa-hls4ml/cache"
EXP8_CACHE = os.path.join(CACHE_DIR, "c5f1757d199f_train_val.pt")

# ---------------------------------------------------------------------------

def main():
    # Compute exp18 cache key
    feature_hash = tensor_cache_key(
        GLOBAL_FEATURE_LABELS + ["sequential_inputs"] + SEQUENTIAL_FEATURE_LABELS,
        GLOBAL_CATEGORICAL_MAPS.keys(),
        SEQUENTIAL_CATEGORICAL_MAPS.keys(),
        NORMALIZE_TARGETS,
    )
    out_path = os.path.join(CACHE_DIR, f"{feature_hash}_train_val.pt")
    print(f"Exp18 cache key: {feature_hash}")
    print(f"Output path:     {out_path}")

    if os.path.exists(out_path):
        print("Cache already exists — nothing to do.")
        return

    print(f"Loading exp8 cache from {EXP8_CACHE} ...")
    cache = torch.load(EXP8_CACHE, weights_only=False, map_location="cpu")
    print("  Loaded.")

    new_inputs = {}
    for split in ("train", "val"):
        print(f"  Augmenting split: {split}")
        inp = cache["inputs"][split]

        s1 = inp["s1_numerical_layer"]          # [N, T, 20]
        g4 = inp["g4_numerical_layer"]           # [N, 48]

        params = s1[:, :, PARAMS_COL]            # [N, T]
        weight_bits = s1[:, :, WEIGHT_BITS_COL]  # [N, T]

        bram_est = torch.ceil(params * weight_bits / _BRAM18_BITS)  # [N, T]

        # Append as new sequential column
        new_s1 = torch.cat([s1, bram_est.unsqueeze(2)], dim=2)      # [N, T, 21]

        # Global: sum over layers then log1p
        total_bram_log = torch.log1p(bram_est.sum(dim=1))           # [N]
        new_g4 = torch.cat([g4, total_bram_log.unsqueeze(1)], dim=1) # [N, 49]

        new_inp = dict(inp)
        new_inp["s1_numerical_layer"] = new_s1
        new_inp["g4_numerical_layer"] = new_g4
        new_inputs[split] = new_inp

        print(f"    s1: {s1.shape} -> {new_s1.shape}")
        print(f"    g4: {g4.shape} -> {new_g4.shape}")
        print(f"    bram_est sample stats: min={bram_est.min():.1f}, max={bram_est.max():.1f}, mean={bram_est.mean():.2f}")
        print(f"    total_bram_log stats:  min={total_bram_log.min():.2f}, max={total_bram_log.max():.2f}, mean={total_bram_log.mean():.2f}")

    # Update meta
    new_meta = dict(cache["meta"])
    new_meta["feature_columns"] = GLOBAL_FEATURE_LABELS + ["sequential_inputs"] + SEQUENTIAL_FEATURE_LABELS

    new_cache = {
        "inputs": new_inputs,
        "targets": cache["targets"],
        "meta": new_meta,
    }

    print(f"Saving augmented cache to {out_path} ...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(new_cache, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
