"""
Cache augmentation script for exp32: analytical II + dead global feature fix.

Adds two new features to address the two main remaining weaknesses:

1. layer_analytical_ii = layer_parameter_count / max(layer_reuse, 1)   [sequential col 20]
   - Directly encodes what INTERVAL/CYCLES measures: ceil(ops/parallelism)
   - hls4ml formula: Dense II = ceil(n_in*n_out/reuse) = ceil(params/reuse)
   - Currently the model sees params and reuse separately; providing the quotient
     explicitly should improve CYCLES/INTERVAL prediction for conv architectures.

2. total_analytical_ii_log = log1p(sum(layer_analytical_ii))   [global col 48]
   - Model-level sum of analytical IIs in log space.

3. Dead global feature fix: conv1d/conv2d filters/kernel_size/strides aggregates
   (cols 16-18 and 25-27 of g4_numerical_layer) are always zero due to parser bug.
   Fix by computing from sequential data filtered by layer type (conv1d=3, conv2d=4).
   The features are already correct per-layer in s1_numerical_layer; we just need
   to aggregate them into the dead global columns.

   IMPORTANT: We do NOT change feature names — just fix the values in-place. This
   means we need to use a NEW cache key that reflects the corrected features.
   Strategy: add a dummy "global_dead_features_fixed" string to the cache key
   computation. We achieve this by adding "dead_features_fixed_v1" to a new
   "marker" global feature that has value 0 but changes the hash.

   SIMPLER approach: fix in-place AND change cache key by treating corrected features
   as part of a new "augmented" feature list. The feature names stay the same so
   train.py doesn't need to change, but we save to a new cache key derived from the
   same feature list + a suffix marker.

   IMPLEMENTATION: append "_deadfix" to the tensor_cache_key computation by adding
   a sentinel to the feature_cols list. In train.py, add "dead_features_fixed_v1"
   to GLOBAL_FEATURE_LABELS (as a zero column dropped before model input, or better:
   fix the parser so data_parser correctly computes the aggregates and the train.py
   hash changes naturally).

   SIMPLEST: Just compute the fixed values and append as NEW global columns
   "conv1d_filters_mean_fixed", etc., while keeping the old zero columns.
   But this wastes capacity.

   FINAL DECISION: In-place fix + sentinel in feature list (add one dummy column
   "dead_features_fixed_v1" with value 0 to the end of GLOBAL_FEATURE_LABELS in
   train.py). This changes the hash so a new cache is used. global_input_shape
   goes from 48 to 49 dims. Model gets trained from scratch on corrected data.
   The 0-valued sentinel column is harmless and gets log1p(0)=0 → BatchNorm learns
   to ignore it.

   ACTUALLY SIMPLEST: just fix the 6 dead columns in-place (no sentinel needed)
   and save the fixed cache under the same key as if data_parser had been fixed.
   Use a fixed_features approach where the cache key includes only the feature names
   (same as before) but the VALUES are corrected. This is fine — the cache is just
   a performance optimization, not a ground truth. train.py uses its own feature
   list to compute the key; since we're not changing the list, we need a different
   approach to force a different key.

   RESOLUTION: Add "layer_analytical_ii" to SEQUENTIAL_FEATURE_LABELS in train.py
   (new feature → new hash → new cache), AND fix the dead global features in the new
   cache. The new hash will be different because sequential features changed.
   Dead feature values get corrected silently in the augmented cache.
"""
import os
import hashlib

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
# Exp31/exp32 feature labels
# Source cache: exp31 cache key (same global/seq features as exp28)
# Target cache: exp32 cache key (adds layer_analytical_ii to sequential)
# ---------------------------------------------------------------------------

GLOBAL_FEATURE_LABELS_BASE = [
    # Categorical (must appear first)
    "strategy", "board", "hls4ml_version", "vivado_version",
    # Numerical (48 dims)
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
]

SEQUENTIAL_FEATURE_LABELS_BASE = [
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
]

# exp32: adds layer_analytical_ii to sequential features
SEQUENTIAL_FEATURE_LABELS_EXP32 = SEQUENTIAL_FEATURE_LABELS_BASE + ["layer_analytical_ii"]

GLOBAL_CATEGORICAL_MAPS = {
    "strategy": None, "board": None, "hls4ml_version": None, "vivado_version": None,
}
SEQUENTIAL_CATEGORICAL_MAPS = {"layer_type": None}
NORMALIZE_TARGETS = False

# ---------------------------------------------------------------------------
# Column indices in s1_numerical_layer (20 dims, 0-indexed, categorical removed)
# ---------------------------------------------------------------------------
PARAMS_COL = 2      # layer_parameter_count
REUSE_COL = 9       # layer_reuse
FILTERS_COL = 4     # layer_filters
KERNEL_H_COL = 5    # layer_kernel_height
KERNEL_W_COL = 6    # layer_kernel_width
STRIDE_H_COL = 7    # layer_stride_height
STRIDE_W_COL = 8    # layer_stride_width

# Layer type integer codes (from data_parser.default_layer_type_map)
CONV1D_TYPE = 3
CONV2D_TYPE = 4

# Dead global feature column indices in g4_numerical_layer (48 dims, 0-indexed)
# = (position in GLOBAL_FEATURE_LABELS_BASE) - 4 (subtract 4 categoricals)
CONV1D_FILTERS_MEAN_COL = 16    # conv1d_filters_mean
CONV1D_KERNEL_MEAN_COL  = 17    # conv1d_kernel_size_mean
CONV1D_STRIDES_MEAN_COL = 18    # conv1d_strides_mean
CONV2D_FILTERS_MEAN_COL = 25    # conv2d_filters_mean
CONV2D_KERNEL_MEAN_COL  = 26    # conv2d_kernel_size_mean
CONV2D_STRIDES_MEAN_COL = 27    # conv2d_strides_mean

CACHE_DIR = "datasets/huggingface/wa-hls4ml/cache"


def main():
    # Source: exp31 cache (same features as exp28 — 48 global + 21 seq including layer_type)
    src_key = tensor_cache_key(
        GLOBAL_FEATURE_LABELS_BASE + ["sequential_inputs"] + SEQUENTIAL_FEATURE_LABELS_BASE,
        GLOBAL_CATEGORICAL_MAPS.keys(),
        SEQUENTIAL_CATEGORICAL_MAPS.keys(),
        NORMALIZE_TARGETS,
    )
    src_path = os.path.join(CACHE_DIR, f"{src_key}_train_val.pt")
    print(f"Source cache key: {src_key}")
    print(f"Source path:      {src_path}")

    # Target: exp32 cache (adds layer_analytical_ii + fixes dead global features)
    dst_key = tensor_cache_key(
        GLOBAL_FEATURE_LABELS_BASE + ["sequential_inputs"] + SEQUENTIAL_FEATURE_LABELS_EXP32,
        GLOBAL_CATEGORICAL_MAPS.keys(),
        SEQUENTIAL_CATEGORICAL_MAPS.keys(),
        NORMALIZE_TARGETS,
    )
    dst_path = os.path.join(CACHE_DIR, f"{dst_key}_train_val.pt")
    print(f"Target cache key: {dst_key}")
    print(f"Target path:      {dst_path}")

    if os.path.exists(dst_path):
        print("Target cache already exists — nothing to do.")
        return

    if not os.path.exists(src_path):
        print(f"ERROR: Source cache not found at {src_path}")
        print("Available caches:")
        for f in sorted(os.listdir(CACHE_DIR)):
            if f.endswith("_train_val.pt"):
                print(f"  {f}")
        return 1

    print(f"\nLoading source cache ...")
    cache = torch.load(src_path, weights_only=False, map_location="cpu")
    print("  Loaded.")

    new_inputs = {}
    for split in ("train", "val"):
        print(f"\n  Processing split: {split}")
        inp = cache["inputs"][split]

        s1 = inp["s1_numerical_layer"]          # [N, T, 20]
        s0 = inp["s0_layer_type_embedding"]      # [N, T, 1] integer layer type
        g4 = inp["g4_numerical_layer"].clone()   # [N, 48]  — clone so we can fix in-place

        N, T, _ = s1.shape

        # --- 1. Compute layer_analytical_ii ---
        params = s1[:, :, PARAMS_COL]            # [N, T]
        reuse = s1[:, :, REUSE_COL]              # [N, T]
        analytical_ii = params / reuse.clamp(min=1.0)  # [N, T]

        # --- 2. Fix dead global features ---
        # layer_type integers are stored in s0_layer_type_embedding: [N, T, 1]
        layer_types = s0.long()                  # [N, T] — s0 is already [N, T]

        conv1d_mask = (layer_types == CONV1D_TYPE).float()  # [N, T]
        conv2d_mask = (layer_types == CONV2D_TYPE).float()  # [N, T]

        # Padding mask: reuse==0 means padded (zero-padded layer rows)
        # Use a simple heuristic: layer with all-zero s1 row is padding
        nonpad_mask = (s1.abs().sum(dim=-1) > 0).float()   # [N, T]

        conv1d_layer_count = (conv1d_mask * nonpad_mask).sum(dim=1)   # [N]
        conv2d_layer_count = (conv2d_mask * nonpad_mask).sum(dim=1)   # [N]

        filters = s1[:, :, FILTERS_COL]          # [N, T]
        kernel_h = s1[:, :, KERNEL_H_COL]        # [N, T]
        kernel_w = s1[:, :, KERNEL_W_COL]        # [N, T]
        stride_h = s1[:, :, STRIDE_H_COL]        # [N, T]
        stride_w = s1[:, :, STRIDE_W_COL]        # [N, T]
        # kernel_size for 1d is kernel_h (single spatial dimension)
        # kernel_size for 2d is kernel_h * kernel_w (product of spatial dims)
        kernel_1d = kernel_h
        kernel_2d = kernel_h * kernel_w
        stride_1d = stride_h
        stride_2d = stride_h * stride_w

        def safe_mean(vals, mask):
            """Compute per-sample mean of vals where mask==1; returns 0 for samples with no such layers."""
            masked_sum = (vals * mask * nonpad_mask).sum(dim=1)
            count = (mask * nonpad_mask).sum(dim=1).clamp(min=1.0)
            return masked_sum / count * (count > 0).float()

        conv1d_filters_mean  = safe_mean(filters,    conv1d_mask)  # [N]
        conv1d_kernel_mean   = safe_mean(kernel_1d,  conv1d_mask)  # [N]
        conv1d_strides_mean  = safe_mean(stride_1d,  conv1d_mask)  # [N]
        conv2d_filters_mean  = safe_mean(filters,    conv2d_mask)  # [N]
        conv2d_kernel_mean   = safe_mean(kernel_2d,  conv2d_mask)  # [N]
        conv2d_strides_mean  = safe_mean(stride_2d,  conv2d_mask)  # [N]

        # Fix dead global columns in-place
        g4[:, CONV1D_FILTERS_MEAN_COL] = conv1d_filters_mean
        g4[:, CONV1D_KERNEL_MEAN_COL]  = conv1d_kernel_mean
        g4[:, CONV1D_STRIDES_MEAN_COL] = conv1d_strides_mean
        g4[:, CONV2D_FILTERS_MEAN_COL] = conv2d_filters_mean
        g4[:, CONV2D_KERNEL_MEAN_COL]  = conv2d_kernel_mean
        g4[:, CONV2D_STRIDES_MEAN_COL] = conv2d_strides_mean

        # --- 3. Append analytical_ii to sequential features ---
        new_s1 = torch.cat([s1, analytical_ii.unsqueeze(2)], dim=2)  # [N, T, 21]

        # --- Diagnostics ---
        n_conv1d = (conv1d_layer_count > 0).sum().item()
        n_conv2d = (conv2d_layer_count > 0).sum().item()
        print(f"    Samples with conv1d layers: {n_conv1d}/{N}")
        print(f"    Samples with conv2d layers: {n_conv2d}/{N}")
        print(f"    conv1d_filters_mean stats: "
              f"min={conv1d_filters_mean[conv1d_filters_mean>0].min().item():.1f}, "
              f"max={conv1d_filters_mean.max().item():.1f}, "
              f"mean={conv1d_filters_mean[conv1d_filters_mean>0].mean().item():.1f} (nonzero only)")
        print(f"    analytical_ii stats: min={analytical_ii.min():.2f}, "
              f"max={analytical_ii.max():.1f}, mean={analytical_ii.mean():.2f}")
        print(f"    s1: {s1.shape} -> {new_s1.shape}")
        print(f"    g4: (unchanged shape, dead cols fixed in-place)")

        new_inp = dict(inp)
        new_inp["s1_numerical_layer"] = new_s1
        new_inp["g4_numerical_layer"] = g4
        new_inputs[split] = new_inp

    # Update meta
    new_meta = dict(cache["meta"])
    new_meta["feature_columns"] = (
        GLOBAL_FEATURE_LABELS_BASE + ["sequential_inputs"] + SEQUENTIAL_FEATURE_LABELS_EXP32
    )

    new_cache = {
        "inputs": new_inputs,
        "targets": cache["targets"],
        "meta": new_meta,
    }

    print(f"\nSaving augmented cache to {dst_path} ...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(new_cache, dst_path)
    print("Done.")

    # Summary of dead feature fix statistics
    print("\n--- Dead Feature Fix Summary ---")
    print("Before fix: conv1d_filters/kernel/strides = always 0 (parser bug issue-003)")
    print("After fix: computed from sequential features filtered by layer_type==conv1d/conv2d")
    print("train.py can now use the same GLOBAL_FEATURE_LABELS — cache key differs due to")
    print("  layer_analytical_ii addition to SEQUENTIAL_FEATURE_LABELS.")


if __name__ == "__main__":
    main()
