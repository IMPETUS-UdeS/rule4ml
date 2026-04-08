---
name: ml-insights
description: Synthesized empirical lessons from past experiments. Updated after each keep/discard decision. Read this before planning a new experiment to avoid repeating known failures.
metadata:
  audience: agent
  workflow: experiment-planning
---

# Empirical Insights

**Last updated**: experiment 21 baseline (commit 9db5f79)

**Best result**: mean SMAPE 1.79 (exp20, 8f2466c), mean R2 0.923 (exp20)

## Architecture Insights

- Hybrid Transformer (CLS + query attention) with learned CLS residual for resource targets works well
- DSP-gated features (layer_dsp_eligible, layer_dsp_multiplier) dramatically improved DSP predictions
- Deeper Transformer (4 layers vs 3) should help DSP R2 according to idea-043

## Feature Insights

- Per-layer weight_bits from precision parsing (exp20) was the breakthrough: DSP SMAPE 4.41→0.74
- Per-layer result_bits (idea-046) and per-layer reuse_factor override (idea-047) are untested high-priority features
- LUT-routed mult (hls_lut_mult_est) caused regression — most layers are DSP-eligible, adding noise

## Loss Function Insights

- MSLE remains the primary working loss; SMAPE direct loss saturates for extreme values
- L1 in log space (log-MAE) crashed due to ROCm GPU hang (exp6)

## Target-Specific Insights

- DSP and LUT are strongest (R2 > 0.98), BRAM/FF moderate (R2 0.96-0.98), CYCLES/INTERVAL weakest (R2 ~0.80)
- CYCLES/INTERVAL may benefit from per-layer reuse override fixes (idea-047)
- FF has highest SMAPE (1.84) among resource targets — architecture or feature change needed

## Promising Directions

1. **idea-047** (high priority): Per-layer reuse_factor override from hls_config — may fix CYCLES/INTERVAL
2. **idea-043** (medium): Deeper Transformer (4 layers) for better DSP R2
3. **idea-046** (high): Per-layer result_bits for activation precision — complements weight_bits breakthrough
4. **idea-044** (high, radical): Auxiliary per-token HLS prediction heads for multi-task supervision
