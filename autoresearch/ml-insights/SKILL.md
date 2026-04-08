---
name: ml-insights
description: Synthesized empirical lessons from past experiments. Updated after each keep/discard decision. Read this before planning a new experiment to avoid repeating known failures.
metadata:
  audience: agent
  workflow: experiment-planning
---

# Empirical Insights

**Last updated**: exp22 (commit dd44902) — SMAPE eps fix applied

**Best result**: mean SMAPE 1.79 with inflated SMAPE (eps=1 bug), true SMAPE ~6.3 with corrected eps

## Architecture Insights

- Hybrid Transformer (CLS + query attention) with learned CLS residual for resource targets works well
- DSP-gated features (layer_dsp_eligible, layer_dsp_multiplier) dramatically improved DSP predictions
- Per-layer reuse_factor override is same class of fix as precision parsing — both correct cascading errors

## Feature Insights

- Per-layer weight_bits from precision parsing (exp20) was the breakthrough for DSP
- Per-layer result_bits caused SMAPE regression (likely noise/collinearity with weight_bits) — discard
- Per-layer reuse_factor override gave modest improvement — keep for future iterations
- SMAPE eps=1.0 was inflating scores by ~3-4x for normalized targets — now corrected to eps=0.1

## Loss Function Insights

- MSLE remains the primary working loss; SMAPE direct loss saturates for extreme values

## Target-Specific Insights

- BRAM/DSP have highest SMAPE with corrected eps (19.0, 8.3) — need targeted improvement
- CYCLES/INTERVAL have lowest SMAPE (1.6-1.7) — already well-predicted
- FF/LUT intermediate SMAPE (3.6-3.8)

## Promising Directions

1. **Lower SMAPE target**: Focus on BRAM (worst target, SMAPE 19.0) — architecture change or targeted features
2. **idea-044** (high, radical): Auxiliary per-token HLS prediction heads for multi-task supervision
3. **idea-043** (medium): Deeper Transformer (4 layers) for better DSP R2
