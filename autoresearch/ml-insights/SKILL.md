---
name: ml-insights
description: Synthesized empirical lessons from past experiments. Updated after each keep/discard decision. Read this before planning a new experiment to avoid repeating known failures.
metadata:
  audience: agent
  workflow: experiment-planning
---

# Empirical Insights

**Last updated**: exp23 (commit 48388b5) — deeper Transformer 4 layers

**Best result**: mean SMAPE 6.08 with corrected eps (deeper Transformer, 4 layers)

## Architecture Insights

- Hybrid Transformer (CLS + query attention) with learned CLS residual for resource targets works well
- Deeper Transformer (4 layers vs 3) improved SMAPE 6.33->6.08 and DSP SMAPE 8.25->7.16
- DSP-gated features (layer_dsp_eligible, layer_dsp_multiplier) remain the key breakthrough

## Feature Insights

- Per-layer weight_bits from precision parsing (exp20) was the breakthrough for DSP
- Per-layer result_bits caused SMAPE regression (likely noise/collinearity with weight_bits) — discarded
- Per-layer reuse_factor override gave modest improvement — keep
- SMAPE eps=1.0 was inflating scores by ~3-4x for normalized targets — now corrected to eps=0.1

## Loss Function Insights

- MSLE remains the primary working loss; SMAPE direct loss saturates for extreme values

## Target-Specific Insights

- BRAM is worst target (SMAPE 19.1) — needs architecture or feature breakthrough
- DSP improved with deeper Transformer (7.16 SMAPE from 8.25)
- CYCLES/INTERVAL well-predicted (SMAPE 1.6-1.7)
- FF/LUT intermediate (SMAPE 3.4-3.5)

## Promising Directions

1. **Focus on BRAM**: Architecture change or targeted features to improve worst target (19.1 SMAPE)
2. **idea-044** (high, radical): Auxiliary per-token HLS prediction heads for multi-task supervision
3. **idea-045** (medium): Larger Transformer (d_model=192, nhead=6) — more capacity for BRAM/DSP
