---
name: ml-insights
description: Synthesized empirical lessons from past experiments. Updated after each keep/discard decision. Read this before planning a new experiment to avoid repeating known failures.
metadata:
  audience: agent
  workflow: experiment-planning
---

# Empirical Insights

**Last updated**: exp24 retry completed (commit 218e389) — larger Transformer regressed, reverted to 4-layer

**Best result**: mean SMAPE 6.08 (exp23, commit 48388b5) — 4-layer Transformer

## Architecture Insights

- 4-layer Transformer (idea-043) improved SMAPE 6.33->6.08, DSP SMAPE 8.25->7.16
- Larger Transformer (d_model=192, nhead=6) regressed SMAPE to 6.44 — more capacity not helpful
- Hybrid Transformer (CLS + query attention) with learned CLS residual remains best architecture

## Feature Insights

- Per-layer weight_bits from precision parsing was DSP breakthrough
- Per-layer result_bits caused regression — discarded
- Per-layer reuse_factor override gave modest improvement — keep
- SMAPE eps fixed from 1.0 to ~0.1, true SMAPE baseline ~6.3

## Target-Specific Insights

- BRAM is worst target (SMAPE ~19) — needs targeted improvement
- DSP improved with deeper Transformer but still high (7.16)
- CYCLES/INTERVAL well-predicted (SMAPE ~1.6-1.7)
- FF/LUT intermediate (SMAPE ~3.4-3.5)

## Promising Directions

1. **idea-044** (high priority, radical): Auxiliary per-token HLS prediction heads — multi-task supervision forcing token representations to be physically meaningful
2. Focus on BRAM improvement (architecture change or targeted features)
3. Explore different learning rate or batch size for 4-layer model
