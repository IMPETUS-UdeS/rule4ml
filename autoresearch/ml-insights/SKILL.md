---
name: ml-insights
description: Synthesized empirical lessons from past experiments. Updated after each keep/discard decision. Read this before planning a new experiment to avoid repeating known failures.
metadata:
  audience: agent
  workflow: experiment-planning
---

# Empirical Insights

**Last updated**: exp31 (commit 20e014f) — NEW BEST SMAPE 6.04

**Best result**: mean SMAPE **6.04** (exp31, commit 20e014f) — scheduler T_0=30, 4-layer Transformer

## Architecture Insights

- 4-layer Transformer (idea-043) improved SMAPE 6.33->6.08
- Scheduler T_0=30 (idea-051) improved SMAPE 6.08->6.04 — **new best**
- CLS residual 0.25 is optimal (tested 0.15, 0.25, 0.5)
- Larger Transformer (d_model=192) regressed — not helpful

## Hyperparameter Insights

- LR 1e-3 optimal (tested 2e-3, 5e-4 — both worse)
- Batch 256 optimal (tested 512 — worse)
- T_0=30 better than T_0=20 (slower decay helped convergence)

## Feature Insights

- Per-layer weight_bits from precision parsing was DSP breakthrough
- Per-layer reuse_factor override gave modest improvement — keep
- Per-layer result_bits caused regression — discarded

## Target-Specific Insights

- BRAM is worst target (SMAPE ~19.6) — needs targeted improvement
- DSP improved significantly with T_0=30 (7.16->6.20)
- CYCLES/INTERVAL well-predicted (SMAPE ~1.6)
- FF/LUT intermediate (SMAPE ~3.6-3.7)

## Promising Directions

1. **Focus on BRAM**: Architecture change or targeted features to improve worst target (19.6 SMAPE)
2. **idea-044** (high priority, radical): Auxiliary per-token HLS prediction heads — multi-task supervision
3. Try dropout tuning or different warmup strategy
