---
name: ml-insights
description: Synthesized empirical lessons from past experiments. Updated after each keep/discard decision. Read this before planning a new experiment to avoid repeating known failures.
metadata:
  audience: agent
  workflow: experiment-planning
---

# Empirical Insights

**Last updated**: exp33 (commit 1b22f38) — NEW BEST SMAPE 5.85

**Best result**: mean SMAPE **5.85** (exp33, commit 1b22f38) — MSLE + Huber loss

## Architecture Insights

- 4-layer Transformer (idea-043) + T_0=30 scheduler is the base
- MSLE + Huber loss component (idea-053) is the latest breakthrough

## Loss Insights

- MSLE + 0.1*Huber loss improved SMAPE 6.04->5.85
- BRAM SMAPE improved 19.56->18.28 (-1.28pp) due to Huber robustness to outliers
- FF and LUT also improved with Huber loss
- DSP slightly worsened but overall mean improved significantly

## Hyperparameter Insights

- T_0=30 optimal for 4-layer Transformer
- LR 1e-3, batch 256 optimal
- CLS residual 0.25 optimal

## Feature Insights

- Per-layer weight_bits from precision parsing was DSP breakthrough
- Per-layer reuse_factor override gave modest improvement — keep
- SMAPE eps fixed from 1.0 to ~0.1, true SMAPE baseline ~6.3

## Target-Specific Insights

- BRAM improved significantly with Huber loss (18.28 SMAPE from 19.56)
- DSP is now weakest target (7.08 SMAPE) — may need targeted improvement
- FF/LUT intermediate (~3.1-3.5)
- CYCLES/INTERVAL well-predicted (~1.5)

## Promising Directions

1. Tune Huber delta parameter (currently 0.5)
2. Try pure Huber loss without MSLE
3. Focus on DSP improvement
