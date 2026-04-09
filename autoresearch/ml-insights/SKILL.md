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
- LR 1e-3, batch 256, weight_decay 1e-4 all optimal
- Lower weight_decay (5e-5) significantly hurt SMAPE (+0.9pp) — regularization is important
- CLS residual 0.25 optimal

## Promising Directions

1. Focus on DSP improvement (now the weakest target at 7.08 SMAPE)
2. Try adding gradient clipping or other regularization
3. Consider different learning rate warmup strategies
