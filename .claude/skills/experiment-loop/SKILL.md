---
name: experiment-loop
description: Procedural instructions for running, evaluating, and recording ML experiments in the rule4ml autoresearch framework.
metadata:
  audience: agent
  workflow: experiment-execution
---

# Experiment Loop Procedure

## Running an Experiment
```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
HASH=$(git rev-parse --short HEAD)
uv run autoresearch --branch-name ${BRANCH} --commit-hash ${HASH} > autoresearch/runs/${BRANCH}/${HASH}/run.log 2>&1 & PID=$!
```

## Safety Check (60 seconds after start)
```bash
sleep 60 && head -20 autoresearch/runs/${BRANCH}/${HASH}/run.log
```
Look for: training epoch lines, no Python stack trace, GPU device name printed.

## Wait for Completion
```bash
while kill -0 $PID 2>/dev/null; do sleep 60; done; echo "Done"
```

## Extract Results
```bash
grep -E "^(smape_|r2_|rmse_|num_epochs|training_seconds|total_seconds|peak_vram_mb|platform):" autoresearch/runs/${BRANCH}/${HASH}/run.log
```

If output is empty → likely crashed. Check:
```bash
tail -50 autoresearch/runs/${BRANCH}/${HASH}/run.log
```

## Recording Results
Add a row to `autoresearch/reports/results.tsv` (tab-separated). Fields:
- `agentic_model`: your model name
- `branch`, `commit`: from git
- `smape_mean`, `r2_mean`: scalars (-1.0 for crashes)
- Per-target SMAPE/R2/RMSE: record the full dict string from the log (e.g. `{"conv1d": 8.1, "dense": 10.4, "all architectures": 9.9}`), -1.0 for crashes
- `num_epochs`: from log (JSON dict of group→epoch count)
- `platform`: GPU name or "CPU"
- `vram_gb`: peak VRAM / 1024 (0.0 for crashes)
- `time_budget`: `TIME_BUDGET` value (0.0 for crashes)
- `status`: baseline | keep | discard | crash
- `description`: short summary of what changed

**Do NOT commit TSV files. Leave them untracked by git.**

## Decision Logic (Post-Experiment)
- **Clear improvement**: (R2 up AND SMAPE down AND RMSE down on ALL targets) → `keep`, continue building on this commit
- **Mixed result**: use judgment. Large mean SMAPE/R2 improvements with minor per-target regressions are worth keeping. Small improvements with significant regressions → `discard`
- **No improvement**: `discard`, git reset to previous commit
- **Crash**: try to fix and rerun once. If still crashes, report in `issues.tsv`

## Git Workflow
- After `keep`: stay on the commit, plan next experiment
- After `discard`: `git reset --hard HEAD~1` to undo the commit, then plan next experiment on the previous commit. **You only keep commits that improved**. Do not ignore this and keep the failed commits in the history.
- Every 4 experiments: record a medium/high-priority radical idea in `ideas.tsv`

## Updating Insights Skill
After each keep/discard decision, ALSO update the `ml-insights` skill with any new findings. Keep the skill concise, it's for quick context refresh, not a full experiment log.
