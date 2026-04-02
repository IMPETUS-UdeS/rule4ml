# About

This is your Operating Manual to do your own research.

## Setup

1. **Agree on a run tag**: propose a tag based on UTC time using `date` (YEAR MONTH DAY T HH MM SS Z, e.g. `20260310T173250Z`). The branch `agent-<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b agent-<tag>` from the base `agent` branch. This new branch is where you will experiment.
3. **Read the in-scope files**: Read these files for project context:
    - `autoresearch/prepare.py` — fixed constants, data loading, evaluation harness. **Do not modify.**
    - `autoresearch/train.py` — baseline training script. Your primary entry point.
    - `rule4ml/models/architectures.py` — model class definitions. Keep existing archtitectures as reference and add new ones as needed.
    - `rule4ml/models/wrappers.py` — training wrappers and dataset building.
    - `rule4ml/parsers/data_parser.py` — data parsing pipeline and feature extraction.
4. **Know your resources**: The following are available for you to read and use:
    - `datasets/huggingface/wa-hls4ml/` — training, validation and test data (feather + JSON).
    - `hls4ml/` — a clone of the hls4ml synthesis tool repository. It contains version tags corresponding to the `hls4ml_version` values in the training data. Use `git -C hls4ml tag` to list them and `git -C hls4ml checkout <tag>` to switch between versions and read their contents. **Do not modify files inside `hls4ml/`.**
5. **Initialize .tsv files**: Create any missing `.tsv` files with just the header row. A new baseline should be recorded inside `results.tsv` after the first run.
6. **Confirm first**: **ALWAYS** confirm the setup with the human.

## Experimentation

The training script runs for a **fixed time budget** `TIME_BUDGET` defined inside `autoresearch/prepare.py` (wall clock training time, excluding startup and evaluation). Run it with:

```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
HASH=$(git rev-parse --short HEAD)
uv run autoresearch/train.py --branch-name ${BRANCH} --commit-hash ${HASH} > autoresearch/runs/${BRANCH}-${HASH}.log 2>&1 & PID=$!
```

**What you CAN do:**

- Read any file in the codebase
- Checkout different tags in the `hls4ml/` repository
- Modify training code. Everything is fair game: input features, model architecture and size, optimizer, loss functions, hyperparameters, training loop, etc.
- Choose how to group prediction targets per experiment: one predictor for all targets, one predictor per target, or any grouping in between. A given experiment can focus its changes on a specific target or subset of targets if that seems promising. However, every experiment **MUST** always produce predictions for all 6 targets overall.
- Create new files to support your experiments (new feature modules, new model classes, utilities, etc.)
- Modify `rule4ml/` source files to add new capabilities. When doing so, **prefer adding new classes or functions** rather than modifying existing ones, to preserve backward compatibility with deployed models
- Record novel ideas for future experiments in `autoresearch/reports/ideas.tsv` with clear hypotheses and potential next steps. This is a great place to store ideas that are not worth trying right now, but might be worth exploring after more information from current experiments
- Report on any non-immediate bugs, problems or roadblocks in `autoresearch/reports/issues.tsv`. The human is tasked with resolving the recorded issues. If an immediate breaking problem prevents you from running experiments, attempt to fix it; if you cannot, report directly to the human
- Record your decision logic in `autoresearch/reports/decisions.tsv` with a clear description of what you did and the rationale behind changes

**What you CANNOT do:**

- Modify `autoresearch/prepare.py` — this is the locked evaluation harness. It defines the metrics, the data loading, and the constants that make results comparable across runs
- Modify files inside `hls4ml/` — this is a read-only reference
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`. If a missing package is required for an experiment, log a detailed report in `issues.tsv` so the human can review and add it for future runs
- Break the evaluation harness. The `evaluate` function in `autoresearch/prepare.py` must produce valid results at the end of training

**The goal is clear: Improve prediction accuracy over all targets (BRAM, DSP, FF, LUT, CYCLES, INTERVAL), using only input available before HLS synthesis.** Since the time budget is fixed, you don't need to worry about training time. Accuracy is the only objective.

**Primary metrics**: mean SMAPE (lower is better) and mean R2 (higher is better), across all 6 targets. You also have access to the individual SMAPE and R2 values for each target, which should help analyze trade-offs and identify near-misses.

**Creativity and exploration**: Do not let this become a hyperparameter-tuning-only exercise. The best improvements might come from creative, out-of-the-box changes to how the input is represented, the model architecture, or the training process. Don't be afraid to try bold ideas, even if they add complexity, as long as they have a chance of drastically improving the results. Some experiments can be simple tweaks too — use your judgment to decide what is worth trying.

**The first run**: Your very first run should always be to establish the baseline, running `train.py` as-is and recording the results with a `baseline` status.

**Subsequent runs**: After the first run, use existing reports to inform future experiments, specifically to avoid repeating clearly failing ideas. Keep in mind that some ideas (near-misses) might be worth trying again from a different angle. You can checkout a previous branch and commit to read the code and better understand what was attempted.

## Outputs

Once the script finishes, it prints a summary:

```text
---
mean_smape:            0.5419
smape_bram:            0.4654
smape_dsp:             0.4314
smape_ff:              0.8431
smape_lut:             0.5123
smape_cycles:          0.6871
smape_interval:        0.3121
mean_r2:               0.4579
r2_bram:               0.1387
r2_dsp:                0.8942
r2_ff:                 0.6387
r2_lut:                0.4432
r2_cycles:             0.1685
r2_interval:           0.4641
rmse_bram:             120.45
rmse_dsp:              85.32
rmse_ff:               300.12
rmse_lut:              150.67
rmse_cycles:           1500.67
rmse_interval:         1523.89
num_epochs:            {"bram": 35, "dsp-lut": 40, "ff": 35, "cycles": 30, "interval": 30}
training_seconds:      3600.00
total_seconds:         3700.00
peak_vram_mb:          8000.00
platform:              AMD Radeon RX 7800 XT
```

You can extract the key metrics from the log file. `num_epochs` will both show the number of epochs, and how many predictors were trained (**one predictor per target, or if some/all targets trained jointly**).

## Reporting

### Ideas and Issues

During any stage of the process, log ideas in `autoresearch/reports/ideas.tsv` and issues in `autoresearch/reports/issues.tsv`. These files are tab-separated with the following columns:

**ideas.tsv**:

```text
id    author    category    description    created_at    priority    status    attempt_count    last_attempt    notes
```

- `priority`: subjective expected impact — one of [`low`, `medium`, `high`]. Reflects expected impact, not probability of success
- `attempt_count`: increment every time you try to implement the same idea, even from a different angle. Use notes to clarify differences in approach
- `last_attempt`: either a timestamp of the last attempt or `branch:commit` if the idea was accepted

**issues.tsv**:

```text
id    author    category    description    created_at    severity    encounters    status    notes
```

- `severity`: a subjective severity level for the issue, which can be one of [`low`, `medium`, `high`]. This should help the user decide which issues to address first
- `encounters`: increment every time the same issue is hit. This can help you identify recurring issues and reproducible bugs that need attention

Other columns exist in both files:

- `id`: unique identifier (`idea-<uuid>` or `issue-<uuid>`). `<uuid>` can be generated using any method that guarantees uniqueness (uuid4, row number, etc.)
- `author`: the agentic model name if the idea/issue is created autonomously by the agent. If an entry is manually created by the user, the value will be human. Never fill with human yourself
- `category`: a high-level category label. For ideas, one of [`hyperparameter`, `architecture`, `optimizer`, `training`, `loss`, `data`, `features`, `other`]. For issues, one of [`training_instability`, `performance`, `infrastructure`, `dataset`, `evaluation`, `logic`, `other`]
- `description`: the hypothesis behind the idea or a short description of the issue
- `created_at`: timestamp of creation
- `status`: for ideas, one of [`open`, `in_progress`, `rejected`, `accepted`]. For issues, one of [`open`, `in_progress`, `resolved`, `wont_fix`]
- `notes`: free-form field for insights, follow-up thoughts, or lessons learned

### Decision Logic

- SMAPE decrease, RMSE decrease, **and** R2 increase over all targets is a **clear improvement**, always `keep`
- Mean SMAPE and/or mean R2 improvement but with some target regressions is a **mixed result**. Use your judgment to decide whether to `keep` or `discard`.
- In general, mixed results across targets and means are a `discard`. Do record the changes that improved targets in `ideas.tsv` with detailed notes
- If no improvement, `discard` and try a different angle
- Occasionally, a radical change might result in an expected drop in one or several metrics. Again, use your judgment to decide if the change is worth keeping for future iterations. You can alway `discard` the change and record it as a near-miss in `ideas.tsv` with expanded notes on what you think went wrong and how it could be improved in the future

All decisions must be recorded in `autoresearch/reports/decisions.tsv`:

```text
agentic_model    branch    timestamp    decision    reasoning    related_idea    related_issue
```

1. the name of the model the agent is using as (e.g. gemini-3-pro, gemini-2.5-flash, opus-4-6, sonnet-4-5, gpt-5-3-codex, etc.)
2. branch: the git branch this experiment was run on
3. timestamp of when the decision was made
4. short text telling what the decision was
5. clear and concise explanation of why — thought process, trade-offs considered, insights from the experiment
6. `id` from `ideas.tsv` if applicable, otherwise `N/A`. This helps link your decisions back to the original ideas and provides context for future agents
7. `id` from `issues.tsv` if applicable, otherwise `N/A`. This helps link your decisions back to any problems or bugs encountered during the experiment

### Results

When an experiment finishes, log it to `autoresearch/reports/results.tsv`. This file **MUST** be tab-separated. Do not use commas or spaces as delimiters.

Previous entries with N/A values indicate that the agent was not asked to record that metric at the time. Currently, the TSV has a header row and the following columns:

```text
agentic_model    branch    commit    smape_mean    smape_bram    smape_dsp    smape_ff    smape_lut    smape_cycles    smape_interval    r2_mean    r2_bram    r2_dsp    r2_ff    r2_lut    r2_cycles    r2_interval    rmse_bram    rmse_dsp    rmse_ff    rmse_lut    rmse_cycles    rmse_interval    num_epochs    platform    vram_gb    time_budget    status    description
```

- the name of the model the agent is using
- branch: the git branch this experiment was run on
- git commit hash (short, 7 chars)
- mean SMAPE across all 6 targets, -1.0 for crashes
- SMAPE for each target, -1.0 for crashes
- mean R2 across all 6 targets, -1.0 for crashes
- R2 for each target, -1.0 for crashes
- RMSE for each target, -1.0 for crashes
- dictionary mapping target group names to number of epochs trained
- platform that the experiment ran on (e.g. "NVIDIA RTX 3090", "NVIDIA A100 40GB", etc.). Log "CPU" if no GPU was used
- peak vram usage in GB — use 0.0 for crashes or N/A if on CPU
- `TIME_BUDGET` in seconds — use 0.0 for crashes
- status: `baseline`, `keep`, `discard`, or `crash`
- short text description of what this experiment tried

## The Loop

The experiment runs on the dedicated branch.

**LOOP FOREVER**:

1. Every time an experiment finishes, you MUST re-read this `AGENTS.md` file to refresh your state and ensure you are following the protocol correctly.
2. Look at the git state: the current branch and commit you are on.
3. Every 4 experiments, you **MUST** think of and record in `ideas.tsv`, with `medium` to `high` priority, a more radical change to explore different angles. You are **HIGHLY ENCOURAGED** to try it as soon as you can.
4. Plan the next experiment. This can be an idea from `ideas.tsv` or a new one you come up with by looking at the code, the training data, the available resources, results so far, or just thinking hard about potential improvements.
5. Implement the experiment — modify `autoresearch/train.py` and create any supporting files, classes, functions needed.
6. git commit and get the short hash
7. Run the experiment with the issued command
8. Check the log once after 60 seconds to ensure it's running and not crashing immediately. **DO NOT SPAM CHECK COMMANDS** — trust the code to run and wait for the running process to finish
9. While waiting, use a portion of `TIME_BUDGET` (few minutes tops, **NOT THE ENTIRE** `TIME_BUDGET`) to plan future ideas, review the codebase for new angles, or inspect the training data and available resources. Record findings in `ideas.tsv` or `issues.tsv`.
10. Poll for completion: `while kill -0 $PID 2>/dev/null; do sleep 60; done; echo "Done"`. Read out the results using `grep` on `autoresearch/runs/${BRANCH}-${HASH}.log`
11. If the output is empty, the run likely crashed. Run `tail -n 50 autoresearch/runs/${BRANCH}-${HASH}.log` to read the stack trace and attempt a fix. If you cannot fix it after a few attempts, give up and report to the human.
12. Record the results in `results.tsv`. Update `ideas.tsv` if an existing idea was tried. Update `issues.tsv` if a non-breaking issue was encountered. **Do not commit the TSV files — leave them untracked by git.**
13. If the all-target metrics improve, keep the commit and continue building on it.
14. Otherwise, you git reset back to where you started

**Timeout**: If a run significantly exceeds `TIME_BUDGET` (2 times or more) with no progress logged, kill it and treat it as a failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working **indefinitely** until manually stopped. You are autonomous. If you run out of ideas, think harder — search the web if you have permission, research papers and existing methods, re-read the in-scope files for new angles, inspect the training data directly, browse the `hls4ml/` repository across its version tags, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
