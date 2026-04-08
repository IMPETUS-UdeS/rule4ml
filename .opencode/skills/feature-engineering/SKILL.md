---
name: feature-engineering
description: How to explore and extract new features from the hls4ml synthesis tool source code and the data pipeline.
metadata:
  audience: agent
  workflow: feature-engineering
---

# Feature Engineering Guide

## Available Data Sources

### 1. Training Data (JSON)
Path: `datasets/huggingface/wa-hls4ml/{train,val,test}/`
Each split has multiple `*.json` files containing model configs with:
- `model_config`: layer definitions, connectivity (`inbound_layers`), shapes
- `hls_config`: per-layer precision overrides, reuse factors, strategy
- `synth_results`: the 6 target values

Parse with: `rule4ml/parsers/data_parser.py` → `json_to_df()`

### 2. hls4ml Source Code
Path: `hls4ml/` (cloned repo with version tags)
Switch versions: `git -C hls4ml checkout v0.8.1` or `v1.1.0`

### 3. Data Parser
Path: `rule4ml/parsers/data_parser.py`

Prefer adding new feature extraction logic here, so it can be part of the packaged project itself, not just the autoresearch workflow.

#### Adding New Features

1. **Identify the feature** in hls4ml source or data JSON
2. **Implement extraction** in `data_parser.py`. Prefer `get_layers_data()` for per-layer sequential features, or `get_global_features()` for per-model global features.
3. **Add the feature name** to `SEQUENTIAL_FEATURE_LABELS` or `GLOBAL_FEATURE_LABELS` in `autoresearch/train.py`
