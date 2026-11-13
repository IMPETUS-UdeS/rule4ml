import json

import pytest

from rule4ml.dataset import load, prepare


@pytest.fixture
def record():
    """Builder of synthetic synthesis records: a dense/relu chain of `blocks`."""
    return _record


@pytest.fixture
def make_dataset(tmp_path):
    """Dataset of synthetic records, one per block count, with LUT growing with depth."""

    def make(block_counts):
        records = [
            _record(blocks, lut=100.0 * blocks + idx, record_id=f"r{idx}")
            for idx, blocks in enumerate(block_counts)
        ]
        source = tmp_path / "records.json"
        source.write_text(json.dumps(records))
        prepare([source], tmp_path / "data.npz")
        return load(tmp_path / "data.npz")

    return make


def _record(blocks, hls=True, lut=500.0, record_id=None):
    def dense(n_in, n_out):
        return {
            "class_name": "Dense",
            "input_shape": [None, n_in],
            "output_shape": [None, n_out],
            "parameters": (n_in + 1) * n_out,
            "trainable_parameters": (n_in + 1) * n_out,
            "use_bias": True,
            "reuse_factor": 4,
        }

    def activation(name, n):
        return {
            "class_name": "Activation",
            "activation": name,
            "input_shape": [None, n],
            "output_shape": [None, n],
            "parameters": 0,
            "trainable_parameters": 0,
            "reuse_factor": 4,
        }

    report = {"bram": "2.0", "dsp": "10.0", "ff": "300.0", "lut": str(lut)}
    return {
        "meta_data": {"model_id": record_id or f"blocks{blocks}"},
        "model_config": [
            {**activation("linear", 16), "class_name": "InputLayer"},
            dense(16, 8),
            activation("relu", 8),
            *[
                layer
                for _ in range(blocks - 1)
                for layer in (dense(8, 8), activation("relu", 8))
            ],
            dense(8, 2),
            activation("linear", 2),
        ],
        "hls_config": {
            "Model": {
                "Precision": "ap_fixed<8, 4>",
                "ReuseFactor": 4,
                "Strategy": "Latency",
            },
            "clock_period": 10.0,
        },
        "resource_report": report,
        "hls_resource_report": report if hls else {},
        "latency_report": {
            "cycles_min": "40",
            "cycles_max": "56",
            "interval_min": "1",
            "interval_max": "3",
        },
        "target_part": "xczu9eg-ffvb1156-2-e",
    }
