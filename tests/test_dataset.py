import json

import numpy as np
import pytest

from rule4ml.dataset import load, prepare, split


def test_prepare_roundtrip(tmp_path, record):
    sequential = record(blocks=2, hls=False)
    del sequential["model_config"][0]  # Sequential records lack the InputLayer
    source = tmp_path / "records.json"
    source.write_text(json.dumps([record(blocks=1), sequential]))
    prepare([source], tmp_path / "data.npz")
    dataset = load(tmp_path / "data.npz")

    assert dataset.offsets.tolist() == [0, 4, 10]  # linear activations skipped

    codes, vocab = dataset.layer_types
    assert vocab[codes].tolist() == (
        ["inputlayer", "dense", "relu", "dense"]
        + ["inputlayer", "dense", "relu", "dense", "relu", "dense"]
    )

    glob = {
        name: dataset.global_values[:, i] for i, name in enumerate(dataset.global_names)
    }
    assert glob["dense_parameters_mean"].tolist() == pytest.approx(
        [(136 + 18) / 2, (136 + 72 + 18) / 3]
    )
    assert glob["integer_bits"].tolist() == [4, 4]

    codes, vocab = dataset.categoricals["board"]
    assert vocab[codes].tolist() == ["zcu102", "zcu102"]

    assert dataset.latencies.tolist() == [[48.0, 2.0], [48.0, 2.0]]
    assert dataset.resources["resource_report"][:, 3].tolist() == [500.0, 500.0]
    assert np.isnan(dataset.resources["hls_resource_report"][1]).all()


def test_load_invalid_offsets(tmp_path, record):
    source = tmp_path / "records.json"
    source.write_text(json.dumps([record(blocks=1)]))
    prepare([source], tmp_path / "data.npz")

    with np.load(tmp_path / "data.npz") as data:
        arrays = dict(data)
    arrays["offsets"] = np.array([4, 0])
    np.savez(tmp_path / "bad.npz", **arrays)

    with pytest.raises(ValueError, match="offsets"):
        load(tmp_path / "bad.npz")


def test_prepare_nested_reports(tmp_path, record):
    raw = record(blocks=1)
    raw["resource_report"] = {
        "VivadoSynthReport": {"LUT": "700", "FF": "80"},
        "CSynthesisReport": {"LUT": "900"},
    }
    del raw["hls_resource_report"]

    source = tmp_path / "records.json"
    source.write_text(json.dumps([raw]))
    prepare([source], tmp_path / "data.npz")
    dataset = load(tmp_path / "data.npz")

    assert dataset.targets(["lut", "ff"], "resource_report").tolist() == [[700, 80]]
    assert dataset.targets(["lut"], "hls_resource_report").tolist() == [[900]]


def test_split_layers(make_dataset):
    dataset = make_dataset([1, 2, 3, 4, 5, 6])
    train, val = split(dataset, 0.5, seed=0)

    ids = sorted([*train.record_ids.tolist(), *val.record_ids.tolist()])
    assert ids == sorted(dataset.record_ids.tolist())

    original = dict(
        zip(dataset.record_ids, np.split(dataset.layer_values, dataset.offsets[1:-1]))
    )

    for part in (train, val):
        for record_id, layers in zip(
            part.record_ids, np.split(part.layer_values, part.offsets[1:-1])
        ):
            assert np.array_equal(layers, original[record_id])
