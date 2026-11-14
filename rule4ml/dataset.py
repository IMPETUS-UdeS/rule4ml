import dataclasses
import hashlib
import json
import os

import numpy as np

CATEGORICALS = ("strategy", "board", "precision", "hls4ml_version", "vivado_version")
REPORTS = ("resource_report", "hls_resource_report")
RESOURCES = ("bram", "dsp", "ff", "lut")
LATENCIES = ("cycles_min", "cycles_max", "interval_min", "interval_max")


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Features and raw targets of synthesis records, layers stored ragged."""

    global_values: np.ndarray  # (N, G) float32
    global_names: tuple
    categoricals: dict  # name -> (codes (N,), vocab)
    layer_values: np.ndarray  # (L, F) float32
    layer_names: tuple
    layer_types: tuple  # (codes (L,), vocab)
    offsets: np.ndarray  # (N + 1,), layers of record i: offsets[i]:offsets[i + 1]
    resources: dict  # report -> (N, len(RESOURCES)) float64, NaN when missing
    latencies: np.ndarray  # (N, len(LATENCIES)) float64, NaN when missing
    record_ids: np.ndarray
    sources: tuple  # "file:sha256" of each source, in order

    def __len__(self):
        return len(self.global_values)

    def targets(self, labels, report=None):
        """Raw targets (N, len(labels)), resources read from the given report."""

        columns = []
        for label in labels:
            if label in ("cycles", "interval"):  # min/max average, as v1 and v2 predict
                bounds = [LATENCIES.index(f"{label}_{end}") for end in ("min", "max")]
                columns.append(self.latencies[:, bounds].mean(axis=1))
            elif label in LATENCIES:
                columns.append(self.latencies[:, LATENCIES.index(label)])
            elif label in RESOURCES:
                if report not in REPORTS:
                    raise ValueError(
                        f"Resource targets need a report, one of {REPORTS}"
                    )
                columns.append(self.resources[report][:, RESOURCES.index(label)])
            else:
                raise ValueError(f"Unknown target: {label}")

        return np.stack(columns, axis=1)

    def columns(self, labels, *, per_layer=False):
        """Numerical features (N, k), or (L, k) per layer, in the order of labels."""

        names, values = (
            (self.layer_names, self.layer_values)
            if per_layer
            else (self.global_names, self.global_values)
        )

        missing = [label for label in labels if label not in names]
        if missing:
            raise ValueError(f"Features missing from the dataset: {missing}")

        return values[:, [names.index(label) for label in labels]]

    def encode(self, label, mapping, *, per_layer=False):
        """Category codes (N,), or (L,) per layer, as values of a model's map."""

        if per_layer:
            if label != "layer_type":
                raise ValueError(f"Datasets have no layer category {label}")
            codes, vocab = self.layer_types
        else:
            if label not in self.categoricals:
                raise ValueError(f"Datasets have no category {label}")
            codes, vocab = self.categoricals[label]

        table = np.array(
            [mapping.get(str(value), -1) for value in vocab], dtype=np.int64
        )
        mapped = table[codes]

        if (mapped < 0).any():
            missing = sorted(
                str(value) for value in vocab[np.unique(codes[mapped < 0])]
            )
            raise ValueError(f"{label} values missing from the model map: {missing}")

        return mapped

    def pad_layers(self, layer_rows):
        """Per-layer rows (L, ...) as zero-padded per-record rows (N, depth, ...)."""

        if np.ndim(layer_rows) == 0 or len(layer_rows) != self.offsets[-1]:
            raise ValueError(f"Expected {self.offsets[-1]} layer rows")

        counts = np.diff(self.offsets)
        record = np.repeat(np.arange(len(self)), counts)
        position = np.arange(len(record)) - self.offsets[record]
        padded = np.zeros(
            (len(self), counts.max(initial=0), *layer_rows.shape[1:]),
            dtype=layer_rows.dtype,
        )
        padded[record, position] = layer_rows

        return padded

    def subset(self, rows):
        """Records at the given indices, in that order."""

        rows = np.asarray(rows, dtype=np.int64)
        counts = np.diff(self.offsets)[rows]
        starts = np.cumsum(counts) - counts
        layers = np.repeat(self.offsets[:-1][rows] - starts, counts) + np.arange(
            counts.sum()
        )

        return dataclasses.replace(
            self,
            global_values=self.global_values[rows],
            categoricals={
                name: (codes[rows], vocab)
                for name, (codes, vocab) in self.categoricals.items()
            },
            layer_values=self.layer_values[layers],
            layer_types=(self.layer_types[0][layers], self.layer_types[1]),
            offsets=np.concatenate([[0], np.cumsum(counts)]),
            resources={
                report: values[rows] for report, values in self.resources.items()
            },
            latencies=self.latencies[rows],
            record_ids=self.record_ids[rows],
        )


def split(dataset, val_ratio, seed=0):
    """Seeded random (train, validation) split of a Dataset."""

    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    rows = np.random.default_rng(seed).permutation(len(dataset))
    n_val = round(len(dataset) * val_ratio)
    if not 0 < n_val < len(dataset):
        raise ValueError(f"val_ratio {val_ratio} leaves an empty split")

    return dataset.subset(np.sort(rows[n_val:])), dataset.subset(np.sort(rows[:n_val]))


def prepare(json_paths, out_path):
    """Extract features and raw targets from synthesis JSON files into one .npz."""

    from rule4ml.features import record_features  # imports Keras and Torch

    global_names, layer_names = None, None
    global_rows, layer_rows, layer_counts, record_ids, sources = [], [], [], [], []
    categoricals = {name: [] for name in CATEGORICALS}
    layer_types = []
    resources = {report: [] for report in REPORTS}
    latencies = []

    for path in json_paths:
        with open(path, "rb") as json_file:
            content = json_file.read()
        sources.append(
            f"{os.path.basename(path)}:{hashlib.sha256(content).hexdigest()}"
        )

        records = json.loads(content)
        del content

        for record in records if isinstance(records, list) else [records]:
            if record is None:  # a few published records are null
                continue

            global_inputs, layers = record_features(record)
            if not layers:
                raise ValueError(f"Record without layers in {path}")

            if global_names is None:
                global_names = [k for k in global_inputs if k not in CATEGORICALS]
                layer_names = [k for k in layers[0] if k != "layer_type"]
            if [k for k in global_inputs if k not in CATEGORICALS] != global_names:
                raise ValueError(f"Unexpected global features in {path}")

            global_rows.append([global_inputs[k] for k in global_names])
            for name in CATEGORICALS:
                categoricals[name].append(global_inputs[name] or "")
            layer_rows.extend([layer[k] for k in layer_names] for layer in layers)
            layer_types.extend(layer["layer_type"] for layer in layers)
            layer_counts.append(len(layers))

            targets = {report: _targets(record, report) for report in REPORTS}
            for report in REPORTS:
                resources[report].append([targets[report][k] for k in RESOURCES])
            latencies.append([targets[REPORTS[0]][k] for k in LATENCIES])

            meta_data = record.get("meta_data", {})
            record_ids.append(meta_data.get("model_id") or meta_data.get("uuid", ""))
        del records

    if not global_rows:
        raise ValueError("No records found.")

    arrays = {
        "sources": np.array(sources),
        "global_values": np.array(global_rows, dtype=np.float32),
        "global_names": np.array(global_names),
        "layer_values": np.array(layer_rows, dtype=np.float32),
        "layer_names": np.array(layer_names),
        "offsets": np.concatenate([[0], np.cumsum(layer_counts)]).astype(np.int64),
        "latencies": np.array(latencies, dtype=np.float64),
        "record_ids": np.array(record_ids),
    }

    for name, values in [*categoricals.items(), ("layer_type", layer_types)]:
        vocab, codes = np.unique(np.array(values), return_inverse=True)
        arrays[f"vocab_{name}"] = vocab
        arrays[f"codes_{name}"] = codes.astype(np.int32)
    for report, values in resources.items():
        arrays[f"resources_{report}"] = np.array(values, dtype=np.float64)

    tmp_path = f"{out_path}.tmp.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, out_path)


def load(path):
    """Load and validate a prepared .npz dataset."""

    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)

    for name in ["global_names", "layer_names", "sources"]:
        if arrays[name].ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")

    n_records = len(arrays["global_values"])
    n_layers = len(arrays["layer_values"])
    shapes = {
        "global_values": (n_records, len(arrays["global_names"])),
        "layer_values": (n_layers, len(arrays["layer_names"])),
        "latencies": (n_records, len(LATENCIES)),
        "record_ids": (n_records,),
        **{f"resources_{report}": (n_records, len(RESOURCES)) for report in REPORTS},
    }

    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} has shape {arrays[name].shape}, expected {shape}")

    offsets = arrays["offsets"]
    if (
        offsets.dtype.kind != "i"
        or offsets.shape != (n_records + 1,)
        or offsets[0] != 0
        or offsets[-1] != n_layers
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError(f"Inconsistent layer offsets in {path}")

    for name in [*CATEGORICALS, "layer_type"]:
        codes, vocab = arrays[f"codes_{name}"], arrays[f"vocab_{name}"]
        length = n_layers if name == "layer_type" else n_records
        if (
            vocab.ndim != 1
            or codes.dtype.kind != "i"
            or codes.shape != (length,)
            or (length and (codes.min() < 0 or codes.max() >= len(vocab)))
        ):
            raise ValueError(f"Invalid {name} codes in {path}")

    return Dataset(
        global_values=arrays["global_values"],
        global_names=tuple(arrays["global_names"].tolist()),
        categoricals={
            name: (arrays[f"codes_{name}"], arrays[f"vocab_{name}"])
            for name in CATEGORICALS
        },
        layer_values=arrays["layer_values"],
        layer_names=tuple(arrays["layer_names"].tolist()),
        layer_types=(arrays["codes_layer_type"], arrays["vocab_layer_type"]),
        offsets=offsets,
        resources={report: arrays[f"resources_{report}"] for report in REPORTS},
        latencies=arrays["latencies"],
        record_ids=arrays["record_ids"],
        sources=tuple(arrays["sources"].tolist()),
    )


def _targets(record, report):
    from rule4ml.parsers.data_parser import get_prediction_targets

    nested_reports = {
        "resource_report": "VivadoSynthReport",
        "hls_resource_report": "CSynthesisReport",
    }

    resources = record.get("resource_report") or {}
    if set(nested_reports.values()) & resources.keys():  # raw data_gen layout
        resources = resources.get(nested_reports[report]) or {}
    else:
        resources = record.get(report) or {}

    return get_prediction_targets(
        {
            "resource_report": resources,
            "latency_report": record.get("latency_report") or {},
        }
    )
