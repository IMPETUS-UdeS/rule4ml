from dataclasses import replace

import numpy as np
import pytest
import torch

from rule4ml.dataset import split
from rule4ml.features import DEFAULT_CLOCK_PERIOD
from rule4ml.models.architectures import (
    GNNSettings,
    MLPSettings,
    TorchGNN,
    TorchMLP,
    TorchTransformer,
    TransformerSettings,
)
from rule4ml.models.callbacks import EarlyStopping
from rule4ml.models.metrics import TorchParametricR2, r2
from rule4ml.models.scaling import (
    AsinhInputScaler,
    BoardPercentScaler,
    BoardScaler,
    Log1pScaler,
)
from rule4ml.parsers.data_parser import (
    default_board_map,
    default_layer_type_map,
    default_strategy_map,
)
from rule4ml.surrogates import MultiSurrogate, TorchSurrogate, TrainSettings

GLOBAL_LABELS = [
    "dense_count",
    "dense_parameters_mean",
    "bit_width",
    "reuse_mean",
    "clock_period",
]
SEQUENTIAL_LABELS = ["layer_input_size", "layer_output_size", "layer_parameter_count"]
GLOBAL_MAPS = {"strategy": default_strategy_map, "board": default_board_map}
SEQUENTIAL_MAPS = {"layer_type": default_layer_type_map}


def _torch_gnn(linear_output=False):
    settings = GNNSettings(  # width-one and unused trailing embeddings
        [4], [8], [1, 1, 2], [1, 2], [8], [False], [0.0], linear_output=linear_output
    )
    return TorchGNN(settings, *_shapes(), GLOBAL_MAPS, SEQUENTIAL_MAPS, device="cpu")


def _torch_transformer(linear_output=False):
    settings = TransformerSettings([4], [1, 2], [2], 8, 2, 1, 16, 0.0, dense_layers=[8])
    settings.linear_output = linear_output
    return TorchTransformer(
        settings, *_shapes(), GLOBAL_MAPS, SEQUENTIAL_MAPS, device="cpu"
    )


def _torch_mlp(linear_output=False):
    settings = MLPSettings([1, 1, 2], [4], [8], [0.0], linear_output=linear_output)
    return TorchMLP(settings, _shapes()[0], (None, 1), GLOBAL_MAPS, device="cpu")


def _keras_mlp(linear_output=False):
    from rule4ml.models.architectures import KerasMLP

    settings = MLPSettings([2, 2], [4], [8], [0.0], linear_output=linear_output)
    return KerasMLP(settings, _shapes()[0], (None, 1), GLOBAL_MAPS)


def _shapes():
    return (
        (None, len(GLOBAL_LABELS) + len(GLOBAL_MAPS)),
        (None, len(SEQUENTIAL_LABELS) + len(SEQUENTIAL_MAPS)),
        (None, 1),
    )


@pytest.mark.parametrize(
    ("surrogate_class", "build", "sequential", "scaler_class"),
    [
        (TorchSurrogate, _torch_gnn, True, None),
        (TorchSurrogate, _torch_transformer, True, None),
        (TorchSurrogate, _torch_mlp, False, BoardScaler),
        (TorchSurrogate, _torch_mlp, False, BoardPercentScaler),
        ("keras", _keras_mlp, False, Log1pScaler),
    ],
)
def test_fit_save_load(
    make_dataset, tmp_path, surrogate_class, build, sequential, scaler_class
):
    surrogate = _surrogate(surrogate_class)
    train, val = split(make_dataset([1, 2, 3, 1, 2, 3, 4, 5]), 0.25, seed=0)

    scaler = None
    if scaler_class == Log1pScaler:
        scaler = Log1pScaler()
    elif scaler_class is not None:
        scaler = scaler_class({"zcu102": {"lut": 1000}}, default_board_map)

    surrogate.set_model(
        build(linear_output=scaler_class is Log1pScaler),  # Log1p centers targets
        GLOBAL_LABELS,
        ["lut"],
        SEQUENTIAL_LABELS if sequential else (),
        report="resource_report",
        scaler=scaler,
        input_scaler=AsinhInputScaler(standardize_layers=sequential),  # reload keeps it
    )

    surrogate.fit(train, TrainSettings(num_epochs=1, batch_size=4), val_dataset=val)
    assert surrogate.evaluate(val)["lut"]["n"] == len(val)

    loaded = _reload(surrogate, tmp_path)
    assert np.array_equal(loaded.predict_dataset(val), surrogate.predict_dataset(val))
    assert loaded.clock_periods == [10.0]


def test_predict_clock_sweep(make_dataset, top_quarks, hls_config):
    surrogate = TorchSurrogate()
    surrogate.set_model(
        _torch_gnn(),
        GLOBAL_LABELS,
        ["lut"],
        SEQUENTIAL_LABELS,
        report="resource_report",
    )
    surrogate.fit(make_dataset([1, 2, 3]), TrainSettings(num_epochs=1, batch_size=4))
    estimator = MultiSurrogate()
    estimator.add_surrogate(surrogate)

    swept = estimator.predict(top_quarks)
    with pytest.warns(UserWarning, match=rf"clock periods \[{DEFAULT_CLOCK_PERIOD}\]"):
        configured = estimator.predict(top_quarks, hls_configs=hls_config)

    assert set(swept["Clock Period"]) == {10.0}  # the training clock
    assert list(configured["Clock Period"]) == [DEFAULT_CLOCK_PERIOD]

    surrogate.io_types = {"dense": ["io_parallel"]}
    with pytest.warns(UserWarning, match="not io_stream"):
        estimator.predict(
            top_quarks,
            hls_configs={**hls_config, "IOType": "io_stream", "ClockPeriod": 10},
        )


def test_input_scaler_integer_layers():
    scaler = AsinhInputScaler(standardize_layers=True)
    scaler.fit(np.array([[1.0, 8.0], [3.0, 2.0]], dtype=np.float32))
    layers = np.array([[[1, 8], [3, 2], [0, 0]]])

    assert np.allclose(
        scaler.transform(layers), scaler.transform(layers.astype(np.float32))
    )


@pytest.mark.parametrize(
    ("surrogate_class", "build", "sequential"),
    [
        (TorchSurrogate, _torch_gnn, True),
        (TorchSurrogate, _torch_mlp, False),
        ("keras", _keras_mlp, False),
    ],
)
def test_load_linear_output(make_dataset, tmp_path, surrogate_class, build, sequential):
    surrogate = _surrogate(surrogate_class)
    model = build(linear_output=True)

    if isinstance(surrogate, TorchSurrogate):
        torch.nn.init.zeros_(model.output_layer.weight)
        torch.nn.init.constant_(model.output_layer.bias, -10.0)
    else:
        kernel, bias = model.layers[-1].get_weights()
        model.layers[-1].set_weights([np.zeros_like(kernel), np.full_like(bias, -10)])

    sequential_labels = SEQUENTIAL_LABELS if sequential else ()
    surrogate.set_model(
        model, GLOBAL_LABELS, ["lut"], sequential_labels, report="resource_report"
    )

    loaded = _reload(surrogate, tmp_path)
    assert np.allclose(loaded.predict_dataset(make_dataset([1, 2, 3])), -10.0)


def _surrogate(surrogate_class):
    if surrogate_class == "keras":
        pytest.importorskip("tensorflow")
        from rule4ml.surrogates import KerasSurrogate

        surrogate_class = KerasSurrogate

    return surrogate_class()


def _reload(surrogate, tmp_path):
    surrogate.save(tmp_path / "bundle")
    config = next((tmp_path / "bundle").glob("*.config.json"))
    weights = next(p for p in (tmp_path / "bundle").iterdir() if p != config)

    loaded = type(surrogate)()
    loaded.load(str(config), str(weights))

    return loaded


@pytest.mark.parametrize("patience", [1, 10])  # stops early, runs out of epochs
def test_fit_restores_best(make_dataset, patience):
    train = make_dataset([1, 2, 3, 4])
    resources = train.resources["resource_report"].copy()
    resources[:, 3] = 0.0  # validation LUT pulls against training
    val = replace(train, resources={**train.resources, "resource_report": resources})

    model = _torch_mlp(linear_output=True)
    for parameter in model.parameters():  # only the output bias learns
        torch.nn.init.zeros_(parameter)
    surrogate = TorchSurrogate()
    surrogate.set_model(model, GLOBAL_LABELS, ["lut"], report="resource_report")

    settings = TrainSettings(
        num_epochs=3, batch_size=4, learning_rate=0.1, loss_function="mse"
    )
    history = surrogate.fit(
        train, settings, val_dataset=val, callbacks=[EarlyStopping(patience=patience)]
    )

    val_loss = np.mean(surrogate.predict_dataset(val) ** 2)
    assert val_loss == pytest.approx(min(history["val"]["loss"]))


@pytest.mark.parametrize("scaler_class", [BoardScaler, Log1pScaler])
def test_fit_scaled_metrics(make_dataset, scaler_class):
    dataset = make_dataset([1, 2, 3, 4, 5, 6, 7, 8])
    codes, vocab = dataset.categoricals["board"]
    vocab = np.append(vocab, "pynq-z2")
    codes = np.where(np.arange(len(codes)) % 2, len(vocab) - 1, codes)
    dataset = replace(
        dataset, categoricals={**dataset.categoricals, "board": (codes, vocab)}
    )

    train, val = split(dataset, 0.5, seed=0)

    if scaler_class == Log1pScaler:
        scaler = Log1pScaler()
    else:
        resources = {"zcu102": {"lut": 1000}, "pynq-z2": {"lut": 50}}
        scaler = scaler_class(resources, default_board_map)

    surrogate = TorchSurrogate()
    surrogate.set_model(
        _torch_mlp(linear_output=scaler_class is Log1pScaler),
        GLOBAL_LABELS,
        ["lut"],
        report="resource_report",
        scaler=scaler,
    )
    metric = TorchParametricR2(0, scaler=scaler, device=torch.device("cpu"))

    settings = TrainSettings(num_epochs=1, batch_size=2, metrics=[metric])
    history = surrogate.fit(train, settings, val_dataset=val)

    true = val.targets(["lut"], "resource_report")[:, 0]
    predicted = surrogate.predict_dataset(val)[:, 0]
    assert history["val"]["r2_0"][-1] == pytest.approx(r2(true, predicted), rel=1e-4)

    if scaler_class == Log1pScaler:  # statistics from the training split only
        lut = train.targets(["lut"], "resource_report")[:, 0]
        assert scaler.mu["lut"] == pytest.approx(np.log1p(lut).mean())


def test_predict_empty_graph(make_dataset):
    dataset = make_dataset([1, 1])
    start, end = dataset.offsets[1:]
    dataset = replace(
        dataset,
        layer_values=dataset.layer_values[start:],
        layer_types=(dataset.layer_types[0][start:], dataset.layer_types[1]),
        offsets=np.array([0, 0, end - start]),
    )

    surrogate = TorchSurrogate()
    surrogate.set_model(
        _torch_gnn(),
        GLOBAL_LABELS,
        ["lut"],
        SEQUENTIAL_LABELS,
        report="resource_report",
    )

    with pytest.raises(ValueError, match="Record 0 .* no layers"):
        surrogate.predict_dataset(dataset)


def test_fit_zero_features(make_dataset):
    surrogate = TorchSurrogate()
    settings = GNNSettings([4], [8], [2, 2], [2], [8], [False], [0.0])
    shapes = ((None, len(GLOBAL_LABELS) + 2), (None, 2), (None, 1))
    model = TorchGNN(settings, *shapes, GLOBAL_MAPS, SEQUENTIAL_MAPS, device="cpu")
    surrogate.set_model(
        model,
        GLOBAL_LABELS,
        ["lut"],
        ["layer_parameter_count"],
        report="resource_report",
    )

    with pytest.raises(ValueError, match="padding"):  # relu layers have no parameters
        surrogate.fit(make_dataset([1, 2]), TrainSettings(num_epochs=1))


def test_training_partial_targets(make_dataset):
    dataset = make_dataset([1, 2, 3])
    resources = dataset.resources["resource_report"].copy()
    resources[1, 3] = np.nan
    latencies = dataset.latencies.copy()
    latencies[2, 0] = np.nan
    dataset = replace(
        dataset,
        resources={**dataset.resources, "resource_report": resources},
        latencies=latencies,
    )

    maps = {
        "strategy": {"resource": 1, "latency": 2},
        "board": {"unused": 1, "zcu102": 2},
    }
    model = TorchMLP(
        MLPSettings([2, 2], [4], [8], [0.0]),
        _shapes()[0],
        (None, 2),
        maps,
        device="cpu",
    )

    surrogate = TorchSurrogate()
    surrogate.set_model(
        model,
        GLOBAL_LABELS,
        ["cycles", "lut"],
        report="resource_report",
        scaler=BoardScaler({"zcu102": {"cycles": 10, "lut": 100}}, maps["board"]),
    )

    inputs, targets, _ = surrogate.training_data(dataset)

    np.testing.assert_allclose(targets, [[4.8, 1.0]])
    for key in model.global_input_keys["categorical"]:
        assert inputs[key].tolist() == [2]
