import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from rule4ml.dataset import split
from rule4ml.models.architectures import (
    GNNSettings,
    MLPSettings,
    TorchGNN,
    TorchMLP,
)
from rule4ml.models.scaling import BoardPercentScaler, BoardScaler, Log1pScaler
from rule4ml.models.wrappers import TorchModelWrapper, TrainSettings
from rule4ml.parsers.data_parser import (
    default_board_map,
    default_layer_type_map,
    default_strategy_map,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only

GLOBAL_LABELS = ["dense_count", "dense_parameters_mean", "bit_width", "reuse_mean"]
SEQUENTIAL_LABELS = ["layer_input_size", "layer_output_size", "layer_parameter_count"]
GLOBAL_MAPS = {"strategy": default_strategy_map, "board": default_board_map}
SEQUENTIAL_MAPS = {"layer_type": default_layer_type_map}


def _torch_gnn(linear_output=False):
    settings = GNNSettings(
        [4], [8], [2, 2], [2], [8], [False], [0.0], linear_output=linear_output
    )
    return TorchGNN(settings, *_shapes(), GLOBAL_MAPS, SEQUENTIAL_MAPS, device="cpu")


def _torch_mlp(linear_output=False):
    settings = MLPSettings([2, 2], [4], [8], [0.0], linear_output=linear_output)
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


BUILDS = [
    (TorchModelWrapper, _torch_gnn, True),
    (TorchModelWrapper, _torch_mlp, False),
    ("keras", _keras_mlp, False),
]


@pytest.mark.parametrize(("wrapper_class", "build", "sequential"), BUILDS)
@pytest.mark.parametrize(
    "scaler_class", [None, BoardScaler, BoardPercentScaler, Log1pScaler]
)
def test_fit_save_load(
    make_dataset, tmp_path, wrapper_class, build, sequential, scaler_class
):
    wrapper = _wrapper(wrapper_class)
    train, val = split(make_dataset([1, 2, 3, 1, 2, 3, 4, 5]), 0.25, seed=0)

    scaler = None
    if scaler_class == Log1pScaler:
        scaler = Log1pScaler()
    elif scaler_class is not None:
        scaler = scaler_class({"zcu102": {"lut": 1000}}, default_board_map)

    wrapper.set_model(
        build(linear_output=scaler_class is Log1pScaler),  # Log1p centers targets
        GLOBAL_LABELS,
        ["lut"],
        SEQUENTIAL_LABELS if sequential else (),
        report="resource_report",
        scaler=scaler,
    )

    wrapper.fit(train, TrainSettings(num_epochs=1, batch_size=4), val_dataset=val)
    assert wrapper.evaluate(val)["lut"]["n"] == len(val)

    loaded = _reload(wrapper, tmp_path)
    assert np.array_equal(loaded.predict_dataset(val), wrapper.predict_dataset(val))


@pytest.mark.parametrize(("wrapper_class", "build", "sequential"), BUILDS)
def test_load_linear_output(make_dataset, tmp_path, wrapper_class, build, sequential):
    wrapper = _wrapper(wrapper_class)
    model = build(linear_output=True)

    if isinstance(wrapper, TorchModelWrapper):
        torch.nn.init.zeros_(model.output_layer.weight)
        torch.nn.init.constant_(model.output_layer.bias, -10.0)
    else:
        kernel, bias = model.layers[-1].get_weights()
        model.layers[-1].set_weights([np.zeros_like(kernel), np.full_like(bias, -10)])

    sequential_labels = SEQUENTIAL_LABELS if sequential else ()
    wrapper.set_model(
        model, GLOBAL_LABELS, ["lut"], sequential_labels, report="resource_report"
    )

    loaded = _reload(wrapper, tmp_path)
    assert np.allclose(loaded.predict_dataset(make_dataset([1, 2, 3])), -10.0)


def _wrapper(wrapper_class):
    if wrapper_class == "keras":
        pytest.importorskip("tensorflow")
        from rule4ml.models.wrappers import KerasModelWrapper

        wrapper_class = KerasModelWrapper

    return wrapper_class()


def _reload(wrapper, tmp_path):
    wrapper.save(tmp_path / "bundle")
    config = next((tmp_path / "bundle").glob("*.config.json"))
    weights = next(p for p in (tmp_path / "bundle").iterdir() if p != config)

    loaded = type(wrapper)()
    loaded.load(str(config), str(weights))

    return loaded


def test_scaler_train_only(make_dataset):
    train, val = split(make_dataset([1, 2, 3, 4, 5, 6]), 0.5, seed=0)
    wrapper = TorchModelWrapper()
    scaler = Log1pScaler()
    wrapper.set_model(
        _torch_mlp(linear_output=True),
        GLOBAL_LABELS,
        ["lut"],
        report="resource_report",
        scaler=scaler,
    )

    wrapper.fit(train, TrainSettings(num_epochs=1, batch_size=4), val_dataset=val)

    lut = train.targets(["lut"], "resource_report")[:, 0]
    assert scaler.mu["lut"] == pytest.approx(np.log1p(lut).mean())


def test_predict_unmapped_board(make_dataset):
    wrapper = TorchModelWrapper()
    maps = {"strategy": default_strategy_map, "board": {"pynq-z2": 1}}
    model = TorchMLP(
        MLPSettings([2, 2], [4], [8], [0.0]), _shapes()[0], (None, 1), maps
    )
    wrapper.set_model(model, GLOBAL_LABELS, ["lut"], report="resource_report")

    with pytest.raises(ValueError, match=r"board values missing.*zcu102"):
        wrapper.predict_dataset(make_dataset([1, 2]))


def test_fit_zero_features(make_dataset):
    wrapper = TorchModelWrapper()
    settings = GNNSettings([4], [8], [2, 2], [2], [8], [False], [0.0])
    shapes = ((None, len(GLOBAL_LABELS) + 2), (None, 2), (None, 1))
    model = TorchGNN(settings, *shapes, GLOBAL_MAPS, SEQUENTIAL_MAPS, device="cpu")
    wrapper.set_model(
        model,
        GLOBAL_LABELS,
        ["lut"],
        ["layer_parameter_count"],
        report="resource_report",
    )

    with pytest.raises(ValueError, match="padding"):  # relu layers have no parameters
        wrapper.fit(make_dataset([1, 2]), TrainSettings(num_epochs=1))


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

    wrapper = TorchModelWrapper()
    wrapper.set_model(
        model,
        GLOBAL_LABELS,
        ["cycles", "lut"],
        report="resource_report",
        scaler=BoardScaler({"zcu102": {"cycles": 10, "lut": 100}}, maps["board"]),
    )

    inputs, targets = wrapper.training_data(dataset)

    np.testing.assert_allclose(targets, [[4.8, 1.0]])
    for key in model.global_input_keys["categorical"]:
        assert inputs[key].tolist() == [2]
