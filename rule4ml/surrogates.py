import itertools
import json
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    import tensorflow as tf
except ModuleNotFoundError as error:
    if error.name != "tensorflow":
        raise
    tf = None

from rule4ml.dataset import UNSEEN_AS_ZERO
from rule4ml.features import DEFAULT_CLOCK_PERIOD, DEFAULT_IO_TYPE, model_features
from rule4ml.models import architectures, metrics, scaling
from rule4ml.models.architectures import (
    TorchGNN,
    TorchMLP,
    TorchTransformer,
)
from rule4ml.models.callbacks import EarlyStopping
from rule4ml.models.losses import get_loss_from_str
from rule4ml.models.utils import get_optimizer_from_str
from rule4ml.parsers.data_parser import (
    boards_data,
)
from rule4ml.parsers.utils import (
    camel_keys_to_snake,
    fixed_precision_to_bit_width,
)


@dataclass
class TrainSettings:
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4

    optimizer: str = "adam"
    loss_function: str = "mae"
    metrics: list = field(default_factory=list)


class BaseSurrogate:
    def __init__(self):
        self.reset()

    def reset(self):
        self.global_input_keys = {"categorical": [], "numerical": ""}
        self.sequential_input_keys = {"categorical": [], "numerical": ""}

        self.global_numerical_labels = []
        self.sequential_numerical_labels = []

        self.global_categorical_maps = {}
        self.sequential_categorical_maps = {}

        self.output_labels = []
        self.output_shape = ()
        self.report = None

        if hasattr(self, "model") and self.model is not None:
            del self.model

        self.model = None
        self.scaler = None
        self.input_scaler = None
        self.clock_periods = None
        self.io_types = None

    def set_model(
        self,
        model,
        global_labels,
        target_labels,
        sequential_labels=(),
        report=None,
        scaler=None,
        input_scaler=None,
    ):
        """Attach a model with its feature labels, targets and scalers.

        Args:
            model: Torch or Keras model, whose maps give the categorical inputs.
            global_labels (list): Numerical global feature labels.
            target_labels (list): Target labels, one per model output.
            sequential_labels (list, optional): Numerical per-layer feature labels.
            report (str, optional): Report of the resource targets, "resource_report"
                or "hls_resource_report".
            scaler (optional): Target scaler.
            input_scaler (optional): Scaler of the numerical inputs.
        """

        self.model = model
        self.global_input_keys = getattr(
            model, "global_input_keys", {"categorical": [], "numerical": ""}
        )
        self.sequential_input_keys = getattr(
            model, "sequential_input_keys", {"categorical": [], "numerical": ""}
        )
        self.global_categorical_maps = getattr(model, "global_categorical_maps", {})
        self.sequential_categorical_maps = getattr(
            model, "sequential_categorical_maps", {}
        )
        self.output_shape = getattr(model, "output_shape", ())

        self.global_numerical_labels = list(global_labels)
        self.sequential_numerical_labels = list(sequential_labels)
        self.output_labels = list(target_labels)
        self.report = report
        self.scaler = scaler
        self.input_scaler = input_scaler
        self.clock_periods = None  # training clocks, set by fit
        self.io_types = None  # {"conv" or "dense": IO types}

        for attribute, labels, maps in [
            ("global_input_shape", global_labels, self.global_categorical_maps),
            (
                "sequential_input_shape",
                sequential_labels,
                self.sequential_categorical_maps,
            ),
        ]:
            shape = getattr(model, attribute, None)
            width = len(labels) + len(maps)
            if (shape[-1] if shape else 0) != width:
                raise ValueError(
                    f"{attribute} {shape} does not fit {len(labels)} numerical "
                    f"labels plus {len(maps)} categorical inputs"
                )

        if self.output_shape[-1] != len(target_labels):
            raise ValueError(
                f"Output shape {self.output_shape} does not fit {len(target_labels)} targets"
            )

    def fit(
        self, dataset, train_settings, val_dataset=None, seed=0, callbacks=(), verbose=0
    ):
        """Train on a prepared Dataset, skipping records without every target.

        Args:
            dataset (Dataset): Training records.
            train_settings (TrainSettings): Epochs, batch size, optimizer and loss.
            val_dataset (Dataset, optional): Validation records.
            seed (int): Seed of the batch order.
            callbacks (tuple): Callbacks run after each epoch.
            verbose (int): Verbosity level.

        Returns:
            The training history, a dict for Torch and a History for Keras.
        """

        if self.model is None:
            raise RuntimeError("A model needs to be set or loaded before training.")

        if "clock_period" in self.global_numerical_labels:
            labelled = np.isfinite(dataset.targets(self.output_labels, self.report))
            clocks = dataset.columns(["clock_period"])[labelled.all(axis=1), 0]
            self.clock_periods = sorted({round(clock, 6) for clock in clocks.tolist()})

        train = self.training_data(dataset, fit_scaler=True)
        val = None if val_dataset is None else self.training_data(val_dataset)

        return self._fit(train, val, train_settings, seed, callbacks, verbose)

    def training_data(self, dataset, fit_scaler=False):
        """Inputs, scaled targets and scaler rows of records with every target."""

        targets = dataset.targets(self.output_labels, self.report)
        dataset = dataset.subset(np.flatnonzero(np.isfinite(targets).all(axis=1)))
        if len(dataset) == 0:
            raise ValueError(f"No records have every target of {self.output_labels}")

        if fit_scaler and self.input_scaler and self.sequential_numerical_labels:
            self.input_scaler.fit(
                dataset.columns(self.sequential_numerical_labels, per_layer=True)
            )

        targets = dataset.targets(self.output_labels, self.report)
        inputs, numerical, codes = self._dataset_inputs(dataset)

        input_rows = None
        if self.scaler:
            input_rows = self._scaler_rows(numerical, codes)
            target_rows = [
                dict(zip(self.output_labels, row)) for row in targets.tolist()
            ]

            if fit_scaler:
                self.scaler.fit(target_rows)

            scaled_rows = map(self.scaler.transform, target_rows, input_rows)
            targets = np.array(
                [[row[label] for label in self.output_labels] for row in scaled_rows]
            )

        return inputs, targets.astype(np.float32), input_rows

    def predict_dataset(self, dataset, mc_dropout=False, verbose=0):
        """Predict every record of a Dataset, in order.

        Args:
            dataset (Dataset): Records to predict.
            mc_dropout (bool): Keep dropout active.
            verbose (int): Verbosity level.

        Returns:
            np.ndarray: Raw-unit predictions, one row per record.
        """

        inputs, numerical, codes = self._dataset_inputs(dataset)
        return self._predict(
            inputs, numerical, codes, mc_dropout=mc_dropout, verbose=verbose
        )

    def evaluate(self, dataset, report=None, verbose=0):
        """SMAPE, RMSE and R2 per output label, over the records where it is known."""

        if (
            "bram" in self.output_labels
            and self.report
            and report not in (None, self.report)
        ):
            raise ValueError(f"BRAM units differ between {self.report} and {report}")

        targets = dataset.targets(self.output_labels, report or self.report)
        predictions = self.predict_dataset(dataset, verbose=verbose)
        return metrics.evaluate(targets, predictions, self.output_labels)

    def predict(
        self,
        models_to_predict,
        hls_configs,
        hls4ml_versions=(),
        vivado_versions=(),
        mc_dropout=False,
        verbose=0,
    ):
        """Predict every combination of models and settings.

        Args:
            models_to_predict (list): Keras or Torch models.
            hls_configs (list): HLS config dicts.
            hls4ml_versions (list, optional): hls4ml versions.
            vivado_versions (list, optional): Vivado versions.
            mc_dropout (bool): Keep dropout active.
            verbose (int): Verbosity level.

        Returns:
            np.ndarray: Raw-unit predictions, one row per combination.
        """

        features = self._features(
            models_to_predict, hls_configs, hls4ml_versions, vivado_versions
        )
        return self.predict_features(features, mc_dropout=mc_dropout, verbose=verbose)

    def mc_predict(
        self,
        models_to_predict,
        hls_configs,
        hls4ml_versions=(),
        vivado_versions=(),
        num_samples=50,
        return_all=False,
        verbose=0,
    ):
        """Predict with MC dropout, as the mean and std over samples.

        Args:
            models_to_predict (list): Keras or Torch models.
            hls_configs (list): HLS config dicts.
            hls4ml_versions (list, optional): hls4ml versions.
            vivado_versions (list, optional): Vivado versions.
            num_samples (int): Number of dropout samples.
            return_all (bool): Also return the samples.
            verbose (int): Verbosity level.

        Returns:
            tuple: Mean and std arrays, plus the samples when return_all is set.
        """

        features = self._features(
            models_to_predict, hls_configs, hls4ml_versions, vivado_versions
        )
        inputs = self._inputs_from_features(features)

        samples = np.array(
            [
                self._predict(*inputs, mc_dropout=True, verbose=verbose)
                for _ in range(num_samples)
            ]
        )

        outputs = (samples.mean(axis=0), samples.std(axis=0))
        return (*outputs, samples) if return_all else outputs

    def _features(
        self, models_to_predict, hls_configs, hls4ml_versions, vivado_versions
    ):
        if self.model is None:
            raise RuntimeError(
                "A model needs to be set or loaded before making predictions."
            )

        for name, value in [
            ("models_to_predict", models_to_predict),
            ("hls_configs", hls_configs),
            ("hls4ml_versions", hls4ml_versions),
            ("vivado_versions", vivado_versions),
        ]:
            if not isinstance(value, (tuple, list)):
                raise TypeError(f"{name} expects a list or tuple")

        return [
            model_features(*combo)
            for combo in itertools.product(
                models_to_predict,
                *self.accepted_settings(hls_configs, hls4ml_versions, vivado_versions),
            )
        ]

    def accepted_settings(self, hls_configs, hls4ml_versions, vivado_versions):
        """Validate settings against the training categories, None when empty."""

        supported = {
            key: self.categorical_values(key) for key in self.global_categorical_maps
        }

        for hls_config in hls_configs:
            board = hls_config.get("board", "").lower()
            strategy = hls_config.get("model", {}).get("strategy", "").lower()
            if board not in supported.get("board", []):
                raise ValueError(
                    f"Model {self.model.name} does not support board: {board}"
                )
            if strategy not in supported.get("strategy", []):
                raise ValueError(
                    f"Model {self.model.name} does not support strategy: {strategy}"
                )

        accepted_versions = []
        for key, versions in [
            ("hls4ml_version", hls4ml_versions),
            ("vivado_version", vivado_versions),
        ]:
            versions = [version for version in versions if version]
            if not versions and key in supported:
                raise ValueError(
                    f"Model {self.model.name} needs {key}s, one of: {supported[key]}"
                )

            for version in versions:
                if key not in supported:
                    raise ValueError(
                        f"Model {self.model.name} does not support {key}: {version}"
                    )

            accepted_versions.append(versions or [None])

        return [list(hls_configs) or [None], *accepted_versions]

    def categorical_values(self, label):
        """Lowercased training values of a global category, in map order."""

        return [value.lower() for value in self.global_categorical_maps.get(label, {})]

    def predict_features(self, features, mc_dropout=False, verbose=0):
        """Predict from (global_inputs, layers) pairs of model_features."""

        if self.model is None:
            raise RuntimeError(
                "A model needs to be set or loaded before making predictions."
            )

        inputs, numerical, codes = self._inputs_from_features(features)
        return self._predict(
            inputs, numerical, codes, mc_dropout=mc_dropout, verbose=verbose
        )

    def _predict(self, inputs, numerical, codes, mc_dropout=False, verbose=0):
        prediction = self._model_predict(
            inputs, 0, mc_dropout=mc_dropout, verbose=verbose
        )

        if self.scaler:
            prediction = np.asarray(
                [
                    list(
                        self.scaler.inverse(
                            dict(zip(self.output_labels, pred_row)), input_row
                        ).values()
                    )
                    for pred_row, input_row in zip(
                        prediction, self._scaler_rows(numerical, codes)
                    )
                ]
            )

        return prediction

    def _scaler_rows(self, numerical, codes):
        return [
            {
                **dict(zip(self.global_numerical_labels, numerical[idx].tolist())),
                **{
                    label: label_codes[idx].item()
                    for label, label_codes in codes.items()
                },
            }
            for idx in range(len(numerical))
        ]

    def _inputs_from_features(self, features):
        global_rows = [global_inputs for global_inputs, _ in features]
        codes = {
            label: np.array(
                [
                    mapping.get(row[label], 0)
                    if label in UNSEEN_AS_ZERO
                    else mapping[row[label]]
                    for row in global_rows
                ]
            )
            for label, mapping in self.global_categorical_maps.items()
        }

        for label in UNSEEN_AS_ZERO:
            if label in codes and (codes[label] == 0).any():
                unseen = sorted(
                    {
                        str(row[label])
                        for row, code in zip(global_rows, codes[label])
                        if not code
                    }
                )
                warnings.warn(
                    f"Model {self.model.name} was not trained on {label} {unseen}, "
                    "so the prediction can be less reliable"
                )

        numerical = np.array(
            [
                [row[label] for label in self.global_numerical_labels]
                for row in global_rows
            ]
        )
        self._warn_unseen_clocks(numerical)
        self._warn_unseen_io_types(global_rows)

        inputs = dict(zip(self.global_input_keys["categorical"], codes.values()))
        inputs[self.global_input_keys["numerical"]] = numerical

        if len(self.sequential_numerical_labels) > 0:
            layer_rows = [layers for _, layers in features]
            depth = max(len(layers) for layers in layer_rows)
            for key, (label, mapping) in zip(
                self.sequential_input_keys["categorical"],
                self.sequential_categorical_maps.items(),
            ):
                inputs[key] = np.array(
                    [
                        [mapping[layer[label]] for layer in layers]
                        + [0] * (depth - len(layers))
                        for layers in layer_rows
                    ]
                )

            padding = [0] * len(self.sequential_numerical_labels)
            inputs[self.sequential_input_keys["numerical"]] = np.array(
                [
                    [
                        [layer[label] for label in self.sequential_numerical_labels]
                        for layer in layers
                    ]
                    + [padding] * (depth - len(layers))
                    for layers in layer_rows
                ]
            )

        return self._scale_inputs(inputs), numerical, codes

    def _dataset_inputs(self, dataset):
        """Get the model inputs of every Dataset record.

        Returns:
            tuple: Model inputs, numerical values and category codes.

        Raises:
            ValueError: If a feature is not finite or a layer reads as padding.
        """

        codes = {
            label: dataset.encode(label, mapping)
            for label, mapping in self.global_categorical_maps.items()
        }

        numerical = dataset.columns(self.global_numerical_labels)
        _check_finite(numerical, self.global_numerical_labels)
        self._warn_unseen_clocks(numerical)

        inputs = dict(zip(self.global_input_keys["categorical"], codes.values()))
        inputs[self.global_input_keys["numerical"]] = numerical

        if self.sequential_numerical_labels:
            for key, (label, mapping) in zip(
                self.sequential_input_keys["categorical"],
                self.sequential_categorical_maps.items(),
            ):
                inputs[key] = dataset.pad_layers(
                    dataset.encode(label, mapping, per_layer=True)
                )

            layer_values = dataset.columns(
                self.sequential_numerical_labels, per_layer=True
            )
            _check_finite(layer_values, self.sequential_numerical_labels)

            if not layer_values.any(axis=1).all():
                raise ValueError(
                    "Some layers have all-zero sequential features and would read "
                    "as padding"
                )

            inputs[self.sequential_input_keys["numerical"]] = dataset.pad_layers(
                layer_values
            )

        return self._scale_inputs(inputs), numerical, codes

    def _warn_unseen_clocks(self, numerical):
        if not self.clock_periods or "clock_period" not in self.global_numerical_labels:
            return

        clocks = numerical[:, self.global_numerical_labels.index("clock_period")]
        unseen = sorted(
            {round(clock, 6) for clock in clocks.tolist()} - set(self.clock_periods)
        )
        if unseen:
            warnings.warn(
                f"Model {self.model.name} was not trained at clock periods {unseen} "
                "ns, so the prediction can be less reliable"
            )

    def _warn_unseen_io_types(self, global_rows):
        if not self.io_types:
            return

        unseen = set()
        for row in global_rows:
            convolutional = row.get("conv1d_count", 0) + row.get("conv2d_count", 0)
            kind = "conv" if convolutional else "dense"
            io_type = row.get("io_type", DEFAULT_IO_TYPE)
            if io_type not in self.io_types.get(kind, []):
                unseen.add((kind, io_type))

        for kind, io_type in sorted(unseen):
            warnings.warn(
                f"Model {self.model.name} was trained on {kind} models with "
                f"{self.io_types.get(kind)}, not {io_type}, so the prediction can be "
                "less reliable"
            )

    def _scale_inputs(self, inputs):
        """Apply the input scaler to the numerical model inputs."""

        if self.input_scaler is None:
            return inputs

        for key in (
            self.global_input_keys["numerical"],
            self.sequential_input_keys["numerical"],
        ):
            if key in inputs:
                inputs[key] = self.input_scaler.transform(inputs[key])

        return inputs

    def to_config(self):
        return {
            "global_numerical_labels": list(self.global_numerical_labels),
            "sequential_numerical_labels": list(self.sequential_numerical_labels),
            "global_categorical_maps": self.global_categorical_maps,
            "sequential_categorical_maps": self.sequential_categorical_maps,
            "output_labels": list(self.output_labels),
            "output_shape": self.output_shape,
            "report": self.report,
            "clock_periods": self.clock_periods,
            "io_types": self.io_types,
            "model_class": self.model.__class__.__name__,
            "model_config": self.model.to_config(),
            "scaler_class": self.scaler.__class__.__name__ if self.scaler else None,
            "scaler_config": self.scaler.to_config() if self.scaler else None,
            "input_scaler_class": (
                self.input_scaler.__class__.__name__ if self.input_scaler else None
            ),
            "input_scaler_config": (
                self.input_scaler.to_config() if self.input_scaler else None
            ),
        }

    def from_config(self, config):
        model = getattr(architectures, config["model_class"]).from_config(
            config["model_config"]
        )

        scaler = None
        if config.get("scaler_class") and config.get("scaler_config"):
            scaler = getattr(scaling, config["scaler_class"]).from_config(
                config["scaler_config"]
            )

        input_scaler = None
        if config.get("input_scaler_class"):
            input_scaler = getattr(scaling, config["input_scaler_class"]).from_config(
                config["input_scaler_config"]
            )

        sequential_labels = config["sequential_numerical_labels"]
        if not getattr(model, "sequential_input_shape", None):
            sequential_labels = ()  # older MLP bundles list layer labels they never read

        self.set_model(
            model,
            config["global_numerical_labels"],
            config["output_labels"],
            sequential_labels,
            report=config.get("report"),
            scaler=scaler,
            input_scaler=input_scaler,
        )
        self.clock_periods = config.get("clock_periods")  # absent in older bundles
        self.io_types = config.get("io_types")

    def save(self, save_dir):
        """Save the class config and model weights."""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        config = self.to_config()
        config_path = os.path.join(save_dir, f"{self.model.name}.config.json")
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=2)

        self._save_weights(save_dir)

    def load(self, config_path, weights_path):
        """Load saved class config."""

        if not config_path.endswith(".json"):
            raise ValueError("Config file name must end with .json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

        self.reset()

        with open(config_path) as json_file:
            config = json.load(json_file)
            self.from_config(config)

        self._load_weights(weights_path)

    def _fit(self, train, val, train_settings, seed, callbacks, verbose):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _model_predict(self, inputs, batch_size, mc_dropout, verbose=0):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _save_weights(self, save_dir):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _load_weights(self, weights_path):
        raise NotImplementedError("This method should be implemented in subclasses.")


class TorchSurrogate(BaseSurrogate):
    def __init__(self):
        super().__init__()

    def set_model(self, model, *args, **kwargs):
        super().set_model(model, *args, **kwargs)
        self.device = model.device

    def _tensors(self, inputs):
        categorical = {
            *self.global_input_keys["categorical"],
            *self.sequential_input_keys["categorical"],
        }

        return {
            key: torch.tensor(
                value, dtype=torch.long if key in categorical else torch.float32
            )
            for key, value in inputs.items()
        }

    def _fit(self, train, val, train_settings, seed, callbacks, verbose):
        self.optimizer = train_settings.optimizer
        if isinstance(self.optimizer, str):
            self.optimizer = get_optimizer_from_str(
                self.optimizer,
                backend="torch",
                params=self.model.parameters(),
                lr=train_settings.learning_rate,
            )

        loss_function = train_settings.loss_function
        if isinstance(loss_function, str):
            loss_function = get_loss_from_str(loss_function, backend="torch")

        metric_functions = train_settings.metrics
        history = {"train": {"loss": [], **{m.name: [] for m in metric_functions}}}
        if val is not None:
            history["val"] = {"loss": [], **{m.name: [] for m in metric_functions}}
            val = (self._tensors(val[0]), torch.from_numpy(val[1]), val[2])

        inputs, targets = self._tensors(train[0]), torch.from_numpy(train[1])
        input_rows = train[2]
        batch_size = train_settings.batch_size
        num_batches = -(-len(targets) // batch_size)
        generator = torch.Generator().manual_seed(seed)

        for epoch in range(train_settings.num_epochs):
            self.model.train()
            order = torch.randperm(len(targets), generator=generator)
            train_loss, seen = 0.0, 0

            with tqdm(
                order.split(batch_size),
                desc=f"Epoch {epoch + 1}/{train_settings.num_epochs}",
                leave=False,
            ) as pbar:
                for rows in pbar:
                    batch = {
                        key: value[rows].to(self.device)
                        for key, value in inputs.items()
                    }
                    batch_targets = targets[rows].to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(batch)

                    loss = loss_function(outputs, batch_targets)
                    if hasattr(self.model, "kl_loss"):
                        loss += self.model.kl_loss / num_batches

                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item() * len(rows)
                    seen += len(rows)

                    batch_rows = _select(input_rows, rows)
                    for m in metric_functions:
                        m.update(
                            batch_targets.detach(),
                            outputs.detach(),
                            self.output_labels,
                            batch_rows,
                        )

                    postfix = {"loss": f"{train_loss / seen:.6f}"}
                    for m in metric_functions:
                        postfix[m.name] = f"{m.result().item():.6f}"
                    pbar.set_postfix(postfix)

            history["train"]["loss"].append(train_loss / seen)
            for m in metric_functions:
                history["train"][m.name].append(m.result().item())
                m.reset()

            summary = f"Epoch {epoch + 1}/{train_settings.num_epochs}\n"
            summary += f"loss: {history['train']['loss'][-1]:.6f} -"
            for m in metric_functions:
                summary += f" {m.name}: {history['train'][m.name][-1]:.6f} -"

            if val is not None:
                val_loss = self._validation_loss(
                    val, loss_function, metric_functions, batch_size
                )
                history["val"]["loss"].append(val_loss)
                for m in metric_functions:
                    history["val"][m.name].append(m.result().item())
                    m.reset()

                summary += f" val_loss: {val_loss:.6f} -"
                for m in metric_functions:
                    summary += f" val_{m.name}: {history['val'][m.name][-1]:.6f} -"

            if verbose > 0:
                print(summary)

            for callback in callbacks:
                if not hasattr(callback, "step"):
                    raise NotImplementedError(
                        f"Callback {callback} does not implement a step() method."
                    )

                monitor = callback.monitor
                monitor_parts = monitor.split("_")
                monitor_value = None
                if len(monitor_parts) > 1 and monitor_parts[0] in history:
                    monitor = "_".join(monitor_parts[1:])
                    monitor_value = history[monitor_parts[0]].get(monitor, None)

                if monitor_value is None:
                    raise ValueError(
                        f"Monitored value {callback.monitor} not found in fit history."
                    )

                cb_result = callback.step(monitor_value[-1], self, verbose=verbose)
                if isinstance(callback, EarlyStopping) and cb_result:
                    return history

        for callback in callbacks:  # epochs ran out before early stopping
            if isinstance(callback, EarlyStopping):
                callback.restore(self, verbose)

        return history

    def _validation_loss(self, val, loss_function, metric_functions, batch_size):
        """Record-weighted validation loss, updating the metrics."""

        inputs, targets, input_rows = val
        self.model.eval()
        total = 0.0

        with torch.no_grad():
            for rows in torch.arange(len(targets)).split(batch_size):
                batch = {
                    key: value[rows].to(self.device) for key, value in inputs.items()
                }
                batch_targets = targets[rows].to(self.device)

                outputs = self.model(batch)
                total += loss_function(outputs, batch_targets).item() * len(rows)

                batch_rows = _select(input_rows, rows)
                for m in metric_functions:
                    m.update(batch_targets, outputs, self.output_labels, batch_rows)

        return total / len(targets)

    def _enable_mc_dropout(self):
        """Set dropout layers to train mode."""

        if self.model is None:
            raise RuntimeError(
                "A model needs to be set or loaded before enabling MC Dropout."
            )

        def set_dropout_train(m):
            if isinstance(  # attention dropout and the encoder fast path skip Dropout
                m,
                (
                    torch.nn.Dropout,
                    torch.nn.MultiheadAttention,
                    torch.nn.TransformerEncoderLayer,
                ),
            ):
                m.train()

        self.model.apply(set_dropout_train)

    def _model_predict(self, inputs, batch_size, mc_dropout, verbose=0):
        self.model.eval()
        if mc_dropout:
            self._enable_mc_dropout()

        with torch.no_grad():
            if isinstance(inputs, dict):
                inputs = {
                    key: value.to(self.device)
                    for key, value in self._tensors(inputs).items()
                }

            outputs = self.model(inputs)
            return outputs.detach().cpu().numpy()

    def _save_weights(self, save_dir):
        weights_path = os.path.join(save_dir, f"{self.model.name}.weights.pt")
        torch.save(self.model.state_dict(), weights_path)

    def _load_weights(self, weights_path):
        if not weights_path.endswith(".pt"):
            raise ValueError("Weights filename must end with .pt")

        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )


class KerasSurrogate(BaseSurrogate):
    def __init__(self):
        if tf is None:
            raise ImportError('Keras support needs: pip install "rule4ml[keras]"')
        super().__init__()

    def _fit(self, train, val, train_settings, seed, callbacks, verbose):
        self._set_mc_dropout(False)  # left on by mc_predict, validation must not drop

        self.optimizer = train_settings.optimizer
        if isinstance(self.optimizer, str):
            self.optimizer = get_optimizer_from_str(
                self.optimizer,
                backend="keras",
                learning_rate=train_settings.learning_rate,
            )

        loss_function = train_settings.loss_function
        if isinstance(loss_function, str):
            loss_function = get_loss_from_str(loss_function, backend="keras")

        self.model.compile(
            optimizer=self.optimizer,
            loss=loss_function,
            metrics=train_settings.metrics,
        )

        batch_size = train_settings.batch_size
        train_data = (
            tf.data.Dataset.from_tensor_slices(train[:2])
            .shuffle(len(train[1]), seed=seed)
            .batch(batch_size)
        )

        val_data = (
            None
            if val is None
            else tf.data.Dataset.from_tensor_slices(val[:2]).batch(batch_size)
        )

        return self.model.fit(
            train_data,
            epochs=train_settings.num_epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=verbose,
        )

    def _model_predict(self, inputs, batch_size, mc_dropout, verbose=0):
        self._set_mc_dropout(mc_dropout)
        return self.model.predict(inputs, batch_size, verbose=verbose)

    def _set_mc_dropout(self, enabled):
        dropouts = [
            layer
            for layer in self.model._flatten_layers(include_self=False)
            if isinstance(layer, architectures.KerasMCDropout)
        ]

        if any(layer.enable_mc != enabled for layer in dropouts):
            for layer in dropouts:
                layer.enable_mc = enabled
            self.model.predict_function = None  # the flag is traced into the graph

    def _save_weights(self, save_dir):
        weights_path = os.path.join(save_dir, f"{self.model.name}.weights.h5")
        self.model.save_weights(weights_path)

    def _load_weights(self, weights_path):
        if not weights_path.endswith(".weights.h5"):
            raise ValueError("Weights filename must end with .weights.h5")

        self.model.load_weights(weights_path)


class MultiSurrogate:
    """One predictor per target group, with predictions merged per model and config."""

    _default_precisions = ("ap_fixed<2, 1>", "ap_fixed<8, 3>", "ap_fixed<16, 6>")
    _default_reuse_factors = (1, 2, 4, 8, 16, 32, 64)

    def __init__(self):
        self._surrogates: dict[str, BaseSurrogate] = {}

    def add_surrogate(self, surrogate: BaseSurrogate):
        key = "-".join([x.upper() for x in surrogate.output_labels])
        self._surrogates[key] = surrogate

    def load_default_surrogates(self):
        base_path = os.path.join(
            os.path.dirname(__file__), "models", "weights", "default"
        )
        default_paths = [
            os.path.join(base_path, name)
            for name in (
                "BRAM",
                "DSP",
                "FF",
                "LUT",
                "CYCLES_MIN",
                "CYCLES_MAX",
                "INTERVAL_MIN",
                "INTERVAL_MAX",
            )
        ]

        for path in default_paths:
            if not os.path.exists(path + ".config.json"):
                continue

            with open(f"{path}.config.json") as f:
                config = json.load(f)

            class_name = config.get("model_class", None)
            if class_name == "KerasMLP":
                if os.path.exists(f"{path}.weights.h5"):
                    surrogate = KerasSurrogate()
                    surrogate.load(f"{path}.config.json", f"{path}.weights.h5")
                    self.add_surrogate(surrogate)
                else:
                    raise FileNotFoundError(
                        f"Weights file not found: {path}.weights.h5"
                    )
            elif class_name in [
                TorchMLP.__name__,
                TorchGNN.__name__,
                TorchTransformer.__name__,
            ]:
                if os.path.exists(f"{path}.weights.pt"):
                    surrogate = TorchSurrogate()
                    surrogate.load(f"{path}.config.json", f"{path}.weights.pt")
                    self.add_surrogate(surrogate)
                else:
                    raise FileNotFoundError(
                        f"Weights file not found: {path}.weights.pt"
                    )
            else:
                raise NotImplementedError(f"Model class not supported: {class_name}")

    def predict(
        self,
        models_to_predict,
        *,
        hls_configs=None,
        hls4ml_versions=None,
        vivado_versions=None,
        round_digits=2,
        verbose=0,
    ):
        """Predict every model over a sweep of settings.

        Args:
            models_to_predict: Keras or Torch model, or a list of them.
            hls_configs (dict or list, optional): HLS configs, camelCase or snake_case.
                None sweeps the training boards, strategies and clocks, with default
                precisions and reuse factors.
            hls4ml_versions (str or list, optional): None sweeps the training versions.
            vivado_versions (str or list, optional): None sweeps the training versions.
            round_digits (int): Decimals of the returned values.
            verbose (int): Verbosity level.

        Returns:
            pd.DataFrame: One row per model and setting, with the raw predictions and
                their share of the board resources in percent.
        """

        if not self._surrogates:
            raise RuntimeError("No surrogates have been added to the MultiSurrogate.")

        if not isinstance(models_to_predict, (tuple, list)):
            models_to_predict = [models_to_predict]

        if hls_configs is not None:
            if isinstance(hls_configs, dict):
                hls_configs = [hls_configs]
            elif not isinstance(hls_configs, (tuple, list)):
                raise TypeError(
                    "hls_configs expects a dictionary or a list of dictionaries."
                )
            hls_configs = [camel_keys_to_snake(config) for config in hls_configs]

        columns = (
            "Model",
            "Board",
            "Strategy",
            "Precision",
            "Reuse Factor",
            "Clock Period",
            "HLS4ML Version",
            "Vivado Version",
        )
        output_rows, feature_cache = {}, {}
        for target_key, surrogate in self._surrogates.items():
            configs = hls_configs
            if configs is None:
                configs = [
                    {
                        "model": {
                            "strategy": strategy,
                            "precision": precision,
                            "reuse_factor": reuse_factor,
                        },
                        "board": board,
                    }
                    for board, strategy, precision, reuse_factor in itertools.product(
                        surrogate.categorical_values("board"),
                        surrogate.categorical_values("strategy"),
                        MultiSurrogate._default_precisions,
                        MultiSurrogate._default_reuse_factors,
                    )
                ]
                if "clock_period" in surrogate.global_numerical_labels:
                    configs = [
                        {**config, "clock_period": clock}
                        for config in configs
                        for clock in surrogate.clock_periods or [DEFAULT_CLOCK_PERIOD]
                    ]

            versions = []
            for label, values in [
                ("hls4ml_version", hls4ml_versions),
                ("vivado_version", vivado_versions),
            ]:
                if values is None:
                    values = surrogate.categorical_values(label)
                elif isinstance(values, str):
                    values = [values]
                versions.append(list(values))

            rows, features = [], []
            for combo in itertools.product(
                enumerate(models_to_predict),
                *surrogate.accepted_settings(configs, *versions),
            ):
                (index, model), config, hls4ml_version, vivado_version = combo
                metadata = (
                    getattr(model, "name", model.__class__.__name__),
                    config["board"],
                    config["model"]["strategy"],
                    config["model"]["precision"],
                    config["model"]["reuse_factor"],
                    config.get("clock_period", DEFAULT_CLOCK_PERIOD),
                    hls4ml_version,
                    vivado_version,
                )

                config_key = json.dumps(config, sort_keys=True, default=str)
                # names and shown settings can repeat across models and configs
                row_key = (index, config_key, hls4ml_version, vivado_version)

                if row_key not in output_rows:
                    output_rows[row_key] = dict(zip(columns, metadata))
                rows.append(output_rows[row_key])

                feature_key = (id(model), config_key, hls4ml_version, vivado_version)
                if feature_key not in feature_cache:
                    feature_cache[feature_key] = model_features(
                        model, config, hls4ml_version, vivado_version
                    )
                features.append(feature_cache[feature_key])

            predictions = surrogate.predict_features(features, verbose=verbose)
            targets = target_key.split("-")

            for row, prediction in zip(rows, predictions.astype(float), strict=True):
                board = boards_data[row["Board"]]
                for target, value in zip(targets, prediction, strict=True):
                    capacity = board.get(f"max_{target.lower()}")
                    if target == "BRAM":  # boards count 18K blocks, Vivado 36K tiles
                        target, capacity = {
                            "resource_report": ("BRAM_36K", capacity / 2),
                            "hls_resource_report": ("BRAM_18K", capacity),
                        }.get(surrogate.report, ("BRAM", None))

                    row[target] = value
                    if capacity:
                        row[f"{target} (%)"] = value / capacity * 100

        outputs_df = pd.DataFrame(list(output_rows.values())).round(round_digits)
        if not any(
            "clock_period" in surrogate.global_numerical_labels
            for surrogate in self._surrogates.values()
        ):
            outputs_df = outputs_df.drop(columns="Clock Period", errors="ignore")

        if not outputs_df.empty:
            for column, sort_key in [
                ("Board", list(boards_data).index),
                ("Precision", lambda value: fixed_precision_to_bit_width(value)[0]),
            ]:
                outputs_df[column] = pd.Categorical(
                    outputs_df[column],
                    categories=sorted(outputs_df[column].unique(), key=sort_key),
                    ordered=True,
                )

        return outputs_df


def _select(input_rows, rows):
    return None if input_rows is None else [input_rows[i] for i in rows.tolist()]


def _check_finite(values, labels):
    finite = np.isfinite(values).all(axis=0)

    if not finite.all():
        bad = [label for label, ok in zip(labels, finite) if not ok]
        raise ValueError(f"Features with non-finite values: {bad}")
