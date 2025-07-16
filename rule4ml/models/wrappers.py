import itertools
import json
import os
import time
from dataclasses import dataclass, field

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

try:
    import torch
except ImportError:
    torch = None

try:
    import onnx
except ImportError:
    onnx = None

from rule4ml.models.architectures import *  # noqa: F403
from rule4ml.models.metrics import rmse, smape
from rule4ml.models.utils import get_loss_from_str, get_optimizer_from_str
from rule4ml.parsers.data_parser import (
    boards_data,
    get_global_inputs,
    get_layers_data,
    to_dataframe,
)
from rule4ml.parsers.network_parser import (
    config_from_keras_model,
    config_from_onnx_model,
    config_from_torch_model,
)
from rule4ml.parsers.utils import camel_keys_to_snake, fixed_precision_to_bit_width


@dataclass
class TrainSettings:
    """
    _summary_

    Args: (all optional, default values inside data class)
        num_epochs (int): _description_
        batch_size (int): _description_
        learning_rate (float): _description_

        optimizer (str): _description_
        loss_function (str): _description_
        metrics (list): _description_
    """

    # Default settings

    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4

    optimizer: str = "adam"
    loss_function: str = "mae"
    metrics: list = field(default_factory=lambda: ["mape"])


class MultiModelWrapper:
    """
    _summary_
    """

    def __init__(self):
        self._models = {}

        self._default_boards = ["pynq-z2", "zcu102", "alveo-u200"]
        self._default_strategies = ["Latency", "Resource"]
        self._default_precisions = ["ap_fixed<2, 1>", "ap_fixed<8, 3>", "ap_fixed<16, 6>"]
        self._reuse_factors = [1, 2, 4, 8, 16, 32, 64]

    def add_model_wrapper(self, model_wrapper):
        key = "-".join([x.upper() for x in model_wrapper.output_labels])
        self._models[key] = model_wrapper

    def load_default_models(self):
        base_path = os.path.join(os.path.dirname(__file__), "weights")
        default_paths = [
            os.path.join(base_path, "BRAM_MLP"),
            os.path.join(base_path, "DSP_MLP"),
            os.path.join(base_path, "FF_MLP"),
            os.path.join(base_path, "LUT_MLP"),
            os.path.join(base_path, "CYCLES_MLP"),
        ]
        for path in default_paths:
            model_wrapper = KerasModelWrapper()
            model_wrapper.load(f"{path}_config.json", f"{path}.weights.h5")
            self.add_model_wrapper(model_wrapper)

    def predict(self, models_to_predict, hls_configs=None, verbose=0):
        """
        _summary_

        Args:
            models_to_predict (_type_): _description_
            hls_configs (_type_, optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
            Exception: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
        """

        if not isinstance(models_to_predict, (tuple, list)):
            models_to_predict = [models_to_predict]

        if hls_configs is None:
            hls_configs = [
                {
                    "model": {
                        "strategy": strategy,
                        "precision": precision,
                        "reuse_factor": reuse_factor,
                    },
                    "board": board,
                }
                for board, strategy, precision, reuse_factor in itertools.product(
                    self._default_boards,
                    self._default_strategies,
                    self._default_precisions,
                    self._reuse_factors,
                )
            ]
        else:
            if not isinstance(hls_configs, (tuple, list)):
                if not isinstance(hls_configs, dict):
                    raise TypeError(
                        "hls_configs argument expects a dictionary or list of dictionaries. See docstring example for correct formatting."
                    )
                hls_configs = [hls_configs]

        for idx in range(len(hls_configs)):
            hls_configs[idx] = camel_keys_to_snake(hls_configs[idx])

            if hls_configs[idx]["board"].lower() not in boards_data:
                raise ValueError(
                    f"Board {hls_configs[idx]['board']} is not currently supported. Supported boards: {boards_data.keys()}"
                )

        outputs = []
        for model_to_predict, hls_config in itertools.product(models_to_predict, hls_configs):
            model_name = getattr(model_to_predict, "name", model_to_predict.__class__.__name__)

            outputs.append(
                {
                    "Model": model_name,
                    "Board": hls_config["board"],
                    "Strategy": hls_config["model"]["strategy"],
                    "Precision": hls_config["model"]["precision"],
                    "Reuse Factor": hls_config["model"]["reuse_factor"],
                }
            )

        for key, estimator_model in self._models.items():
            estimator_predictions = estimator_model.predict(
                models_to_predict, hls_configs, verbose=verbose
            ).astype(float)

            target_labels = key.split("-")
            for idx, label in enumerate(target_labels):
                if "cycles" not in label.lower():
                    target_labels[idx] = f"{label} (%)"

            for idx, model_predictions in enumerate(estimator_predictions):
                outputs[idx].update(
                    {target_labels[k]: model_predictions[k] for k in range(len(model_predictions))}
                )

        outputs_df = pd.DataFrame(outputs).round(2)
        if not outputs_df.empty:
            sorted_boards = sorted(
                outputs_df["Board"].unique(), key=lambda x: list(boards_data.keys()).index(x)
            )
            outputs_df["Board"] = pd.Categorical(
                outputs_df["Board"], categories=sorted_boards, ordered=True
            )

            sorted_precisions = sorted(
                outputs_df["Precision"].unique(), key=lambda x: fixed_precision_to_bit_width(x)[0]
            )
            outputs_df["Precision"] = pd.Categorical(
                outputs_df["Precision"], categories=sorted_precisions, ordered=True
            )

        return outputs_df


class BaseModelWrapper:
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

        if hasattr(self, "model") and self.model is not None:
            del self.model

        self.model = None
        self.dataset = None

    def set_model(self, model):
        self.model = model

        self.global_input_keys = getattr(
            model, "global_input_keys", {"categorical": [], "numerical": ""}
        )
        self.sequential_input_keys = getattr(
            model, "sequential_input_keys", {"categorical": [], "numerical": ""}
        )

        self.global_categorical_maps = getattr(model, "global_categorical_maps", {})
        self.sequential_categorical_maps = getattr(model, "sequential_categorical_maps", {})

        self.output_shape = getattr(model, "output_shape", ())

    def build_inputs(self, inputs_df):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before building inputs.")

        input_dict = {}
        for key, label in zip(
            self.global_input_keys["categorical"], self.global_categorical_maps.keys()
        ):
            input_dict[key] = inputs_df[label].values

        global_numerical_inputs_df = inputs_df.drop(
            self.global_categorical_maps.keys(), axis=1
        ).select_dtypes(
            exclude=[object]
        )  # excluding nested "sequential_inputs" Series
        self.global_numerical_labels = global_numerical_inputs_df.columns.values

        input_dict[self.global_input_keys["numerical"]] = global_numerical_inputs_df.values

        if "sequential_inputs" in inputs_df:
            seq_inputs_series = inputs_df["sequential_inputs"]
            if not seq_inputs_series.empty:
                for key, label in zip(
                    self.sequential_input_keys["categorical"],
                    self.sequential_categorical_maps.keys(),
                ):
                    input_dict[key] = seq_inputs_series.apply(lambda df: df[label]).values

                seq_numerical_inputs = seq_inputs_series.apply(
                    lambda df: df.drop(self.sequential_categorical_maps.keys(), axis=1)
                    .select_dtypes(exclude=[object])
                    .values
                ).values
                seq_numerical_inputs = np.asarray([arr for arr in seq_numerical_inputs])

                self.sequential_numerical_labels = (
                    seq_inputs_series.iloc[0]
                    .drop(self.sequential_categorical_maps.keys(), axis=1)
                    .select_dtypes(exclude=[object])
                    .columns.values
                )

                input_dict[self.sequential_input_keys["numerical"]] = seq_numerical_inputs

        return input_dict

    def build_dataset(self, targets_df, batch_size):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before building a dataset.")

        self.batch_size = batch_size
        self.output_labels = targets_df.columns.values

    def fit(self):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before training.")
        if self.dataset is None:
            raise Exception("A dataset needs to be built before training a model.")
        return

    def predict(self, models_to_predict, hls_configs, verbose=0):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before making predictions.")

        if not isinstance(models_to_predict, (tuple, list)):
            raise TypeError("models_to_predict expects a list of keras or torch models.")
        if not isinstance(hls_configs, (tuple, list)):
            raise TypeError("hls_configs expects a list of HLS configuration dictionaries.")

        # Check if hls_configs boards and strategies are supported
        supported_boards = self.global_categorical_maps.get("board", [])
        supported_strategies = self.global_categorical_maps.get("strategy", [])
        for hls_config in hls_configs:
            config_board = hls_config.get("board", None)
            config_strategy = hls_config["model"].get("strategy", None)
            if config_board.lower() not in supported_boards:
                raise ValueError(
                    f"Model \"{self.model.name}\" does not support board \"{config_board}\". Supported boards: {list(supported_boards.keys())}"
                )
            if config_strategy.lower() not in supported_strategies:
                raise ValueError(
                    f"Model \"{self.model.name}\" does not support strategy \"{config_strategy}\". Supported strategies: {list(supported_strategies.keys())}"
                )

        global_inputs = []
        sequential_inputs = []

        for model_to_predict, hls_config in itertools.product(models_to_predict, hls_configs):
            if isinstance(model_to_predict, keras.Model):
                model_config = config_from_keras_model(model_to_predict, hls_config)
            elif torch is not None and isinstance(model_to_predict, torch.nn.Module):
                model_config = config_from_torch_model(model_to_predict, hls_config)
            elif onnx is not None and isinstance(model_to_predict, onnx.ModelProto):
                model_config = config_from_onnx_model(model_to_predict, hls_config)
            else:
                if torch is not None:
                    if onnx is not None:
                        raise TypeError(f"Unsupported model type: {type(model_to_predict)}")
                    else:
                        raise ImportError("Failed to import module onnx.")
                elif onnx is not None:
                    raise ImportError("Failed to import module torch.")

            global_inputs.append(get_global_inputs(model_config, hls_config))
            sequential_inputs.append(get_layers_data(model_config))

        inputs_df = to_dataframe(
            global_inputs=global_inputs,
            sequential_inputs=sequential_inputs,
            global_categorical_maps=self.global_categorical_maps,
            sequential_categorical_maps=self.sequential_categorical_maps,
            meta_data=[],
            targets=[],
        )

        return self.predict_from_df(inputs_df, verbose=verbose)

    def predict_from_df(self, inputs_df, targets=None, verbose=0):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before making predictions.")

        feature_labels = list(self.global_numerical_labels) + list(
            self.global_categorical_maps.keys()
        )
        if len(self.sequential_numerical_labels) > 0:
            sequential_labels = list(self.sequential_numerical_labels) + list(
                self.sequential_categorical_maps.keys()
            )
            inputs_df["sequential_inputs"] = inputs_df["sequential_inputs"].apply(
                lambda x: x[sequential_labels]
            )
            feature_labels += ["sequential_inputs"]

        inputs_df = inputs_df[feature_labels]
        inputs = self.build_inputs(inputs_df)

        start_time = time.time()
        prediction = self._model_predict(inputs, 0, verbose=verbose)
        total_time = time.time() - start_time

        if verbose > 1:
            r2 = r2_score(targets, prediction)
            print(f"R2 Score: {r2:.1f}")

            smape_value = smape(targets, prediction)
            print(f"SMAPE: {smape_value:.1f}%")

            rmse_value = rmse(targets, prediction)
            print(f"RMSE: {rmse_value:.1f}")

            avg_inference_time = total_time / len(inputs_df)
            print(f"Average Inference Time: {avg_inference_time:.2E} seconds")

            return prediction, r2, smape_value, rmse_value, avg_inference_time

        return prediction

    def to_config(self):
        return {
            "global_numerical_labels": list(self.global_numerical_labels),
            "sequential_numerical_labels": list(self.sequential_numerical_labels),
            "global_categorical_maps": self.global_categorical_maps,
            "sequential_categorical_maps": self.sequential_categorical_maps,
            "output_labels": list(self.output_labels),
            "output_shape": self.output_shape,
            "model_class": self.model.__class__.__name__,
            "model_config": self.model.to_config(),
        }

    def from_config(self, config):
        self.global_numerical_labels = config["global_numerical_labels"]
        self.sequential_numerical_labels = config["sequential_numerical_labels"]
        self.global_categorical_maps = config["global_categorical_maps"]
        self.sequential_categorical_maps = config["sequential_categorical_maps"]

        self.output_labels = config["output_labels"]
        self.output_shape = config["output_shape"]

        model_class = config["model_class"]
        self.model_config = config["model_config"]
        self.model = globals()[model_class].from_config(self.model_config)
        self.set_model(self.model)

    def save(self, save_dir):
        """
        Save the class config and model weights.

        Args:
            save_dir (_type_): _description_
        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        config = self.to_config()
        config_path = os.path.join(save_dir, f"{self.model.name}_config.json")
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=2)

        self._save_weights(save_dir)

    def load(self, config_path, weights_path):
        """
        Load saved class config.

        Args:
            config_path (_type_): _description_
        """

        if not config_path.endswith(".json"):
            raise ValueError("Config file name must end with .json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

        self.reset()

        with open(config_path) as json_file:
            # Load the class config
            config = json.load(json_file)
            self.from_config(config)

        self._load_weights(weights_path)

    def _model_predict(self, inputs, batch_size, verbose=0):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _save_weights(self, save_dir):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _load_weights(self, weights_path):
        raise NotImplementedError("This method should be implemented in subclasses.")


class TorchModelWrapper(BaseModelWrapper):
    def __init__(self, device=None):
        super().__init__()

        if torch is None:
            raise ImportError("Torch import failed. Please install torch to use this wrapper.")

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

    def _process_input_dict(self, input_dict):
        input_tensors = []
        for key, value in input_dict.items():
            if (
                key in self.global_input_keys["categorical"]
                or key in self.sequential_input_keys["categorical"]
            ):
                # Convert categorical inputs to long tensors
                input_tensors.append(torch.tensor(value, dtype=torch.long))
            else:
                # Convert numerical inputs to float tensors
                input_tensors.append(torch.tensor(value, dtype=torch.float32))

        return input_tensors

    def build_dataset(
        self,
        inputs_df,
        targets_df,
        batch_size,
        val_ratio=0.2,
        val_inputs_df=None,
        val_targets_df=None,
        train_repeats=1,
        shuffle=True,
        verbose=0,
    ):
        super().build_dataset(targets_df, batch_size)

        input_dict = self.build_inputs(inputs_df)
        if verbose > 0:
            for key, value in input_dict.items():
                print(f"{key}: {value.shape}")

        input_tensors = self._process_input_dict(input_dict)
        target_tensors = torch.tensor(targets_df.values)

        self.dataset = torch.utils.data.TensorDataset(*input_tensors, target_tensors)
        if val_inputs_df is not None and val_targets_df is not None:
            train_subset = torch.utils.data.TensorDataset(*input_tensors, target_tensors)
            
            val_input_dict = self.build_inputs(val_inputs_df)
            if verbose > 0:
                for key, value in val_input_dict.items():
                    print(f"val_{key}: {value.shape}")

            val_target_tensors = torch.tensor(val_targets_df.values)
            val_input_tensors = []
            for key, value in val_input_dict.items():
                if (
                    key in self.global_input_keys["categorical"]
                    or key in self.sequential_input_keys["categorical"]
                ):
                    val_input_tensors.append(torch.tensor(value, dtype=torch.long))
                else:
                    val_input_tensors.append(torch.tensor(value, dtype=torch.float32))
            val_subset = torch.utils.data.TensorDataset(*val_input_tensors, val_target_tensors)
        else:
            total_size = len(inputs_df)
            val_size = int(total_size * val_ratio)
            val_indices = list(range(val_size))
            train_indices = list(range(val_size, total_size)) * train_repeats
            train_subset = torch.utils.data.Subset(self.dataset, train_indices)
            val_subset = torch.utils.data.Subset(self.dataset, val_indices)

        def collate_fn(batch):
            batch = list(zip(*batch))
            inputs = {key: torch.stack(batch[i]) for i, key in enumerate(input_dict.keys())}
            targets = torch.stack(batch[-1])
            return inputs, targets

        self.train_data = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        self.val_data = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def fit(self, train_settings: TrainSettings, verbose=0):
        super().fit()

        optimizer = train_settings.optimizer
        if isinstance(optimizer, str):
            optimizer = get_optimizer_from_str(
                optimizer,
                backend="torch",
                params=self.model.parameters(),
                lr=train_settings.learning_rate,
            )

        loss_function = train_settings.loss_function
        if isinstance(loss_function, str):
            loss_function = get_loss_from_str(loss_function, backend="torch")

        metric_functions = train_settings.metrics

        history = {
            "train": {
                "loss": [],
                **{m.name: [] for m in metric_functions},
            },
        }
        if self.val_data:
            history["val"] = {
                "loss": [],
                **{m.name: [] for m in metric_functions},
            }

        for epoch in range(train_settings.num_epochs):
            self.model.train()
            train_loss = 0.0

            with tqdm(
                self.train_data,
                desc=f"Epoch {epoch + 1}/{train_settings.num_epochs}",
                leave=False,
            ) as pbar:
                for inputs, targets in pbar:
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    for m in metric_functions:
                        m.update(outputs, targets)

                    postfix = {"loss": f"{(train_loss / (pbar.n + 1)):.4f}"}
                    for m in metric_functions:
                        current_metric = m.result().item()
                        postfix[m.name] = f"{current_metric:.4f}"

                    pbar.set_postfix(postfix)

            avg_train_loss = train_loss / len(self.train_data)
            history["train"]["loss"].append(avg_train_loss)
            for m in metric_functions:
                history["train"][m.name].append(m.result().item())
                m.reset()  # Reset metrics for validation or next epoch

            summary = f"Epoch {epoch + 1}/{train_settings.num_epochs}\n"
            summary += f"loss: {avg_train_loss:.4f} -"
            for m in metric_functions:
                summary += f" {m.name}: {history['train'][m.name][-1]:.4f} -"

            if self.val_data:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for inputs, targets in self.val_data:
                        inputs = {key: value.to(self.device) for key, value in inputs.items()}
                        targets = targets.to(self.device)

                        outputs = self.model(inputs)

                        loss = loss_function(outputs, targets)
                        val_loss += loss.item()
                        for m in metric_functions:
                            m.update(outputs, targets)

                avg_val_loss = val_loss / len(self.val_data)
                history["val"]["loss"].append(avg_val_loss)
                for m in metric_functions:
                    history["val"][m.name].append(m.result().item())
                    m.reset()

                summary += f" val_loss: {avg_val_loss:.4f} -"
                for m in metric_functions:
                    summary += f" val_{m.name}: {history['val'][m.name][-1]:.4f} -"

            if verbose > 0:
                print(summary)

        return history

    def _model_predict(self, inputs, batch_size, verbose=0):
        self.model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict):
                input_tensors = self._process_input_dict(inputs)
                inputs = {key: value.to(self.device) for key, value in zip(inputs.keys(), input_tensors)}

            outputs = self.model(inputs)
            return outputs.detach().cpu().numpy()

    def _save_weights(self, save_dir):
        weights_path = os.path.join(save_dir, f"{self.model.name}_weights.pt")
        torch.save(self.model.state_dict(), weights_path)

    def _load_weights(self, weights_path):
        if not weights_path.endswith(".pt"):
            raise ValueError("Weights filename must end with .pt")

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))


class KerasModelWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__()

    def build_dataset(
        self,
        inputs_df,
        targets_df,
        batch_size,
        val_ratio=0.2,
        val_inputs_df=None,
        val_targets_df=None,
        train_repeats=1,
        shuffle=True,
        verbose=0,
    ):
        super().build_dataset(targets_df, batch_size)

        input_dict = self.build_inputs(inputs_df)
        if verbose > 0:
            for key, value in input_dict.items():
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")

        self.dataset = tf.data.Dataset.from_tensor_slices((input_dict, np.array(targets_df.values)))
        if val_inputs_df is not None and val_targets_df is not None:
            val_input_dict = self.build_inputs(val_inputs_df)

            train_data = self.dataset.repeat(train_repeats)
            if shuffle:
                train_data = train_data.shuffle(len(train_data))
            train_data = train_data.batch(batch_size)

            val_data = tf.data.Dataset.from_tensor_slices((val_input_dict, np.array(val_targets_df.values)))
            val_data = val_data.batch(batch_size)

            self.train_data = train_data
            self.val_data = val_data
        elif val_ratio > 0.0:
            val_len = int(len(self.dataset) * val_ratio)
            val_data = self.dataset.take(val_len)
            val_data = val_data.batch(batch_size)
            train_data = self.dataset.skip(val_len).repeat(train_repeats)
            if shuffle:
                train_data = train_data.shuffle(len(self.dataset) - val_len)
            train_data = train_data.batch(batch_size)

            self.train_data = train_data
            self.val_data = val_data
        else:
            train_data = self.dataset.repeat(train_repeats)
            if shuffle:
                train_data = train_data.shuffle(len(train_data))
            train_data = train_data.batch(batch_size)

            self.train_data = train_data
            self.val_data = None

    def fit(self, train_settings: TrainSettings, callbacks=[], verbose=0):
        super().fit()

        optimizer = train_settings.optimizer
        if isinstance(optimizer, str):
            optimizer = get_optimizer_from_str(
                optimizer, backend="keras", learning_rate=train_settings.learning_rate
            )

        loss_function = train_settings.loss_function
        if isinstance(loss_function, str):
            loss_function = get_loss_from_str(loss_function, backend="keras")

        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=train_settings.metrics,
        )

        return self.model.fit(
            self.train_data,
            epochs=train_settings.num_epochs,
            batch_size=train_settings.batch_size,
            validation_data=self.val_data,
            callbacks=callbacks,
            verbose=verbose,
        )

    def _model_predict(self, inputs, batch_size, verbose=0):
        return self.model.predict(inputs, batch_size, verbose=verbose)

    def _save_weights(self, save_dir):
        weights_path = os.path.join(save_dir, f"{self.model.name}.weights.h5")
        self.model.save_weights(weights_path)

    def _load_weights(self, weights_path):
        if not weights_path.endswith(".weights.h5"):
            raise ValueError("Weights filename must end with .weights.h5")

        self.model.load_weights(weights_path)
