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

from rule4ml.models.architectures import (
    KerasMCDropout,
    KerasMLP,
    KerasTransformer,
    TorchGNN,
    TorchMLP,
)
from rule4ml.models.callbacks import EarlyStopping
from rule4ml.models.metrics import rmse, smape
from rule4ml.models.scaling import *  # noqa: F403
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
from rule4ml.parsers.utils import camel_keys_to_snake, fixed_precision_to_bit_width, to_lower_keys


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
        self.scaler = None

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

    def build_dataset(
        self,
        inputs_df,
        targets_df,
        batch_size,
        val_inputs_df=None,
        val_targets_df=None,
        scaler=None,
    ):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before building a dataset.")

        self.batch_size = batch_size
        self.output_labels = targets_df.columns.values

        if scaler:
            targets_df = pd.DataFrame(
                [
                    scaler.transform(target_row, input_row)
                    for target_row, input_row in zip(
                        targets_df.to_dict(orient="records"), inputs_df.to_dict(orient="records")
                    )
                ],
                index=targets_df.index,
            )
            if val_targets_df is not None:
                val_targets_df = pd.DataFrame(
                    [
                        scaler.transform(target_row, input_row)
                        for target_row, input_row in zip(
                            val_targets_df.to_dict(orient="records"),
                            val_inputs_df.to_dict(orient="records"),
                        )
                    ],
                    index=val_targets_df.index,
                )
        self.scaler = scaler
        return targets_df, val_targets_df

    def fit(self):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before training.")
        if self.dataset is None:
            raise Exception("A dataset needs to be built before training a model.")
        return

    def predict_from_df(self, inputs_df, targets=None, mc_dropout=False, verbose=0):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before making predictions.")

        feature_labels = list(self.global_numerical_labels) + list(
            self.global_categorical_maps.keys()
        )
        if len(self.sequential_numerical_labels) > 0:
            sequential_labels = list(self.sequential_numerical_labels) + list(
                self.sequential_categorical_maps.keys()
            )
            inputs_df.loc[:, "sequential_inputs"] = inputs_df["sequential_inputs"].apply(
                lambda x: x[sequential_labels]
            )
            feature_labels += ["sequential_inputs"]

        inputs_df = inputs_df[feature_labels]
        inputs = self.build_inputs(inputs_df)

        start_time = time.time()
        prediction = self._model_predict(inputs, 0, mc_dropout=mc_dropout, verbose=verbose)
        total_time = time.time() - start_time

        if self.scaler:
            prediction = np.asarray(
                [
                    list(
                        self.scaler.inverse(
                            dict(zip(self.output_labels, pred_row)),
                            dict(zip(inputs_df.columns, inputs_df.iloc[idx])),
                        ).values()
                    )
                    for idx, pred_row in enumerate(prediction)
                ]
            )

        if verbose > 1:
            r2 = r2_score(targets, prediction, force_finite=False)
            print(f"R2 Score: {r2:.2f}")

            smape_value = smape(targets, prediction)
            print(f"SMAPE: {smape_value:.2f}%")

            rmse_value = rmse(targets, prediction)
            print(f"RMSE: {rmse_value:.2f}")

            avg_inference_time = total_time / len(inputs_df)
            print(f"Average Inference Time: {avg_inference_time:.2E} seconds")

            return prediction, r2, smape_value, rmse_value, avg_inference_time

        return prediction

    def predict(
        self,
        models_to_predict,
        hls_configs,
        hls4ml_versions,
        vivado_versions,
        mc_dropout=False,
        verbose=0,
    ):
        if self.model is None:
            raise Exception("A model needs to be set or loaded before making predictions.")

        if not isinstance(models_to_predict, (tuple, list)):
            raise TypeError("models_to_predict expects a list of keras or torch models.")
        if not isinstance(hls_configs, (tuple, list)):
            raise TypeError("hls_configs expects a list of HLS configuration dictionaries.")
        if not isinstance(hls4ml_versions, (tuple, list)):
            raise TypeError("hls4ml_versions expects a list or tuple of version strings.")
        if not isinstance(vivado_versions, (tuple, list)):
            raise TypeError("vivado_versions expects a list or tuple of version strings.")

        supported_categoricals = {}
        for key, mapping in self.global_categorical_maps.items():
            supported_categoricals[key] = list(to_lower_keys(mapping).keys())

        accepted_hls_configs = []
        for hls_config in hls_configs:
            board = hls_config.get("board", "").lower()
            strategy = hls_config.get("model", {}).get("strategy", "").lower()
            if board not in supported_categoricals.get("board", []):
                raise Exception(f"Model {self.model.name} does not support board: {board}")
            if strategy not in supported_categoricals.get("strategy", []):
                raise Exception(f"Model {self.model.name} does not support strategy: {strategy}")
            accepted_hls_configs.append(hls_config)
        if not accepted_hls_configs:
            accepted_hls_configs = [None]

        accepted_hls4ml_versions = []
        for hls4ml_version in hls4ml_versions:
            if hls4ml_version:
                if hls4ml_version.lower() not in supported_categoricals.get("hls4ml_version", []):
                    raise Exception(
                        f"Model {self.model.name} does not support hls4ml_version: {hls4ml_version}"
                    )
                accepted_hls4ml_versions.append(hls4ml_version)
        if not accepted_hls4ml_versions:
            accepted_hls4ml_versions = [None]

        accepted_vivado_versions = []
        for vivado_version in vivado_versions:
            if vivado_version:
                if vivado_version.lower() not in supported_categoricals.get("vivado_version", []):
                    raise Exception(
                        f"Model {self.model.name} does not support vivado_version: {vivado_version}"
                    )
                accepted_vivado_versions.append(vivado_version)
        if not accepted_vivado_versions:
            accepted_vivado_versions = [None]

        global_inputs = []
        sequential_inputs = []

        for model_to_predict, hls_config, hls4ml_version, vivado_version in itertools.product(
            models_to_predict,
            accepted_hls_configs,
            accepted_hls4ml_versions,
            accepted_vivado_versions,
        ):
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

            global_inputs.append(
                get_global_inputs(
                    model_config,
                    hls_config,
                    hls4ml_version=hls4ml_version,
                    vivado_version=vivado_version,
                )
            )
            sequential_inputs.append(get_layers_data(model_config))

        inputs_df = to_dataframe(
            global_inputs=global_inputs,
            sequential_inputs=sequential_inputs,
            global_categorical_maps=self.global_categorical_maps,
            sequential_categorical_maps=self.sequential_categorical_maps,
            meta_data=[],
            targets=[],
        )

        return self.predict_from_df(inputs_df, mc_dropout=mc_dropout, verbose=verbose)

    def mc_predict_from_df(self, inputs_df, num_samples=50, return_all=False, verbose=0):
        """
        Perform MC Dropout predictions from a dataframe.

        Args:
            inputs_df (pd.DataFrame): input dataframe.
            num_samples (int, optional): number of samples for MC Dropout. Defaults to 100.
            return_all (bool, optional): whether to return all samples or just the mean and std. Defaults to False.
            verbose (int, optional): verbosity level. Defaults to 0.

        Returns:
            tuple: mean and std of predictions if return_all is False, else mean, std, and all samples.
        """
        if self.model is None:
            raise Exception("A model needs to be set or loaded before making predictions.")

        all_predictions = []
        for _ in range(num_samples):
            predictions = self.predict_from_df(inputs_df, mc_dropout=True, verbose=verbose)
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions)
        predictions_mean = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)

        outputs = (predictions_mean, predictions_std)
        if return_all:
            outputs += (all_predictions,)

        return outputs

    def mc_predict(
        self, models_to_predict, hls_configs, num_samples=50, return_all=False, verbose=0
    ):
        """
        Perform MC Dropout predictions.

        Args:
            models_to_predict (list): list of models to predict.
            hls_configs (list): list of HLS configuration dictionaries.
            num_samples (int, optional): number of samples for MC Dropout. Defaults to 100.
            return_all (bool, optional): whether to return all samples or just the mean and std. Defaults to False.
            verbose (int, optional): verbosity level. Defaults to 0.

        Returns:
            tuple: mean and std of predictions if return_all is False, else mean, std, and all samples.
        """
        if self.model is None:
            raise Exception("A model needs to be set or loaded before making predictions.")

        all_predictions = []
        for _ in range(num_samples):
            predictions = self.predict(
                models_to_predict, hls_configs, mc_dropout=True, verbose=verbose
            )
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions)
        predictions_mean = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)

        outputs = (predictions_mean, predictions_std)
        if return_all:
            outputs += (all_predictions,)

        return outputs

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
            "scaler_class": self.scaler.__class__.__name__ if self.scaler else None,
            "scaler_config": self.scaler.to_config() if self.scaler else None,
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

        scaler_class = config.get("scaler_class", None)
        scaler_config = config.get("scaler_config", None)
        if scaler_class and scaler_config:
            self.scaler = globals()[scaler_class].from_config(scaler_config)

    def save(self, save_dir):
        """
        Save the class config and model weights.

        Args:
            save_dir (_type_): _description_
        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        config = self.to_config()
        config_path = os.path.join(save_dir, f"{self.model.name}.config.json")
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

    def _enable_mc_dropout(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _model_predict(self, inputs, batch_size, mc_dropout, verbose=0):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _save_weights(self, save_dir):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _load_weights(self, weights_path):
        raise NotImplementedError("This method should be implemented in subclasses.")


class TorchModelWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__()
        if torch is None:
            raise ImportError("Torch import failed. Please install torch to use this wrapper.")

    def set_model(self, model):
        super().set_model(model)
        self.device = model.device

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
        scaler=None,
        verbose=0,
    ):
        targets_df, val_targets_df = super().build_dataset(
            targets_df, batch_size, val_targets_df=val_targets_df, scaler=scaler
        )

        input_dict = self.build_inputs(inputs_df)
        if verbose > 0:
            for key, value in input_dict.items():
                print(f"{key}: {value.shape}")

        input_tensors = self._process_input_dict(input_dict)
        target_tensors = torch.tensor(targets_df.values)

        self.dataset = torch.utils.data.TensorDataset(*input_tensors, target_tensors)
        if val_inputs_df is not None and val_targets_df is not None:
            indices = torch.randperm(len(self.dataset))
            train_indices = indices.tolist() * train_repeats
            train_subset = torch.utils.data.Subset(self.dataset, train_indices)

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

            indices = torch.randperm(total_size)
            val_indices = indices[:val_size].tolist()
            train_indices = indices[val_size:].tolist() * train_repeats
            train_subset = torch.utils.data.Subset(self.dataset, train_indices)
            val_subset = torch.utils.data.Subset(self.dataset, val_indices)

        def collate_fn(batch):
            batch = list(zip(*batch))
            inputs = {key: torch.stack(batch[i]) for i, key in enumerate(input_dict.keys())}
            targets = torch.stack(batch[-1]).to(dtype=torch.float32)
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

    def fit(self, train_settings: TrainSettings, callbacks=[], verbose=0):
        super().fit()

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

        num_batches = len(self.train_data)
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

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    loss = loss_function(outputs, targets)
                    if hasattr(self.model, "kl_loss"):
                        loss += 1 / num_batches * self.model.kl_loss

                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    for m in metric_functions:
                        m.update(targets.detach(), outputs.detach(), self.output_labels)

                    postfix = {"loss": f"{(train_loss / (pbar.n + 1)):.6f}"}
                    for m in metric_functions:
                        current_metric = m.result().item()
                        postfix[m.name] = f"{current_metric:.6f}"

                    pbar.set_postfix(postfix)

            avg_train_loss = train_loss / len(self.train_data)
            history["train"]["loss"].append(avg_train_loss)
            for m in metric_functions:
                history["train"][m.name].append(m.result().item())
                m.reset()  # Reset metrics for validation or next epoch

            summary = f"Epoch {epoch + 1}/{train_settings.num_epochs}\n"
            summary += f"loss: {avg_train_loss:.6f} -"
            for m in metric_functions:
                summary += f" {m.name}: {history['train'][m.name][-1]:.6f} -"

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
                            m.update(targets, outputs, self.output_labels)

                avg_val_loss = val_loss / len(self.val_data)
                history["val"]["loss"].append(avg_val_loss)
                for m in metric_functions:
                    history["val"][m.name].append(m.result().item())
                    m.reset()

                summary += f" val_loss: {avg_val_loss:.6f} -"
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
                    raise Exception(f"Monitored value {callback.monitor} not found in fit history.")

                cb_result = callback.step(monitor_value[-1], self, verbose=verbose)
                if isinstance(callback, EarlyStopping) and cb_result:
                    return history

        return history

    def _enable_mc_dropout(self):
        """
        Enable MC Dropout by setting all dropout layers to train mode.
        """
        if self.model is None:
            raise Exception("A model needs to be set or loaded before enabling MC Dropout.")

        def set_dropout_train(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.model.apply(set_dropout_train)

    def _model_predict(self, inputs, batch_size, mc_dropout, verbose=0):
        self.model.eval()
        if mc_dropout:
            self._enable_mc_dropout()

        with torch.no_grad():
            if isinstance(inputs, dict):
                input_tensors = self._process_input_dict(inputs)
                inputs = {
                    key: value.to(self.device) for key, value in zip(inputs.keys(), input_tensors)
                }

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
        scaler=None,
        verbose=0,
    ):
        targets_df, val_targets_df = super().build_dataset(
            targets_df, batch_size, val_targets_df=val_targets_df, scaler=scaler
        )

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

            val_data = tf.data.Dataset.from_tensor_slices(
                (val_input_dict, np.array(val_targets_df.values))
            )
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

        self.optimizer = train_settings.optimizer
        if isinstance(self.optimizer, str):
            self.optimizer = get_optimizer_from_str(
                self.optimizer, backend="keras", learning_rate=train_settings.learning_rate
            )

        loss_function = train_settings.loss_function
        if isinstance(loss_function, str):
            loss_function = get_loss_from_str(loss_function, backend="keras")

        self.model.compile(
            optimizer=self.optimizer,
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

    def _enable_mc_dropout(self):
        """
        Enable MC Dropout by setting all dropout layers to train mode.
        """
        if self.model is None:
            raise Exception("A model needs to be set or loaded before enabling MC Dropout.")

        for layer in self.model.layers:
            if isinstance(layer, KerasMCDropout):
                layer.training = True

    def _model_predict(self, inputs, batch_size, mc_dropout, verbose=0):
        if mc_dropout:
            self._enable_mc_dropout()
        return self.model.predict(inputs, batch_size, verbose=verbose)

    def _save_weights(self, save_dir):
        weights_path = os.path.join(save_dir, f"{self.model.name}.weights.h5")
        self.model.save_weights(weights_path)

    def _load_weights(self, weights_path):
        if not weights_path.endswith(".weights.h5"):
            raise ValueError("Weights filename must end with .weights.h5")

        self.model.load_weights(weights_path)


class MultiModelWrapper:
    """
    _summary_
    """

    _default_precisions = ["ap_fixed<2, 1>", "ap_fixed<8, 3>", "ap_fixed<16, 6>"]
    _default_reuse_factors = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self):
        self._models: dict[str, BaseModelWrapper] = {}

    def add_model_wrapper(self, model_wrapper: BaseModelWrapper):
        key = "-".join([x.upper() for x in model_wrapper.output_labels])
        self._models[key] = model_wrapper

    def load_default_models(self):
        base_path = os.path.join(os.path.dirname(__file__), "weights", "v2", "gnn")
        default_paths = [
            os.path.join(base_path, "BRAM"),
            os.path.join(base_path, "DSP"),
            os.path.join(base_path, "FF"),
            os.path.join(base_path, "LUT"),
            os.path.join(base_path, "CYCLES"),
            os.path.join(base_path, "INTERVAL"),
        ]
        for path in default_paths:
            if not os.path.exists(path + ".config.json"):
                continue

            with open(f"{path}.config.json") as f:
                config = json.load(f)
            class_name = config.get("model_class", None)
            if class_name in [KerasMLP.__name__, KerasTransformer.__name__]:
                if os.path.exists(f"{path}.weights.h5"):
                    model_wrapper = KerasModelWrapper()
                    model_wrapper.load(f"{path}.config.json", f"{path}.weights.h5")
                    self.add_model_wrapper(model_wrapper)
                else:
                    raise FileNotFoundError(f"Weights file not found: {path}.weights.h5")
            elif class_name in [TorchMLP.__name__, TorchGNN.__name__]:
                if os.path.exists(f"{path}.weights.pt"):
                    model_wrapper = TorchModelWrapper()
                    model_wrapper.load(f"{path}.config.json", f"{path}.weights.pt")
                    self.add_model_wrapper(model_wrapper)
                else:
                    raise FileNotFoundError(f"Weights file not found: {path}.weights.pt")
            else:
                raise NotImplementedError(f"Model class not supported: {class_name}")

    def predict(self, models_to_predict, **kwargs):
        """
        Predict FPGA implementation metrics for one or more models across one or more
        configuration(s), and return the results as a tabular `pandas.DataFrame`.

        This method evaluates every combination of:
        - each entry in `models_to_predict`,
        - each entry in `hls_configs` (either provided or auto-generated from defaults or values seen during training),
        - each entry in `hls4ml_versions` (either provided or auto-generated from defaults or values seen during training),
        - each entry in `vivado_versions` (either provided or auto-generated from defaults or values seen during training)

        Parameters
        ----------
        models_to_predict : A Keras/Torch model or a list/tuple of such models
            A single model or a collection of models to evaluate.

        **kwargs
            Optional keyword arguments:

            hls_configs : dict or list[dict], optional
                One configuration dictionary or a list of configuration dictionaries describing the
                target board and hls4ml model settings. Keys may be in camelCase or snake_case; they
                are normalized internally.

                Expected structure (minimum):
                {
                    "board": "<board_name>",
                    "model": {
                        "strategy": "<strategy>",
                        "precision": "<precision>",
                        "reuse_factor": <int>
                    }
                }

                If omitted, configurations are generated from defaults and values seen during each estimator's training.

            hls4ml_versions : str or list[str] or tuple[str], optional
                The hls4ml version(s) to associate with the model(s) being predicted.

            vivado_versions : str or list[str] or tuple[str], optional
                The Vivado version(s) to associate with the model(s) being predicted.

            round_digits : int, default=2
                Number of decimal places to round the output DataFrame values to.

            verbose : int, default=0
                Verbosity level forwarded to each estimator's `predict` call.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of predictions with one row per (model, hls_config) pair.

        Raises
        ------
        TypeError
            If `hls_configs` is provided but is not a dict or a list/tuple of dicts.

        ValueError
            If a configuration specifies a board that is not present in `boards_data`.
        """

        if not self._models:
            raise Exception("No models have been added to the MultiModelWrapper.")

        if not isinstance(models_to_predict, (tuple, list)):
            models_to_predict = [models_to_predict]

        hls_configs = kwargs.get("hls_configs", None)
        hls_configs_by_estimator = {}
        if hls_configs is None:
            for estimator_key, estimator_model in self._models.items():
                estimator_boards = self._get_categorical_values(estimator_model, "board")
                estimator_strategies = self._get_categorical_values(estimator_model, "strategy")
                hls_configs_by_estimator[estimator_key] = [
                    {
                        "model": {
                            "strategy": strategy,
                            "precision": precision,
                            "reuse_factor": reuse_factor,
                        },
                        "board": board,
                    }
                    for board, strategy, precision, reuse_factor in itertools.product(
                        estimator_boards,
                        estimator_strategies,
                        MultiModelWrapper._default_precisions,
                        MultiModelWrapper._default_reuse_factors,
                    )
                ]
        else:
            if not isinstance(hls_configs, (tuple, list)):
                if not isinstance(hls_configs, dict):
                    raise TypeError(
                        "hls_configs argument expects a dictionary or list of dictionaries. See docstring example for correct formatting."
                    )
                hls_configs = [hls_configs]
            hls_configs_by_estimator = {key: list(hls_configs) for key in self._models.keys()}

        for estimator_configs in hls_configs_by_estimator.values():
            for idx in range(len(estimator_configs)):
                estimator_configs[idx] = camel_keys_to_snake(estimator_configs[idx])

        hls4ml_versions = kwargs.get("hls4ml_versions", None)
        hls4ml_versions_by_estimator = {}
        if hls4ml_versions is None:
            for estimator_key, estimator_model in self._models.items():
                hls4ml_versions_by_estimator[estimator_key] = self._get_categorical_values(
                    estimator_model, "hls4ml_version"
                )
        else:
            if isinstance(hls4ml_versions, str):
                hls4ml_versions = [hls4ml_versions]
            hls4ml_versions_by_estimator = {
                key: list(hls4ml_versions) for key in self._models.keys()
            }

        vivado_versions = kwargs.get("vivado_versions", None)
        vivado_versions_by_estimator = {}
        if vivado_versions is None:
            for estimator_key, estimator_model in self._models.items():
                vivado_versions_by_estimator[estimator_key] = self._get_categorical_values(
                    estimator_model, "vivado_version"
                )
        else:
            if isinstance(vivado_versions, str):
                vivado_versions = [vivado_versions]
            vivado_versions_by_estimator = {
                key: list(vivado_versions) for key in self._models.keys()
            }

        outputs = []
        outputs_by_key: dict[str, dict] = {}
        for key, estimator_model in self._models.items():
            estimator_hls_configs = hls_configs_by_estimator[key] or [None]
            estimator_hls4ml_versions = hls4ml_versions_by_estimator[key] or [None]
            estimator_vivado_versions = vivado_versions_by_estimator[key] or [None]
            estimator_product = list(
                itertools.product(
                    models_to_predict,
                    estimator_hls_configs,
                    estimator_hls4ml_versions,
                    estimator_vivado_versions,
                )
            )

            for model_to_predict, hls_config, hls4ml_version, vivado_version in estimator_product:
                model_name = getattr(model_to_predict, "name", model_to_predict.__class__.__name__)
                output_key = (
                    model_name,
                    hls_config["board"],
                    hls_config["model"]["strategy"],
                    hls_config["model"]["precision"],
                    hls_config["model"]["reuse_factor"],
                    hls4ml_version,
                    vivado_version,
                )
                if output_key not in outputs_by_key:
                    outputs_by_key[output_key] = {
                        "Model": model_name,
                        "Board": hls_config["board"],
                        "Strategy": hls_config["model"]["strategy"],
                        "Precision": hls_config["model"]["precision"],
                        "Reuse Factor": hls_config["model"]["reuse_factor"],
                        "HLS4ML Version": hls4ml_version,
                        "Vivado Version": vivado_version,
                    }
                    outputs.append(outputs_by_key[output_key])

            estimator_predictions = estimator_model.predict(
                models_to_predict,
                estimator_hls_configs,
                estimator_hls4ml_versions,
                estimator_vivado_versions,
                verbose=kwargs.get("verbose", 0),
            ).astype(float)

            target_labels = key.split("-")
            for idx, model_predictions in enumerate(estimator_predictions):
                model_to_predict, hls_config, hls4ml_version, vivado_version = estimator_product[
                    idx
                ]
                model_name = getattr(model_to_predict, "name", model_to_predict.__class__.__name__)
                output_key = (
                    model_name,
                    hls_config["board"],
                    hls_config["model"]["strategy"],
                    hls_config["model"]["precision"],
                    hls_config["model"]["reuse_factor"],
                    hls4ml_version,
                    vivado_version,
                )
                output_row = outputs_by_key[output_key]
                predictions = {}
                for k in range(len(model_predictions)):
                    predictions[target_labels[k]] = model_predictions[k]
                    available_key = f"max_{target_labels[k].lower()}"
                    if available_key in boards_data[output_row["Board"]]:
                        predictions[f"{target_labels[k]} (%)"] = (
                            model_predictions[k] / boards_data[output_row["Board"]][available_key]
                        ) * 100
                output_row.update(predictions)

        outputs_df = pd.DataFrame(outputs).round(kwargs.get("round_digits", 2))
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

    def _get_categorical_values(self, model_wrapper, key, fallback=[]):
        categorical_maps = getattr(model_wrapper, "global_categorical_maps", {}) or {}
        if key in categorical_maps:
            map_keys = list(categorical_maps[key].keys())
            return [k.lower() if isinstance(k, str) else k for k in map_keys]

        return list(fallback)
