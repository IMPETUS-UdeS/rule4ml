import os
from dataclasses import dataclass, field
from datetime import datetime

import keras
import keras_tuner as kt
import numpy as np
import optuna

from rule4ml.models.architectures import GNNSettings, KerasMLP, MLPSettings, TorchGNN
from rule4ml.models.callbacks import EarlyStopping, ModelCheckpoint
from rule4ml.models.metrics import (
    KerasParametricR2,
    KerasParametricSMAPE,
    TorchParametricR2,
    TorchParametricSMAPE,
)
from rule4ml.models.wrappers import KerasModelWrapper, TorchModelWrapper, TrainSettings


@dataclass(frozen=True)
class SearchData:
    """Datasets and features every trial of a search trains on."""

    train: object  # rule4ml.dataset.Dataset
    val: object
    global_labels: list
    target_labels: list
    global_categorical_maps: dict
    sequential_labels: list = field(default_factory=list)
    sequential_categorical_maps: dict = field(default_factory=dict)
    report: str | None = None
    scaler: object = None

    def set_model(self, wrapper, model):
        wrapper.set_model(
            model,
            self.global_labels,
            self.target_labels,
            self.sequential_labels,
            report=self.report,
            scaler=self.scaler,
        )


class KerasSearcher:
    """Hyperparameter search for Keras predictors."""

    def __init__(self):
        self.model_wrapper = KerasModelWrapper()
        self.arrays = None

    def mlp_model_builder_generator(self, data: SearchData):
        self.arrays = None
        categorical_maps = data.global_categorical_maps

        def mlp_model_builder(hp):
            mlp_settings = MLPSettings(
                embedding_layers=[
                    hp.Choice(name=f"embedding_output_{idx}", values=[4, 8, 16])
                    for idx in range(len(categorical_maps))
                ],
                numerical_dense_layers=[
                    hp.Choice(
                        name=f"numerical_units_{idx}", values=[16, 32, 64, 128, 256]
                    )
                    for idx in range(hp.Int("numerical_count", 1, 5))
                ],
                dense_layers=[
                    hp.Choice(name=f"units_{idx}", values=[16, 32, 64, 128, 256])
                    for idx in range(hp.Int("dense_count", 1, 8))
                ],
                dense_dropouts=[
                    hp.Choice(name=f"dropout_{idx}", values=[0.0, 0.1, 0.25, 0.5])
                    for idx in range(hp.Int("dropout_count", 1, 8))
                ],
            )

            input_shape = (None, len(data.global_labels) + len(categorical_maps))
            output_shape = (None, len(data.target_labels))

            target_labels = data.target_labels
            metrics = [
                KerasParametricSMAPE(idx, name=f"smape_{target_labels[idx]}", eps=1)
                for idx in range(len(target_labels))
            ]
            metrics += [
                KerasParametricR2(idx, name=f"r2_{target_labels[idx]}", eps=1)
                for idx in range(len(target_labels))
            ]

            self.train_settings = TrainSettings(
                num_epochs=20,
                batch_size=64,
                # batch_size=hp.Choice(name="batch_size", values=[32, 64, 128, 256]),
                learning_rate=hp.Choice(
                    name="learning_rate", values=[1e-4, 1e-3, 1e-2]
                ),
                optimizer=keras.optimizers.Adam,
                loss_function="msle",
                metrics=metrics,
            )

            mlp_model = KerasMLP(
                settings=mlp_settings,
                input_shape=input_shape,
                output_shape=output_shape,
                categorical_maps=categorical_maps,
                name=f"{'-'.join([x.upper() for x in target_labels])}_MLP",
            )
            mlp_model.compile(
                optimizer=self.train_settings.optimizer(
                    learning_rate=self.train_settings.learning_rate
                ),
                loss=self.train_settings.loss_function,
                metrics=self.train_settings.metrics,
            )
            data.set_model(self.model_wrapper, mlp_model)

            if self.arrays is None:  # input names do not depend on hyperparameters
                self.arrays = (
                    self.model_wrapper.training_data(data.train, fit_scaler=True),
                    self.model_wrapper.training_data(data.val),
                )

            return self.model_wrapper.model

        return mlp_model_builder

    def mlp_search(
        self, data: SearchData, directory="./mlp_search", project_name=None, verbose=0
    ):
        model_builder = self.mlp_model_builder_generator(data)

        project_name = project_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tuner = kt.Hyperband(
            hypermodel=model_builder,
            objective="val_loss",
            max_epochs=20,
            factor=3,
            hyperband_iterations=1,
            directory=directory,
            project_name=project_name,
        )

        if verbose > 0:
            self.tuner.search_space_summary()

        (inputs, targets), val = self.arrays
        self.tuner.search(
            inputs,
            targets,
            validation_data=val,
            batch_size=self.train_settings.batch_size,
        )

    def load_tuner(self, data: SearchData, directory, project_name):
        """Load a finished search and set an untrained model with the best settings."""
        model_builder = self.mlp_model_builder_generator(data)

        self.tuner = kt.Hyperband(
            hypermodel=model_builder,
            objective="val_loss",
            max_epochs=20,
            factor=3,
            hyperband_iterations=1,
            directory=directory,
            project_name=project_name,
            overwrite=False,
        )

        model_builder(self.tuner.get_best_hyperparameters(1)[0])


class TorchSearcher:
    def __init__(self, device=None, seed=0):
        self.model_wrapper = TorchModelWrapper()
        self.data = None
        self.seed = seed
        self.study = None
        self.study_name = None
        self.db_path = None
        self.device = device
        self.trial_number = 0

    def _set_model(self, gnn_settings):
        data = self.data
        model = TorchGNN(
            settings=gnn_settings,
            global_input_shape=(
                None,
                len(data.global_labels) + len(data.global_categorical_maps),
            ),
            sequential_input_shape=(
                None,
                len(data.sequential_labels) + len(data.sequential_categorical_maps),
            ),
            output_shape=(None, len(data.target_labels)),
            global_categorical_maps=data.global_categorical_maps,
            sequential_categorical_maps=data.sequential_categorical_maps,
            name=f"{'-'.join([x.upper() for x in data.target_labels])}_GNN_{self.trial_number - 1}",
            device=self.device,
        )
        data.set_model(self.model_wrapper, model)

    def gnn_objective(self, trial):
        self.trial_number += 1
        numerical_dense_count = trial.suggest_int("numerical_dense_count", 0, 2)
        gconv_count = trial.suggest_int("gconv_count", 1, 5)
        dense_count = trial.suggest_int("dense_count", 1, 5)
        bayesian_dense = [
            trial.suggest_categorical(f"bayesian_dense_{idx}", [True, False])
            for idx in range(dense_count)
        ]
        gnn_settings = GNNSettings(
            global_embedding_layers=[
                trial.suggest_categorical(f"global_embedding_{idx}", [8, 16, 32])
                for idx in range(len(self.data.global_categorical_maps))
            ],
            seq_embedding_layers=[
                trial.suggest_categorical(f"seq_embedding_{idx}", [8, 16, 32])
                for idx in range(len(self.data.sequential_categorical_maps))
            ],
            numerical_dense_layers=[
                trial.suggest_categorical(f"numerical_dense_{idx}", [16, 32, 64])
                for idx in range(numerical_dense_count)
            ],
            gconv_layers=[
                trial.suggest_categorical(f"gconv_{idx}", [16, 32, 64, 128])
                for idx in range(gconv_count)
            ],
            dense_layers=[
                trial.suggest_categorical(f"dense_{idx}", [16, 32, 64, 128])
                for idx in range(dense_count)
            ],
            dense_dropouts=[
                trial.suggest_categorical(f"dropout_{idx}", [0.0, 0.1, 0.2, 0.5])
                for idx in range(dense_count)
            ],
            bayesian_dense=bayesian_dense,
            bayesian_output=any(bayesian_dense),
        )
        self._set_model(gnn_settings)

        target_labels = self.data.target_labels
        metrics = [
            TorchParametricSMAPE(idx, name=f"smape_{target_labels[idx]}", eps=1e-6)
            for idx in range(len(target_labels))
        ]
        metrics += [
            TorchParametricR2(idx, name=f"r2_{target_labels[idx]}", eps=1e-6)
            for idx in range(len(target_labels))
        ]

        if self.batch_size:
            batch_size = self.batch_size
        else:
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        train_settings = TrainSettings(
            num_epochs=self.n_epochs,
            batch_size=batch_size,
            learning_rate=trial.suggest_categorical(
                "learning_rate", [1e-5, 1e-4, 1e-3]
            ),
            optimizer="adam",
            loss_function="msle",
            metrics=metrics,
        )

        path = os.path.dirname(self.db_path)
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=5,
                min_delta=0.0,
                restore_best=False,
            ),
            ModelCheckpoint(dirpath=path, monitor="val_loss", mode="min"),
        ]

        history = self.model_wrapper.fit(
            self.data.train,
            train_settings,
            val_dataset=self.data.val,
            seed=self.seed,
            callbacks=callbacks,
            verbose=1,
        )
        return np.min(history["val"]["loss"])

    def gnn_search(
        self,
        data: SearchData,
        directory="./torch_search",
        project_name=None,
        n_trials=20,
        n_epochs=5,
        batch_size=None,
        direction="minimize",
    ):
        self.data = data
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        os.makedirs(directory, exist_ok=True)
        project_name = project_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.study_name = project_name
        self.db_path = os.path.join(directory, f"{project_name}.db")
        db_uri = f"sqlite:///{self.db_path}"

        self.study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            direction=direction,
            study_name=self.study_name,
            storage=db_uri,
        )
        self.study.set_user_attr("num_epochs", n_epochs)
        self.study.set_user_attr("batch_size", batch_size)
        self.study.optimize(
            self.gnn_objective, n_trials=n_trials, show_progress_bar=True
        )

    def load_tuner(self, data: SearchData, directory, project_name):
        self.data = data

        self.study_name = project_name
        self.db_path = os.path.join(directory, f"{project_name}.db")
        db_uri = f"sqlite:///{self.db_path}"

        self.study = optuna.load_study(
            study_name=self.study_name,
            storage=db_uri,
        )

    def best_params(self):
        return self.study.best_params if self.study else {}

    def best_gnn_settings(self):
        best_params = self.best_params()
        bayesian_dense = [
            best_params.get(f"bayesian_dense_{idx}", False)
            for idx in range(best_params["dense_count"])
        ]
        gnn_settings = GNNSettings(
            global_embedding_layers=[
                best_params[f"global_embedding_{idx}"]
                for idx in range(len(self.data.global_categorical_maps))
            ],
            seq_embedding_layers=[
                best_params[f"seq_embedding_{idx}"]
                for idx in range(len(self.data.sequential_categorical_maps))
            ],
            numerical_dense_layers=[
                best_params[f"numerical_dense_{idx}"]
                for idx in range(best_params["numerical_dense_count"])
            ],
            gconv_layers=[
                best_params[f"gconv_{idx}"] for idx in range(best_params["gconv_count"])
            ],
            dense_layers=[
                best_params[f"dense_{idx}"] for idx in range(best_params["dense_count"])
            ],
            dense_dropouts=[
                best_params[f"dropout_{idx}"]
                for idx in range(best_params["dense_count"])
            ],
            bayesian_dense=bayesian_dense,
            bayesian_output=any(bayesian_dense),
        )
        return gnn_settings

    def best_train_settings(self):
        best_params = self.best_params()
        train_settings = TrainSettings(
            num_epochs=self.study.user_attrs.get("num_epochs", 20),
            batch_size=best_params.get("batch_size")
            or self.study.user_attrs.get("batch_size")
            or 64,
            learning_rate=best_params.get("learning_rate", 1e-4),
            optimizer=best_params.get("optimizer", "adam"),
            loss_function=best_params.get("loss_function", "msle"),
            metrics=[
                TorchParametricSMAPE(idx, name=f"smape_{idx}", eps=1)
                for idx in range(len(self.data.target_labels))
            ]
            + [
                TorchParametricR2(idx, name=f"r2_{idx}", eps=1)
                for idx in range(len(self.data.target_labels))
            ],
        )
        return train_settings

    def best_trial(self):
        return self.study.best_trial if self.study else None

    def load_best(self):
        """Set an untrained model with the best trial's settings."""
        if self.study is None:
            raise ValueError(
                "No study loaded. Please run gnn_search or load_tuner first."
            )
        self._set_model(self.best_gnn_settings())
