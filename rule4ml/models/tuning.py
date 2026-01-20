import os
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


class KerasSearcher:
    """
    _summary_
    """

    def __init__(self):
        self.model_wrapper = KerasModelWrapper()

    def mlp_model_builder_generator(
        self, inputs_df, targets_df, categorical_maps, val_inputs_df=None, val_targets_df=None
    ):
        def mlp_model_builder(hp):
            mlp_settings = MLPSettings(
                embedding_layers=[
                    hp.Choice(name=f"embedding_output_{idx}", values=[4, 8, 16])
                    for idx in range(len(categorical_maps))
                ],
                numerical_dense_layers=[
                    hp.Choice(name=f"numerical_units_{idx}", values=[16, 32, 64, 128, 256])
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

            input_shape = (None, len(inputs_df.columns))
            output_shape = (None, len(targets_df.columns))

            target_labels = list(targets_df.columns)
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
                learning_rate=hp.Choice(name="learning_rate", values=[1e-4, 1e-3, 1e-2]),
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
            self.model_wrapper.set_model(mlp_model)

            if self.model_wrapper.dataset is None:
                self.model_wrapper.build_dataset(
                    inputs_df,
                    targets_df,
                    self.train_settings.batch_size,
                    val_ratio=0.15,
                    val_inputs_df=val_inputs_df,
                    val_targets_df=val_targets_df,
                    train_repeats=10,
                    shuffle=True,
                    verbose=0,
                )

            return self.model_wrapper.model

        return mlp_model_builder

    def mlp_search(
        self,
        inputs_df,
        targets_df,
        categorical_maps,
        val_inputs_df=None,
        val_targets_df=None,
        directory="./mlp_search",
        project_name=None,
        verbose=0,
    ):
        model_builder = self.mlp_model_builder_generator(
            inputs_df, targets_df, categorical_maps, val_inputs_df, val_targets_df
        )

        project_name = project_name or datetime.now().strftime('%Y%m%d-%H%M%S')
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

        self.tuner.search(
            self.model_wrapper.train_data,
            validation_data=self.model_wrapper.val_data,
            batch_size=self.model_wrapper.batch_size,
        )

    def load_tuner(
        self,
        inputs_df,
        targets_df,
        categorical_maps,
        directory,
        project_name,
        val_inputs_df=None,
        val_targets_df=None,
    ):
        model_builder = self.mlp_model_builder_generator(
            inputs_df, targets_df, categorical_maps, val_inputs_df, val_targets_df
        )

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

        best_hp = self.tuner.get_best_hyperparameters(1)[0]
        model_builder = self.mlp_model_builder_generator(inputs_df, targets_df, categorical_maps)
        model_builder(best_hp)


class TorchSearcher:
    def __init__(self, device=None):
        self.model_wrapper = TorchModelWrapper()
        self.study = None
        self.study_name = None
        self.db_path = None
        self.device = device
        self.trial_number = 0

    def _set_model(self, gnn_settings):
        target_labels = list(self.targets_df.columns)

        global_input_shape = (None, len(self.inputs_df.columns) - 1)  # -1 for sequential features
        sequential_input_shape = (None, len(self.inputs_df["sequential_inputs"].iloc[0].columns))
        output_shape = (None, len(target_labels))

        self.model_wrapper.set_model(
            TorchGNN(
                settings=gnn_settings,
                global_input_shape=global_input_shape,
                sequential_input_shape=sequential_input_shape,
                output_shape=output_shape,
                global_categorical_maps=self.global_categorical_maps,
                sequential_categorical_maps=self.sequential_categorical_maps,
                name=f"{'-'.join([x.upper() for x in target_labels])}_GNN_{self.trial_number-1}",
                device=self.device,
            )
        )

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
                for idx in range(len(self.global_categorical_maps))
            ],
            seq_embedding_layers=[
                trial.suggest_categorical(f"seq_embedding_{idx}", [8, 16, 32])
                for idx in range(len(self.sequential_categorical_maps))
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

        target_labels = list(self.targets_df.columns)
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
            learning_rate=trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3]),
            optimizer="adam",
            loss_function="msle",
            metrics=metrics,
        )

        if self.model_wrapper.dataset is None or self.batch_size != batch_size:
            self.model_wrapper.build_dataset(
                inputs_df=self.inputs_df,
                targets_df=self.targets_df,
                batch_size=train_settings.batch_size,
                val_ratio=0.15,
                val_inputs_df=self.val_inputs_df,
                val_targets_df=self.val_targets_df,
                shuffle=True,
            )

        path = os.path.dirname(self.db_path)
        callbacks = [
            EarlyStopping(
                monitor="val_loss", mode="min", patience=5, min_delta=0.0, restore_best=False
            ),
            ModelCheckpoint(dirpath=path, monitor="val_loss", mode="min"),
        ]

        history = self.model_wrapper.fit(train_settings, callbacks=callbacks, verbose=1)
        return np.min(history["val"]["loss"])

    def gnn_search(
        self,
        inputs_df,
        targets_df,
        global_categorical_maps,
        sequential_categorical_maps,
        val_inputs_df=None,
        val_targets_df=None,
        directory="./torch_search",
        project_name=None,
        n_trials=20,
        n_epochs=5,
        batch_size=None,
        direction="minimize",
    ):
        self.num_epochs = n_epochs
        self.inputs_df = inputs_df
        self.targets_df = targets_df
        self.val_inputs_df = val_inputs_df
        self.val_targets_df = val_targets_df
        self.global_categorical_maps = global_categorical_maps
        self.sequential_categorical_maps = sequential_categorical_maps
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
        self.study.optimize(self.gnn_objective, n_trials=n_trials, show_progress_bar=True)

    def load_tuner(
        self,
        inputs_df,
        targets_df,
        global_categorical_maps,
        sequential_categorical_maps,
        directory,
        project_name,
        val_inputs_df=None,
        val_targets_df=None,
    ):
        self.inputs_df = inputs_df
        self.targets_df = targets_df
        self.val_inputs_df = val_inputs_df
        self.val_targets_df = val_targets_df
        self.global_categorical_maps = global_categorical_maps
        self.sequential_categorical_maps = sequential_categorical_maps

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
                for idx in range(len(self.global_categorical_maps))
            ],
            seq_embedding_layers=[
                best_params[f"seq_embedding_{idx}"]
                for idx in range(len(self.sequential_categorical_maps))
            ],
            numerical_dense_layers=[
                best_params[f"numerical_dense_{idx}"]
                for idx in range(best_params["numerical_dense_count"])
            ],
            gconv_layers=[best_params[f"gconv_{idx}"] for idx in range(best_params["gconv_count"])],
            dense_layers=[best_params[f"dense_{idx}"] for idx in range(best_params["dense_count"])],
            dense_dropouts=[
                best_params[f"dropout_{idx}"] for idx in range(best_params["dense_count"])
            ],
            bayesian_dense=bayesian_dense,
            bayesian_output=any(bayesian_dense),
        )
        return gnn_settings

    def best_train_settings(self):
        best_params = self.best_params()
        train_settings = TrainSettings(
            num_epochs=best_params.get("num_epochs", 20),
            batch_size=best_params.get("batch_size", 64),
            learning_rate=best_params.get("learning_rate", 1e-4),
            optimizer=best_params.get("optimizer", "adam"),
            loss_function=best_params.get("loss_function", "msle"),
            metrics=[
                TorchParametricSMAPE(idx, name=f"smape_{idx}", eps=1)
                for idx in range(len(self.targets_df.columns))
            ]
            + [
                TorchParametricR2(idx, name=f"r2_{idx}", eps=1)
                for idx in range(len(self.targets_df.columns))
            ],
        )
        return train_settings

    def best_trial(self):
        return self.study.best_trial if self.study else None

    def load_best(self):
        if self.study is None:
            raise ValueError("No study loaded. Please run gnn_search or load_tuner first.")

        best_params = self.best_params()
        bayesian_dense = [
            best_params.get(f"bayesian_dense_{idx}", False)
            for idx in range(best_params["dense_count"])
        ]
        gnn_settings = GNNSettings(
            global_embedding_layers=[
                best_params[f"global_embedding_{idx}"]
                for idx in range(len(self.global_categorical_maps))
            ],
            seq_embedding_layers=[
                best_params[f"seq_embedding_{idx}"]
                for idx in range(len(self.sequential_categorical_maps))
            ],
            numerical_dense_layers=[
                best_params[f"numerical_dense_{idx}"]
                for idx in range(best_params["numerical_dense_count"])
            ],
            gconv_layers=[best_params[f"gconv_{idx}"] for idx in range(best_params["gconv_count"])],
            dense_layers=[best_params[f"dense_{idx}"] for idx in range(best_params["dense_count"])],
            dense_dropouts=[
                best_params[f"dropout_{idx}"] for idx in range(best_params["dense_count"])
            ],
            bayesian_dense=bayesian_dense,
            bayesian_output=any(bayesian_dense),
        )
        self._set_model(gnn_settings)
