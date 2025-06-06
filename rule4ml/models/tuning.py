from datetime import datetime

import keras
import keras_tuner as kt

from rule4ml.models.architectures import (
    MLPSettings,
    KerasMLP,
)
from rule4ml.models.wrappers import (
    TrainSettings,
    KerasModelWrapper,
)
from rule4ml.models.metrics import KerasParametricSMAPE, KerasParametricR2


class Searcher:
    """
    _summary_
    """

    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    def mlp_model_builder_generator(self, inputs_df, targets_df, categorical_maps):
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
                KerasParametricSMAPE(idx, name=f"smape_{target_labels[idx]}", eps=1) \
                for idx in range(len(target_labels))
            ]
            metrics += [
                KerasParametricR2(idx, name=f"r2_{target_labels[idx]}", eps=1) \
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
                    train_repeats=10,
                    shuffle=True,
                    verbose=0,
                )

            return self.model_wrapper.model

        return mlp_model_builder

    def mlp_search(
        self, inputs_df, targets_df, categorical_maps, directory="./mlp_search", verbose=0
    ):
        model_builder = self.mlp_model_builder_generator(inputs_df, targets_df, categorical_maps)

        start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.tuner = kt.Hyperband(
            hypermodel=model_builder,
            objective="val_loss",
            max_epochs=20,
            factor=3,
            hyperband_iterations=1,
            directory=directory,
            project_name=start_time,
        )

        if verbose > 0:
            self.tuner.search_space_summary()

        self.tuner.search(
            self.model_wrapper.train_data,
            validation_data=self.model_wrapper.val_data,
            batch_size=self.model_wrapper.batch_size,
        )

    def load_tuner(self, inputs_df, targets_df, categorical_maps, directory, project_name):
        model_builder = self.mlp_model_builder_generator(inputs_df, targets_df, categorical_maps)

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
