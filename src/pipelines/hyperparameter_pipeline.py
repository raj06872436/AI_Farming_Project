# ==============================================================================
# src/pipelines/hyperparameter_pipeline.py
# Hyperparameter tuning pipeline using Keras Tuner.
# ==============================================================================

import os
from typing import Optional, Dict

import tensorflow as tf

from src.config.settings import Config
from src.data.dataset import DatasetManager
from src.models.model_factory import ModelFactory
from src.utils.logger import get_logger
from src.utils.helpers import save_json, save_csv

logger = get_logger(__name__)


class HyperparameterPipeline:
    """
    Hyperparameter tuning using Keras Tuner.
    Tunes learning rate, dropout, dense units, and optimizer.
    Falls back to manual grid search if keras-tuner is unavailable.
    """

    def __init__(self, config: Config):
        self.config = config
        self.dataset_mgr = DatasetManager(config)
        self.factory = ModelFactory(config)

    def _build_tunable_model(self, hp, model_name: str):
        """Build a model with tunable hyperparameters."""
        builder = self.factory.get_builder(model_name)

        # Tunable hyperparameters
        lr = hp.Choice("learning_rate", values=self.config.hp.lr_choices)
        dropout = hp.Choice("dropout_rate", values=self.config.hp.dropout_choices)
        dense_units = hp.Choice("dense_units", values=self.config.hp.dense_units_choices)
        optimizer_name = hp.Choice("optimizer", values=self.config.hp.optimizer_choices)

        model = builder.build(
            num_classes=self.dataset_mgr.num_classes,
            dense_units_1=dense_units,
            dense_units_2=max(dense_units // 2, 32),
            dropout_rate=dropout,
        )

        model = builder.compile_model(model, learning_rate=lr, optimizer_name=optimizer_name)
        return model

    def run_tuning(self, model_name: str) -> Dict:
        """
        Run hyperparameter tuning for a given model.

        Args:
            model_name: Name of the model to tune.

        Returns:
            Dictionary with best hyperparameters and results.
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}...")

        train_gen, val_gen, _ = self.dataset_mgr.prepare_data(use_augmentation=True)

        try:
            import keras_tuner as kt

            tuner = kt.RandomSearch(
                hypermodel=lambda hp: self._build_tunable_model(hp, model_name),
                objective="val_accuracy",
                max_trials=self.config.hp.max_trials,
                executions_per_trial=self.config.hp.executions_per_trial,
                directory=os.path.join(self.config.paths.logs_dir, "hp_tuning"),
                project_name=model_name,
                overwrite=True,
            )

            tuner.search(
                train_gen,
                validation_data=val_gen,
                epochs=self.config.hp.tuner_epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True
                    ),
                ],
                verbose=1,
            )

            # Get best hyperparameters
            best_hp = tuner.get_best_hyperparameters(1)[0]
            best_params = {
                "model": model_name,
                "learning_rate": best_hp.get("learning_rate"),
                "dropout_rate": best_hp.get("dropout_rate"),
                "dense_units": best_hp.get("dense_units"),
                "optimizer": best_hp.get("optimizer"),
            }

            # Get best model results
            best_model = tuner.get_best_models(1)[0]
            val_loss, val_acc = best_model.evaluate(val_gen, verbose=0)
            best_params["best_val_accuracy"] = round(val_acc, 4)
            best_params["best_val_loss"] = round(val_loss, 4)

            # Save search summary
            tuner_summary = tuner.results_summary(num_trials=5)

        except ImportError:
            logger.warning("keras-tuner not installed. Running manual grid search...")
            best_params = self._manual_grid_search(model_name, train_gen, val_gen)

        except Exception as e:
            logger.error(f"Tuning failed for {model_name}: {e}")
            best_params = {"model": model_name, "error": str(e)}

        finally:
            tf.keras.backend.clear_session()

        # Save results
        output_path = os.path.join(
            self.config.paths.summary_dir, f"{model_name}_best_hyperparameters.json"
        )
        save_json(best_params, output_path)
        logger.info(f"Best params for {model_name}: {best_params}")

        return best_params

    def _manual_grid_search(self, model_name: str, train_gen, val_gen) -> Dict:
        """Fallback manual grid search if keras-tuner is unavailable."""
        logger.info(f"Manual grid search for {model_name}...")

        best_acc = 0.0
        best_params = {}

        # Simplified grid
        for lr in [1e-3, 1e-4]:
            for dropout in [0.3, 0.5]:
                for units in [128, 256]:
                    try:
                        builder = self.factory.get_builder(model_name)
                        model = builder.build(
                            num_classes=self.dataset_mgr.num_classes,
                            dense_units_1=units,
                            dense_units_2=units // 2,
                            dropout_rate=dropout,
                        )
                        model = builder.compile_model(model, learning_rate=lr)

                        model.fit(
                            train_gen, validation_data=val_gen,
                            epochs=3, verbose=0,
                        )

                        val_loss, val_acc = model.evaluate(val_gen, verbose=0)

                        if val_acc > best_acc:
                            best_acc = val_acc
                            best_params = {
                                "model": model_name,
                                "learning_rate": lr,
                                "dropout_rate": dropout,
                                "dense_units": units,
                                "optimizer": "adam",
                                "best_val_accuracy": round(val_acc, 4),
                                "best_val_loss": round(val_loss, 4),
                            }

                        tf.keras.backend.clear_session()

                    except Exception as e:
                        logger.warning(f"Grid search trial failed: {e}")
                        continue

        return best_params

    def run_all_models(self) -> Dict[str, Dict]:
        """Run tuning for all configured models."""
        all_results = {}
        for model_name in self.config.model.model_names:
            result = self.run_tuning(model_name)
            all_results[model_name] = result

        # Save combined results
        rows = list(all_results.values())
        save_csv(rows, os.path.join(
            self.config.paths.summary_dir, "hyperparameter_tuning_results.csv"
        ))

        return all_results
