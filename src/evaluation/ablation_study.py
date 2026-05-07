# ==============================================================================
# src/evaluation/ablation_study.py
# Ablation study: systematically evaluate the impact of each component.
# ==============================================================================

import os
import time
from typing import List, Optional, Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config.settings import Config
from src.entity.model_entity import AblationResult
from src.models.model_factory import ModelFactory
from src.utils.logger import get_logger
from src.utils.helpers import save_csv

logger = get_logger(__name__)


class AblationStudy:
    """
    Performs ablation experiments to measure the contribution of each component:
    1. Base model (frozen, no augmentation)
    2. + Data augmentation
    3. + Fine-tuning (unfreezing top layers)
    4. + Hyperparameter tuning (best params)
    """

    def __init__(self, config: Config):
        self.config = config
        self.factory = ModelFactory(config)

    def run_single_experiment(
        self,
        model_name: str,
        train_gen,
        val_gen,
        experiment_name: str,
        use_augmentation: bool = False,
        fine_tune: bool = False,
        epochs: int = 5,
        learning_rate: Optional[float] = None,
    ) -> AblationResult:
        """Run a single ablation experiment."""
        lr = learning_rate or self.config.training.learning_rate
        logger.info(f"Ablation [{model_name}] - {experiment_name}")

        start_time = time.time()

        # Build model
        builder = self.factory.get_builder(model_name)
        model = builder.build()
        model = builder.compile_model(model, learning_rate=lr)

        # Train Phase 1
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                ),
            ],
        )

        # Fine-tune if requested
        if fine_tune:
            model = builder.unfreeze_for_fine_tuning(model)
            model = builder.compile_model(model, learning_rate=lr * 0.1)
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True
                    ),
                ],
            )

        training_time = time.time() - start_time

        # Evaluate
        val_gen.reset()
        y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
        y_true = val_gen.classes[:len(y_pred)]

        result = AblationResult(
            model_name=model_name,
            experiment_name=experiment_name,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
            recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f1_score=f1_score(y_true, y_pred, average="weighted", zero_division=0),
            training_time_seconds=training_time,
        )

        tf.keras.backend.clear_session()

        logger.info(
            f"  {experiment_name}: Acc={result.accuracy:.4f} | "
            f"F1={result.f1_score:.4f} | Time={training_time:.1f}s"
        )
        return result

    def run_full_ablation(
        self,
        model_name: str,
        train_gen_no_aug,
        train_gen_aug,
        val_gen,
    ) -> List[AblationResult]:
        """Run all ablation experiments for a model."""
        results = []
        epochs = min(self.config.training.initial_epochs, 5)

        # Experiment 1: Base model (no augmentation, no fine-tuning)
        r1 = self.run_single_experiment(
            model_name, train_gen_no_aug, val_gen,
            "Base Model", use_augmentation=False, fine_tune=False, epochs=epochs
        )
        results.append(r1)

        # Experiment 2: + Augmentation
        r2 = self.run_single_experiment(
            model_name, train_gen_aug, val_gen,
            "+ Augmentation", use_augmentation=True, fine_tune=False, epochs=epochs
        )
        results.append(r2)

        # Experiment 3: + Fine-tuning
        r3 = self.run_single_experiment(
            model_name, train_gen_aug, val_gen,
            "+ Fine-tuning", use_augmentation=True, fine_tune=True, epochs=epochs
        )
        results.append(r3)

        return results

    def save_ablation_results(
        self,
        results: List[AblationResult],
        output_dir: Optional[str] = None,
    ) -> str:
        """Save ablation results to CSV."""
        output_dir = output_dir or self.config.paths.summary_dir
        os.makedirs(output_dir, exist_ok=True)
        rows = [r.to_summary_dict() for r in results]
        filepath = os.path.join(output_dir, "ablation_study.csv")
        save_csv(rows, filepath)
        logger.info(f"Ablation results saved to {filepath}")
        return filepath
