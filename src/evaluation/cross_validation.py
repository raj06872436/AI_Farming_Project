# ==============================================================================
# src/evaluation/cross_validation.py
# K-Fold Cross Validation implementation for robust model evaluation.
# ==============================================================================

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Optional

from src.config.settings import Config
from src.entity.model_entity import CrossValidationResult
from src.models.model_factory import ModelFactory
from src.utils.logger import get_logger
from src.utils.helpers import save_csv

logger = get_logger(__name__)


class CrossValidator:
    """
    Performs K-Fold Stratified Cross Validation for reliable model evaluation.
    Reports mean and standard deviation of metrics across folds.
    """

    def __init__(self, config: Config):
        self.config = config
        self.k_folds = config.cv.k_folds
        self.factory = ModelFactory(config)

    def run_cross_validation(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        epochs: Optional[int] = None,
    ) -> CrossValidationResult:
        """
        Run K-Fold Stratified Cross Validation for a given model.

        Args:
            model_name: Name of the model architecture to evaluate.
            X: Image data array of shape (N, H, W, C).
            y: One-hot encoded labels of shape (N, num_classes).
            epochs: Number of training epochs per fold.

        Returns:
            CrossValidationResult with per-fold metrics.
        """
        epochs = epochs or self.config.cv.cv_epochs
        y_indices = np.argmax(y, axis=1)

        skf = StratifiedKFold(
            n_splits=self.k_folds, shuffle=True, random_state=self.config.data.seed
        )

        result = CrossValidationResult(model_name=model_name, k_folds=self.k_folds)

        logger.info(f"Starting {self.k_folds}-Fold CV for {model_name}...")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_indices)):
            logger.info(f"  Fold {fold_idx + 1}/{self.k_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Build a fresh model for each fold
            try:
                model = self.factory.build_model(model_name)

                # Train
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=self.config.data.batch_size,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=3, restore_best_weights=True
                        ),
                    ],
                )

                # Evaluate
                y_pred_proba = model.predict(X_val, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
                y_true = np.argmax(y_val, axis=1)

                fold_acc = accuracy_score(y_true, y_pred)
                fold_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                fold_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                fold_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                result.fold_accuracies.append(fold_acc)
                result.fold_precisions.append(fold_prec)
                result.fold_recalls.append(fold_rec)
                result.fold_f1_scores.append(fold_f1)

                logger.info(
                    f"    Fold {fold_idx + 1}: Acc={fold_acc:.4f} | "
                    f"Prec={fold_prec:.4f} | Rec={fold_rec:.4f} | F1={fold_f1:.4f}"
                )

            except Exception as e:
                logger.error(f"    Fold {fold_idx + 1} failed: {e}")
                continue

            finally:
                # Free memory
                tf.keras.backend.clear_session()

        logger.info(
            f"CV Complete for {model_name}: "
            f"Acc={result.mean_accuracy:.4f}±{result.std_accuracy:.4f} | "
            f"F1={result.mean_f1:.4f}±{result.std_f1:.4f}"
        )

        return result

    def run_all_models(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[CrossValidationResult]:
        """
        Run cross validation for all models specified in configuration.

        Args:
            X: Image data array.
            y: One-hot encoded labels.

        Returns:
            List of CrossValidationResult for each model.
        """
        results = []
        for model_name in self.config.model.model_names:
            try:
                result = self.run_cross_validation(model_name, X, y)
                results.append(result)
            except Exception as e:
                logger.error(f"CV failed for {model_name}: {e}")
        return results

    def save_cv_results(
        self,
        results: List[CrossValidationResult],
        output_dir: Optional[str] = None,
    ) -> str:
        """Save cross validation results to CSV."""
        output_dir = output_dir or self.config.paths.metrics_dir
        rows = [r.to_summary_dict() for r in results]
        filepath = os.path.join(output_dir, "cross_validation_results.csv")
        save_csv(rows, filepath)
        logger.info(f"CV results saved to {filepath}")
        return filepath


import os
