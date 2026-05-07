# ==============================================================================
# src/evaluation/metrics.py
# Comprehensive evaluation metrics computation.
# Generates all metrics required for research-level analysis.
# ==============================================================================

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.config.settings import Config
from src.entity.model_entity import ModelMetrics
from src.utils.logger import get_logger
from src.utils.helpers import save_csv, save_json

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Calculates comprehensive evaluation metrics for trained models.
    Supports multiclass classification with one-vs-rest ROC/PR curves.
    """

    def __init__(self, config: Config):
        self.config = config
        self.class_names = config.data.class_names
        self.num_classes = config.data.num_classes

    def compute_all_metrics(
        self,
        model: tf.keras.Model,
        model_name: str,
        test_generator,
        model_path: Optional[str] = None,
    ) -> ModelMetrics:
        """
        Compute all evaluation metrics for a model on the test set.

        Args:
            model: Trained Keras model.
            model_name: Name of the model architecture.
            test_generator: Test data generator.
            model_path: Path to saved model file (for size measurement).

        Returns:
            ModelMetrics dataclass with all computed metrics.
        """
        logger.info(f"Computing metrics for {model_name}...")

        # ── Get predictions ──
        test_generator.reset()
        y_pred_proba = model.predict(test_generator, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = test_generator.classes[:len(y_pred)]

        # One-hot encode true labels for ROC/PR
        y_true_onehot = label_binarize(y_true, classes=range(self.num_classes))

        # ── Core Metrics ──
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        # ── AUC Score ──
        try:
            auc_val = roc_auc_score(y_true_onehot, y_pred_proba, multi_class="ovr", average="weighted")
        except Exception as e:
            logger.warning(f"Could not compute AUC for {model_name}: {e}")
            auc_val = 0.0

        # ── Confusion Matrix ──
        cm = confusion_matrix(y_true, y_pred)

        # ── Classification Report ──
        cls_report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0,
        )

        # ── ROC Curves (one-vs-rest) ──
        fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
        for i in range(self.num_classes):
            try:
                fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
                roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            except Exception:
                fpr_dict[i] = np.array([0, 1])
                tpr_dict[i] = np.array([0, 1])
                roc_auc_dict[i] = 0.0

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= self.num_classes
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(all_fpr, mean_tpr)

        # ── Precision-Recall Curves ──
        pr_prec_dict, pr_rec_dict, pr_auc_dict = {}, {}, {}
        for i in range(self.num_classes):
            try:
                pr_prec_dict[i], pr_rec_dict[i], _ = precision_recall_curve(
                    y_true_onehot[:, i], y_pred_proba[:, i]
                )
                pr_auc_dict[i] = average_precision_score(y_true_onehot[:, i], y_pred_proba[:, i])
            except Exception:
                pr_prec_dict[i] = np.array([0, 1])
                pr_rec_dict[i] = np.array([1, 0])
                pr_auc_dict[i] = 0.0

        # ── Inference Time ──
        inference_time = self._measure_inference_time(model)

        # ── Model Size ──
        total_params = model.count_params()
        trainable_params = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        model_size_mb = 0.0
        if model_path and os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        else:
            # Estimate from parameter count (4 bytes per float32 param)
            model_size_mb = (total_params * 4) / (1024 * 1024)

        # ── Build Result ──
        metrics = ModelMetrics(
            model_name=model_name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            auc_score=auc_val,
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            per_class_f1=per_class_f1,
            confusion_matrix=cm,
            classification_report=cls_report,
            fpr=fpr_dict,
            tpr=tpr_dict,
            roc_auc=roc_auc_dict,
            pr_precision=pr_prec_dict,
            pr_recall=pr_rec_dict,
            pr_auc=pr_auc_dict,
            inference_time_ms=inference_time,
            model_size_mb=model_size_mb,
            total_params=total_params,
            trainable_params=trainable_params,
        )

        logger.info(
            f"Metrics for {model_name}: "
            f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | "
            f"F1={f1:.4f} | AUC={auc_val:.4f} | "
            f"Inference={inference_time:.1f}ms | Size={model_size_mb:.1f}MB"
        )

        return metrics

    def _measure_inference_time(
        self, model: tf.keras.Model, num_runs: int = 50
    ) -> float:
        """
        Measure average inference time per image in milliseconds.

        Args:
            model: Keras model.
            num_runs: Number of inference runs to average.

        Returns:
            Average inference time in milliseconds.
        """
        dummy_input = np.random.rand(1, self.config.data.image_size,
                                     self.config.data.image_size, 3).astype(np.float32)

        # Warm up
        for _ in range(5):
            model.predict(dummy_input, verbose=0)

        # Measure
        times = []
        for _ in range(num_runs):
            start = time.time()
            model.predict(dummy_input, verbose=0)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        logger.debug(f"Inference time: {avg_time:.1f}ms (avg of {num_runs} runs)")
        return avg_time

    def save_metrics_report(
        self,
        all_metrics: List[ModelMetrics],
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Save a comparison table of all model metrics as CSV.

        Args:
            all_metrics: List of ModelMetrics for each model.
            output_dir: Output directory. Defaults to reports/metrics.

        Returns:
            Path to saved CSV file.
        """
        output_dir = output_dir or self.config.paths.metrics_dir
        rows = [m.to_summary_dict() for m in all_metrics]
        filepath = os.path.join(output_dir, "model_comparison.csv")
        save_csv(rows, filepath)
        logger.info(f"Model comparison saved to {filepath}")
        return filepath

    def save_classification_reports(
        self,
        all_metrics: List[ModelMetrics],
        output_dir: Optional[str] = None,
    ) -> None:
        """Save individual classification reports for each model."""
        output_dir = output_dir or self.config.paths.metrics_dir
        for m in all_metrics:
            filepath = os.path.join(output_dir, f"{m.model_name}_classification_report.txt")
            with open(filepath, "w") as f:
                f.write(f"Classification Report — {m.model_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(m.classification_report)
            logger.debug(f"Classification report saved: {filepath}")
