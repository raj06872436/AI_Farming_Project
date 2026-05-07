# ==============================================================================
# src/entity/model_entity.py
# Data classes representing results and metrics throughout the pipeline.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class TrainingResult:
    """Stores results from a single model training run."""
    model_name: str
    history: Dict[str, List[float]]  # keras history.history dict
    training_time_seconds: float = 0.0
    initial_epochs: int = 0
    fine_tune_epochs: int = 0
    total_epochs: int = 0
    best_val_accuracy: float = 0.0
    best_val_loss: float = float("inf")
    final_train_accuracy: float = 0.0
    final_train_loss: float = 0.0

    def __post_init__(self):
        if self.history:
            val_acc = self.history.get("val_accuracy", [0])
            val_loss = self.history.get("val_loss", [float("inf")])
            train_acc = self.history.get("accuracy", [0])
            train_loss = self.history.get("loss", [0])
            self.best_val_accuracy = max(val_acc) if val_acc else 0.0
            self.best_val_loss = min(val_loss) if val_loss else float("inf")
            self.final_train_accuracy = train_acc[-1] if train_acc else 0.0
            self.final_train_loss = train_loss[-1] if train_loss else 0.0
            self.total_epochs = len(train_acc)


@dataclass
class ModelMetrics:
    """Stores comprehensive evaluation metrics for a model."""
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    # Per-class metrics
    per_class_precision: Optional[np.ndarray] = None
    per_class_recall: Optional[np.ndarray] = None
    per_class_f1: Optional[np.ndarray] = None
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    # Classification report (string)
    classification_report: str = ""
    # ROC data
    fpr: Optional[Dict[int, np.ndarray]] = None
    tpr: Optional[Dict[int, np.ndarray]] = None
    roc_auc: Optional[Dict[int, float]] = None
    # PR curve data
    pr_precision: Optional[Dict[int, np.ndarray]] = None
    pr_recall: Optional[Dict[int, np.ndarray]] = None
    pr_auc: Optional[Dict[int, float]] = None
    # Timing
    inference_time_ms: float = 0.0
    # Model size
    model_size_mb: float = 0.0
    total_params: int = 0
    trainable_params: int = 0

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a flat summary for CSV export."""
        return {
            "Model": self.model_name,
            "Accuracy": round(self.accuracy, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1 Score": round(self.f1_score, 4),
            "AUC": round(self.auc_score, 4),
            "Inference Time (ms)": round(self.inference_time_ms, 2),
            "Model Size (MB)": round(self.model_size_mb, 2),
            "Total Params": self.total_params,
            "Trainable Params": self.trainable_params,
        }


@dataclass
class EvaluationResult:
    """Aggregated evaluation result for a model."""
    model_name: str
    training_result: Optional[TrainingResult] = None
    metrics: Optional[ModelMetrics] = None


@dataclass
class CrossValidationResult:
    """Results from K-Fold cross validation."""
    model_name: str
    k_folds: int = 5
    fold_accuracies: List[float] = field(default_factory=list)
    fold_precisions: List[float] = field(default_factory=list)
    fold_recalls: List[float] = field(default_factory=list)
    fold_f1_scores: List[float] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_accuracies)) if self.fold_accuracies else 0.0

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.fold_accuracies)) if self.fold_accuracies else 0.0

    @property
    def mean_precision(self) -> float:
        return float(np.mean(self.fold_precisions)) if self.fold_precisions else 0.0

    @property
    def std_precision(self) -> float:
        return float(np.std(self.fold_precisions)) if self.fold_precisions else 0.0

    @property
    def mean_recall(self) -> float:
        return float(np.mean(self.fold_recalls)) if self.fold_recalls else 0.0

    @property
    def std_recall(self) -> float:
        return float(np.std(self.fold_recalls)) if self.fold_recalls else 0.0

    @property
    def mean_f1(self) -> float:
        return float(np.mean(self.fold_f1_scores)) if self.fold_f1_scores else 0.0

    @property
    def std_f1(self) -> float:
        return float(np.std(self.fold_f1_scores)) if self.fold_f1_scores else 0.0

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "Model": self.model_name,
            "K-Folds": self.k_folds,
            "Mean Accuracy": f"{self.mean_accuracy:.4f} ± {self.std_accuracy:.4f}",
            "Mean Precision": f"{self.mean_precision:.4f} ± {self.std_precision:.4f}",
            "Mean Recall": f"{self.mean_recall:.4f} ± {self.std_recall:.4f}",
            "Mean F1": f"{self.mean_f1:.4f} ± {self.std_f1:.4f}",
        }


@dataclass
class AblationResult:
    """Results from ablation study experiments."""
    model_name: str
    experiment_name: str  # e.g., "base", "+augmentation", "+fine_tuning", "+hp_tuning"
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_time_seconds: float = 0.0

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "Model": self.model_name,
            "Experiment": self.experiment_name,
            "Accuracy": round(self.accuracy, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1 Score": round(self.f1_score, 4),
            "Training Time (s)": round(self.training_time_seconds, 1),
        }
