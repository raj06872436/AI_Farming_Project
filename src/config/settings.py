# ==============================================================================
# src/config/settings.py
# Central configuration for the entire project.
# Single source of truth for all hyperparameters, paths, and model settings.
# ==============================================================================

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DataConfig:
    """Configuration for dataset and preprocessing."""
    dataset_path: str = os.getenv("DATASET_PATH", "PlantVillage")
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    num_classes: int = int(os.getenv("NUM_CLASSES", "15"))
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = int(os.getenv("SEED", "42"))

    # Augmentation parameters
    rotation_range: float = 30.0
    horizontal_flip: bool = True
    zoom_range: float = 0.2
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    width_shift_range: float = 0.15
    height_shift_range: float = 0.15
    fill_mode: str = "nearest"

    # Class names (auto-populated from dataset directory)
    class_names: List[str] = field(default_factory=list)

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (self.image_size, self.image_size, 3)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    initial_epochs: int = int(os.getenv("INITIAL_EPOCHS", "5"))
    fine_tune_epochs: int = int(os.getenv("FINE_TUNE_EPOCHS", "15"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.0001"))
    fine_tune_lr: float = float(os.getenv("FINE_TUNE_LR", "0.00001"))
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"
    
    # Classifier head
    dense_units_1: int = 256
    dense_units_2: int = 128
    dropout_rate: float = 0.3

    # Callbacks
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Fine-tuning: number of layers to unfreeze from the top
    fine_tune_layers: int = 50


@dataclass
class ModelConfig:
    """Configuration for model architectures."""
    model_names: List[str] = field(default_factory=lambda: [
        m.strip() for m in os.getenv("MODELS", "MobileNetV2,ResNet50,EfficientNetB0,DenseNet121").split(",")
    ])
    weights: str = "imagenet"
    include_top: bool = False


@dataclass
class PathConfig:
    """Configuration for all output paths."""
    saved_models_dir: str = os.getenv("SAVED_MODELS_DIR", "saved_models")
    reports_dir: str = os.getenv("REPORTS_DIR", "reports")
    logs_dir: str = os.getenv("LOGS_DIR", "logs")

    @property
    def graphs_dir(self) -> str:
        return os.path.join(self.reports_dir, "graphs")

    @property
    def confusion_matrix_dir(self) -> str:
        return os.path.join(self.reports_dir, "confusion_matrix")

    @property
    def roc_curves_dir(self) -> str:
        return os.path.join(self.reports_dir, "roc_curves")

    @property
    def metrics_dir(self) -> str:
        return os.path.join(self.reports_dir, "metrics")

    @property
    def gradcam_dir(self) -> str:
        return os.path.join(self.reports_dir, "gradcam")

    @property
    def summary_dir(self) -> str:
        return os.path.join(self.reports_dir, "summary")

    def create_all_dirs(self):
        """Create all output directories if they don't exist."""
        dirs = [
            self.saved_models_dir,
            self.reports_dir,
            self.graphs_dir,
            self.confusion_matrix_dir,
            self.roc_curves_dir,
            self.metrics_dir,
            self.gradcam_dir,
            self.summary_dir,
            self.logs_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


@dataclass
class CrossValidationConfig:
    """Configuration for K-Fold cross validation."""
    k_folds: int = int(os.getenv("K_FOLDS", "5"))
    # Use fewer epochs for CV to save time
    cv_epochs: int = 10


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter tuning."""
    max_trials: int = 10
    executions_per_trial: int = 1
    tuner_epochs: int = 10
    # Search spaces
    lr_choices: List[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
    dropout_choices: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5])
    dense_units_choices: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    optimizer_choices: List[str] = field(default_factory=lambda: ["adam", "sgd", "rmsprop"])


@dataclass
class Config:
    """Master configuration aggregating all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    hp: HyperparameterConfig = field(default_factory=HyperparameterConfig)

    def __post_init__(self):
        """Initialize directories and load class names."""
        self.paths.create_all_dirs()
        self._load_class_names()

    def _load_class_names(self):
        """Auto-detect class names from dataset directory."""
        if os.path.isdir(self.data.dataset_path):
            self.data.class_names = sorted([
                d for d in os.listdir(self.data.dataset_path)
                if os.path.isdir(os.path.join(self.data.dataset_path, d))
            ])
            self.data.num_classes = len(self.data.class_names)

    def summary(self) -> Dict:
        """Return a summary dict of key configuration parameters."""
        return {
            "dataset_path": self.data.dataset_path,
            "image_size": self.data.image_size,
            "batch_size": self.data.batch_size,
            "num_classes": self.data.num_classes,
            "models": self.model.model_names,
            "initial_epochs": self.training.initial_epochs,
            "fine_tune_epochs": self.training.fine_tune_epochs,
            "learning_rate": self.training.learning_rate,
            "k_folds": self.cv.k_folds,
        }
