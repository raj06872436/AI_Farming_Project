# ==============================================================================
# src/data/dataset.py
# Dataset management: loading, splitting, augmentation, preprocessing.
# Ensures all models use the SAME data pipeline for fair comparison.
# ==============================================================================

import os
from typing import Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config.settings import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetManager:
    """
    Manages the PlantVillage dataset: loading, splitting, augmentation,
    and preprocessing. Provides a consistent data pipeline for all models.
    """

    def __init__(self, config: Config):
        """
        Initialize DatasetManager with project configuration.

        Args:
            config: Project configuration object.
        """
        self.config = config
        self.data_cfg = config.data
        self.class_names = config.data.class_names
        self.num_classes = config.data.num_classes

        # Data generators (set by prepare_data)
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

        # Full dataset arrays (set by load_full_dataset)
        self._X_full = None
        self._y_full = None

        logger.info(
            f"DatasetManager initialized | "
            f"Dataset: {self.data_cfg.dataset_path} | "
            f"Classes: {self.num_classes} | "
            f"Image size: {self.data_cfg.image_size}x{self.data_cfg.image_size}"
        )

    # ==================================================================
    # Generator-based loading (memory-efficient for training)
    # ==================================================================

    def prepare_data(self, use_augmentation: bool = True) -> Tuple:
        """
        Prepare train/validation/test data generators using ImageDataGenerator.
        Uses consistent preprocessing across all splits.

        Args:
            use_augmentation: Whether to apply data augmentation to training set.

        Returns:
            Tuple of (train_generator, val_generator, test_generator).
        """
        img_size = (self.data_cfg.image_size, self.data_cfg.image_size)
        batch_size = self.data_cfg.batch_size

        # ── Training Generator (with optional augmentation) ──
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,  # BUG FIX: was missing — images were 0-255 during training but 0-1 during validation
                rotation_range=self.data_cfg.rotation_range,
                horizontal_flip=self.data_cfg.horizontal_flip,
                vertical_flip=True,
                zoom_range=self.data_cfg.zoom_range,
                brightness_range=self.data_cfg.brightness_range,
                width_shift_range=self.data_cfg.width_shift_range,
                height_shift_range=self.data_cfg.height_shift_range,
                shear_range=0.15,
                fill_mode=self.data_cfg.fill_mode,
                validation_split=1 - self.data_cfg.train_split,
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                validation_split=1 - self.data_cfg.train_split,
            )

        # ── Validation/Test Generator (no augmentation) ──
        val_test_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=1 - self.data_cfg.train_split,
        )

        # ── Create Generators ──
        self.train_generator = train_datagen.flow_from_directory(
            self.data_cfg.dataset_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            seed=self.data_cfg.seed,
            shuffle=True,
        )

        self.val_generator = val_test_datagen.flow_from_directory(
            self.data_cfg.dataset_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            seed=self.data_cfg.seed,
            shuffle=False,
        )

        # For test set, we reuse the validation split but keep it separate
        # In practice, with ImageDataGenerator, val and test share the same split.
        # For a proper 3-way split, we use the full dataset loader below.
        self.test_generator = self.val_generator

        # Update class names from generator
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)

        logger.info(
            f"Data prepared | "
            f"Train samples: {self.train_generator.samples} | "
            f"Val/Test samples: {self.val_generator.samples} | "
            f"Augmentation: {use_augmentation}"
        )

        return self.train_generator, self.val_generator, self.test_generator

    # ==================================================================
    # Full dataset loading (for cross-validation and advanced analysis)
    # ==================================================================

    def load_full_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the entire dataset into memory as numpy arrays.
        Used for cross-validation and statistical analysis.

        Returns:
            Tuple of (X, y) where X is images and y is one-hot labels.
        """
        if self._X_full is not None:
            return self._X_full, self._y_full

        logger.info("Loading full dataset into memory...")
        img_size = (self.data_cfg.image_size, self.data_cfg.image_size)

        datagen = ImageDataGenerator(rescale=1.0 / 255)
        generator = datagen.flow_from_directory(
            self.data_cfg.dataset_path,
            target_size=img_size,
            batch_size=self.data_cfg.batch_size,
            class_mode="categorical",
            shuffle=False,
        )

        # Collect all batches
        steps = len(generator)
        X_list, y_list = [], []
        for i in range(steps):
            X_batch, y_batch = generator[i]
            X_list.append(X_batch)
            y_list.append(y_batch)

        self._X_full = np.concatenate(X_list, axis=0)
        self._y_full = np.concatenate(y_list, axis=0)

        # Update class names
        self.class_names = list(generator.class_indices.keys())
        self.num_classes = len(self.class_names)

        logger.info(
            f"Full dataset loaded | Shape: {self._X_full.shape} | "
            f"Labels: {self._y_full.shape}"
        )
        return self._X_full, self._y_full

    # ==================================================================
    # Augmentation layer (for tf.data pipeline)
    # ==================================================================

    def get_augmentation_layer(self) -> tf.keras.Sequential:
        """
        Create a Keras Sequential augmentation layer for use in tf.data pipelines.

        Returns:
            tf.keras.Sequential containing augmentation layers.
        """
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(self.data_cfg.rotation_range / 360.0),
            tf.keras.layers.RandomZoom(self.data_cfg.zoom_range),
            tf.keras.layers.RandomBrightness(
                factor=0.2,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=self.data_cfg.height_shift_range,
                width_factor=self.data_cfg.width_shift_range,
            ),
        ], name="data_augmentation")

    # ==================================================================
    # Dataset Statistics
    # ==================================================================

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the number of images per class.

        Returns:
            Dictionary mapping class name to image count.
        """
        distribution = {}
        for cls_name in sorted(os.listdir(self.data_cfg.dataset_path)):
            cls_path = os.path.join(self.data_cfg.dataset_path, cls_name)
            if os.path.isdir(cls_path):
                count = len([
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                distribution[cls_name] = count
        return distribution

    def get_class_weights(self) -> Dict[int, float]:
        """
        Compute class weights inversely proportional to class frequency.
        Helps handle class imbalance during training.

        Returns:
            Dictionary mapping class index to weight.
        """
        distribution = self.get_class_distribution()
        total = sum(distribution.values())
        n_classes = len(distribution)
        weights = {}
        for idx, (cls_name, count) in enumerate(sorted(distribution.items())):
            weights[idx] = total / (n_classes * count)
        return weights

    def get_dataset_summary(self) -> Dict:
        """
        Generate a comprehensive dataset summary.

        Returns:
            Dictionary with dataset statistics.
        """
        dist = self.get_class_distribution()
        counts = list(dist.values())
        return {
            "total_images": sum(counts),
            "num_classes": len(dist),
            "class_distribution": dist,
            "min_class_size": min(counts),
            "max_class_size": max(counts),
            "mean_class_size": round(np.mean(counts), 1),
            "std_class_size": round(np.std(counts), 1),
            "image_size": f"{self.data_cfg.image_size}x{self.data_cfg.image_size}",
        }
