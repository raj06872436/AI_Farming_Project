# ==============================================================================
# src/models/base_model.py
# Abstract base class for all transfer learning model builders.
# Provides a common interface for building models with consistent classifier heads.
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config.settings import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModelBuilder(ABC):
    """
    Abstract base class for transfer learning model construction.
    All model builders inherit from this class to ensure a consistent
    architecture pattern across different backbone networks.
    """

    def __init__(self, config: Config):
        """
        Initialize the builder with project configuration.

        Args:
            config: Project configuration object.
        """
        self.config = config
        self.data_cfg = config.data
        self.training_cfg = config.training
        self.model_cfg = config.model

    @abstractmethod
    def get_base_model(self) -> tf.keras.Model:
        """
        Load and return the pretrained base model (backbone).
        Must be implemented by each subclass.

        Returns:
            Pretrained Keras model without classification head.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the human-readable name of this model architecture."""
        pass

    @abstractmethod
    def get_last_conv_layer_name(self) -> str:
        """
        Return the name of the last convolutional layer in the base model.
        Used for Grad-CAM visualization.
        """
        pass

    def build(
        self,
        num_classes: Optional[int] = None,
        dense_units_1: Optional[int] = None,
        dense_units_2: Optional[int] = None,
        dropout_rate: Optional[float] = None,
    ) -> tf.keras.Model:
        """
        Build the complete model: pretrained backbone + custom classifier head.

        The classifier head consists of:
        1. GlobalAveragePooling2D
        2. BatchNormalization
        3. Dense(dense_units_1) + ReLU
        4. Dropout
        5. Dense(dense_units_2) + ReLU
        6. Dropout
        7. Dense(num_classes) + Softmax

        Args:
            num_classes: Number of output classes. Defaults to config value.
            dense_units_1: First dense layer units. Defaults to config value.
            dense_units_2: Second dense layer units. Defaults to config value.
            dropout_rate: Dropout rate. Defaults to config value.

        Returns:
            Compiled Keras Model ready for training.
        """
        num_classes = num_classes or self.data_cfg.num_classes
        dense_units_1 = dense_units_1 or self.training_cfg.dense_units_1
        dense_units_2 = dense_units_2 or self.training_cfg.dense_units_2
        dropout_rate = dropout_rate or self.training_cfg.dropout_rate

        # Load pretrained backbone
        base_model = self.get_base_model()
        base_model.trainable = False  # Freeze for Phase 1 training

        # Build classifier head
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="batch_norm")(x)
        x = layers.Dense(dense_units_1, activation="relu", name="dense_1")(x)
        x = layers.Dropout(dropout_rate, name="dropout_1")(x)
        x = layers.Dense(dense_units_2, activation="relu", name="dense_2")(x)
        x = layers.Dropout(dropout_rate, name="dropout_2")(x)
        output = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = models.Model(
            inputs=base_model.input,
            outputs=output,
            name=self.get_model_name(),
        )

        logger.info(
            f"Built {self.get_model_name()} | "
            f"Total params: {model.count_params():,} | "
            f"Trainable: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}"
        )

        return model

    def compile_model(
        self,
        model: tf.keras.Model,
        learning_rate: Optional[float] = None,
        optimizer_name: Optional[str] = None,
    ) -> tf.keras.Model:
        """
        Compile the model with specified optimizer and loss.

        Args:
            model: Keras model to compile.
            learning_rate: Learning rate. Defaults to config value.
            optimizer_name: Optimizer name. Defaults to config value.

        Returns:
            Compiled model.
        """
        lr = learning_rate or self.training_cfg.learning_rate
        opt_name = optimizer_name or self.training_cfg.optimizer

        optimizer = self._get_optimizer(opt_name, lr)

        model.compile(
            optimizer=optimizer,
            loss=self.training_cfg.loss,
            metrics=["accuracy"],
        )

        logger.info(
            f"Compiled {model.name} | "
            f"Optimizer: {opt_name} | "
            f"LR: {lr} | "
            f"Loss: {self.training_cfg.loss}"
        )
        return model

    def unfreeze_for_fine_tuning(
        self,
        model: tf.keras.Model,
        num_layers_to_unfreeze: Optional[int] = None,
    ) -> tf.keras.Model:
        """
        Unfreeze the top N layers of the base model for fine-tuning.

        Args:
            model: The model to unfreeze layers in.
            num_layers_to_unfreeze: Number of layers to unfreeze from the top.
                Defaults to config value.

        Returns:
            Model with unfrozen layers.
        """
        n = num_layers_to_unfreeze or self.training_cfg.fine_tune_layers

        # Make base model trainable
        base = model.layers[0] if hasattr(model.layers[0], 'layers') else None

        # Find the base model within the full model
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base = layer
                break

        if base is None:
            # Fallback: unfreeze top N layers of the full model
            model.trainable = True
            for layer in model.layers[:-n]:
                layer.trainable = False
        else:
            base.trainable = True
            for layer in base.layers[:-n]:
                layer.trainable = False

        trainable_count = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        logger.info(
            f"Fine-tuning {model.name} | "
            f"Unfrozen top {n} layers | "
            f"Trainable params: {trainable_count:,}"
        )
        return model

    def get_callbacks(self, model_name: str) -> list:
        """
        Create training callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

        Args:
            model_name: Name used for checkpoint file naming.

        Returns:
            List of Keras callbacks.
        """
        checkpoint_path = os.path.join(
            self.config.paths.saved_models_dir,
            f"{model_name}_best.keras"
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.training_cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.training_cfg.reduce_lr_factor,
                patience=self.training_cfg.reduce_lr_patience,
                min_lr=self.training_cfg.min_lr,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]
        return callbacks

    @staticmethod
    def _get_optimizer(name: str, lr: float):
        """Get a Keras optimizer by name."""
        optimizers = {
            "adam": tf.keras.optimizers.Adam(learning_rate=lr),
            "sgd": tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=lr),
        }
        return optimizers.get(name.lower(), tf.keras.optimizers.Adam(learning_rate=lr))


# Need os import for checkpoint path
import os
