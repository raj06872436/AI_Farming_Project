# ==============================================================================

# src/models/efficientnet.py

# Improved EfficientNetB0 transfer learning model builder.

# Optimized for PlantVillage disease classification.

# ==============================================================================

import tensorflow as tf

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EfficientNetB0Builder(BaseModelBuilder):
    """Improved EfficientNetB0 Builder."""

    def __init__(self, config: Config):
        super().__init__(config)

    # ------------------------------------------------------------------
    # BASE MODEL
    # ------------------------------------------------------------------

    def get_base_model(self) -> tf.keras.Model:
        """Load EfficientNetB0 pretrained on ImageNet."""

        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=self.data_cfg.image_shape,
            include_top=False,
            weights="imagenet"
        )

        # IMPORTANT
        # Freeze initially for stable transfer learning
        base_model.trainable = False

        logger.info(
            f"Loaded EfficientNetB0 | "
            f"Layers: {len(base_model.layers)}"
        )

        return base_model

    # ------------------------------------------------------------------
    # MODEL NAME
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "EfficientNetB0"

    # ------------------------------------------------------------------
    # BUILD FULL MODEL
    # ------------------------------------------------------------------

    def build(self, num_classes: int) -> tf.keras.Model:

        base_model = self.get_base_model()

        inputs = tf.keras.Input(
            shape=self.data_cfg.image_shape
        )

        # Undo ImageDataGenerator's rescale=1/255
        # EfficientNet's preprocess_input expects [0, 255]
        x = tf.keras.layers.Rescaling(255.0)(inputs)

        # Apply EfficientNet's proper preprocessing
        # (torch-style: ImageNet mean/std normalization)
        x = tf.keras.applications.efficientnet.preprocess_input(x)

        x = base_model(
            x,
            training=False
        )

        # Better pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # BatchNorm improves stability
        x = tf.keras.layers.BatchNormalization()(x)

        # Dense layer
        x = tf.keras.layers.Dense(
            256,
            activation="relu"
        )(x)

        # Dropout reduces overfitting
        x = tf.keras.layers.Dropout(0.4)(x)

        outputs = tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )(x)

        model = tf.keras.Model(
            inputs,
            outputs,
            name="EfficientNetB0"
        )

        return model

    # ------------------------------------------------------------------
    # COMPILE MODEL
    # ------------------------------------------------------------------

    def compile_model(
        self,
        model: tf.keras.Model,
        learning_rate: float
    ) -> tf.keras.Model:

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]
        )

        logger.info(
            f"Compiled EfficientNetB0 | "
            f"LR={learning_rate}"
        )

        return model

    # ------------------------------------------------------------------
    # FINE TUNING
    # ------------------------------------------------------------------

    def unfreeze_for_fine_tuning(
        self,
        model: tf.keras.Model
    ) -> tf.keras.Model:

        logger.info(
            "Unfreezing top EfficientNet layers "
            "for fine-tuning..."
        )

        # Find base model
        base_model = None

        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            logger.warning(
                "Base model not found!"
            )
            return model

        # Unfreeze only LAST layers
        base_model.trainable = True

        for layer in base_model.layers[:-30]:
            layer.trainable = False

        logger.info(
            "Fine-tuning last 30 layers only"
        )

        return model

    # ------------------------------------------------------------------
    # GRAD-CAM LAYER
    # ------------------------------------------------------------------

    def get_last_conv_layer_name(self) -> str:
        """Last conv layer for Grad-CAM."""
        return "top_conv"