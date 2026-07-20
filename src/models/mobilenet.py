# ==============================================================================
# src/models/mobilenet.py
# Improved MobileNetV2 transfer learning model builder.
# Optimized for PlantVillage disease classification.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MobileNetV2Builder(BaseModelBuilder):
    """
    Improved builder for MobileNetV2 transfer learning model.

    Key fixes over the base builder:
    - Proper tf-style preprocessing (MobileNetV2 expects [-1, 1] range)
    - Custom build() with Rescaling + preprocess_input for correct input normalization
    - Model-specific fine-tuning (unfreeze last 50 layers)
    """

    def __init__(self, config: Config):
        super().__init__(config)

    # ------------------------------------------------------------------
    # BASE MODEL
    # ------------------------------------------------------------------

    def get_base_model(self) -> tf.keras.Model:
        """Load MobileNetV2 pretrained on ImageNet."""
        base = tf.keras.applications.MobileNetV2(
            input_shape=self.data_cfg.image_shape,
            include_top=False,
            weights="imagenet",
        )
        # Freeze initially for stable transfer learning
        base.trainable = False
        logger.info(f"Loaded MobileNetV2 | Layers: {len(base.layers)}")
        return base

    def get_model_name(self) -> str:
        return "MobileNetV2"

    # ------------------------------------------------------------------
    # BUILD FULL MODEL (custom — fixes preprocessing mismatch)
    # ------------------------------------------------------------------

    def build(self, num_classes: int = None, **kwargs) -> tf.keras.Model:
        """
        Build MobileNetV2 with proper tf-style preprocessing.

        The data pipeline uses rescale=1/255 (outputs [0,1]).
        MobileNetV2's preprocess_input expects [0,255] and applies
        tf-style normalization: scales to [-1, 1].

        We undo the 1/255 rescaling, then apply the native preprocessing.
        """
        num_classes = num_classes or self.data_cfg.num_classes

        base_model = self.get_base_model()

        inputs = tf.keras.Input(shape=self.data_cfg.image_shape)

        # Undo ImageDataGenerator's rescale=1/255
        # MobileNetV2's preprocess_input expects [0, 255]
        x = layers.Rescaling(255.0)(inputs)

        # Apply MobileNetV2's proper tf-style preprocessing
        # (scales pixels to [-1, 1])
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

        x = base_model(x, training=False)

        # Classifier head
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="batch_norm")(x)
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.Dense(128, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = tf.keras.Model(inputs, outputs, name="MobileNetV2")

        total_params = model.count_params()
        trainable_params = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        logger.info(
            f"Built MobileNetV2 (custom) | "
            f"Total params: {total_params:,} | "
            f"Trainable: {trainable_params:,}"
        )

        return model

    # ------------------------------------------------------------------
    # COMPILE MODEL
    # ------------------------------------------------------------------

    def compile_model(
        self,
        model: tf.keras.Model,
        learning_rate: float = None,
        optimizer_name: str = None,
    ) -> tf.keras.Model:
        """Compile with precision/recall metrics for better monitoring."""
        lr = learning_rate or self.training_cfg.learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        logger.info(f"Compiled MobileNetV2 | LR={lr}")
        return model

    # ------------------------------------------------------------------
    # FINE TUNING
    # ------------------------------------------------------------------

    def unfreeze_for_fine_tuning(
        self,
        model: tf.keras.Model,
        num_layers_to_unfreeze: int = None,
    ) -> tf.keras.Model:
        """
        Unfreeze the top layers of MobileNetV2 for fine-tuning.

        MobileNetV2 has 155 layers. Unfreezing the last 50 covers
        the final inverted residual blocks for domain adaptation.
        """
        n = num_layers_to_unfreeze or 50

        logger.info(f"Unfreezing top {n} layers of MobileNetV2 for fine-tuning...")

        # Find the base model (MobileNetV2 backbone) within the functional model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            logger.warning("MobileNetV2 base model not found! Falling back to full model unfreezing.")
            model.trainable = True
            for layer in model.layers[:-n]:
                layer.trainable = False
        else:
            base_model.trainable = True
            for layer in base_model.layers[:-n]:
                layer.trainable = False

        trainable_count = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        logger.info(
            f"Fine-tuning MobileNetV2 | "
            f"Unfrozen top {n} layers | "
            f"Trainable params: {trainable_count:,}"
        )
        return model

    # ------------------------------------------------------------------
    # GRAD-CAM LAYER
    # ------------------------------------------------------------------

    def get_last_conv_layer_name(self) -> str:
        """Last conv layer in MobileNetV2 for Grad-CAM."""
        return "out_relu"
