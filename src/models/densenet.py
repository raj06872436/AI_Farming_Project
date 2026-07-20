# ==============================================================================
# src/models/densenet.py
# Improved DenseNet121 transfer learning model builder.
# Optimized for PlantVillage disease classification.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseNet121Builder(BaseModelBuilder):
    """
    Improved builder for DenseNet121 transfer learning model.

    Key fixes over the base builder:
    - Proper torch-style preprocessing (DenseNet expects ImageNet mean/std normalization)
    - Custom build() with Rescaling + preprocess_input for correct input normalization
    - Model-specific fine-tuning (unfreeze last 40 layers covering dense block 4)
    """

    def __init__(self, config: Config):
        super().__init__(config)

    # ------------------------------------------------------------------
    # BASE MODEL
    # ------------------------------------------------------------------

    def get_base_model(self) -> tf.keras.Model:
        """Load DenseNet121 pretrained on ImageNet."""
        base = tf.keras.applications.DenseNet121(
            input_shape=self.data_cfg.image_shape,
            include_top=False,
            weights="imagenet",
        )
        # Freeze initially for stable transfer learning
        base.trainable = False
        logger.info(f"Loaded DenseNet121 | Layers: {len(base.layers)}")
        return base

    def get_model_name(self) -> str:
        return "DenseNet121"

    # ------------------------------------------------------------------
    # BUILD FULL MODEL (custom — fixes preprocessing mismatch)
    # ------------------------------------------------------------------

    def build(self, num_classes: int = None, **kwargs) -> tf.keras.Model:
        """
        Build DenseNet121 with proper torch-style preprocessing.

        The data pipeline uses rescale=1/255 (outputs [0,1]).
        DenseNet121's preprocess_input expects [0,255] and applies
        torch-style normalization: ImageNet mean/std per channel.

        We undo the 1/255 rescaling, then apply the native preprocessing.
        """
        num_classes = num_classes or self.data_cfg.num_classes

        base_model = self.get_base_model()

        inputs = tf.keras.Input(shape=self.data_cfg.image_shape)

        # Undo ImageDataGenerator's rescale=1/255
        # DenseNet121's preprocess_input expects [0, 255]
        x = layers.Rescaling(255.0)(inputs)

        # Apply DenseNet121's proper torch-style preprocessing
        # (ImageNet mean/std normalization per channel)
        x = tf.keras.applications.densenet.preprocess_input(x)

        x = base_model(x, training=False)

        # Classifier head
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="batch_norm")(x)
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.Dense(128, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = tf.keras.Model(inputs, outputs, name="DenseNet121")

        total_params = model.count_params()
        trainable_params = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        logger.info(
            f"Built DenseNet121 (custom) | "
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
        logger.info(f"Compiled DenseNet121 | LR={lr}")
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
        Unfreeze the top layers of DenseNet121 for fine-tuning.

        DenseNet121 has 427 layers. Unfreezing the last 40 covers
        the final dense block for domain adaptation.
        """
        n = num_layers_to_unfreeze or 40

        logger.info(f"Unfreezing top {n} layers of DenseNet121 for fine-tuning...")

        # Find the base model (DenseNet121 backbone) within the functional model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            logger.warning("DenseNet121 base model not found! Falling back to full model unfreezing.")
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
            f"Fine-tuning DenseNet121 | "
            f"Unfrozen top {n} layers | "
            f"Trainable params: {trainable_count:,}"
        )
        return model

    # ------------------------------------------------------------------
    # GRAD-CAM LAYER
    # ------------------------------------------------------------------

    def get_last_conv_layer_name(self) -> str:
        """Last conv layer in DenseNet121 for Grad-CAM."""
        return "relu"
