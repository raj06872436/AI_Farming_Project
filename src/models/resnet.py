# ==============================================================================
# src/models/resnet.py
# Improved ResNet50 transfer learning model builder.
# Optimized for PlantVillage disease classification.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResNet50Builder(BaseModelBuilder):
    """
    Improved builder for ResNet50 transfer learning model.

    Key fixes over the base builder:
    - Proper caffe-style preprocessing (ResNet expects BGR, ImageNet mean subtracted)
    - Larger classifier head (512 → 256) to handle ResNet's 2048-dim feature output
    - Model-specific fine-tuning that unfreezes full conv5 block (80 layers)
    - Lower dropout for the larger capacity network
    """

    def __init__(self, config: Config):
        super().__init__(config)

    # ------------------------------------------------------------------
    # BASE MODEL
    # ------------------------------------------------------------------

    def get_base_model(self) -> tf.keras.Model:
        """Load ResNet50 pretrained on ImageNet."""
        base = tf.keras.applications.ResNet50(
            input_shape=self.data_cfg.image_shape,
            include_top=False,
            weights="imagenet",
        )
        # Freeze initially for stable transfer learning
        base.trainable = False
        logger.info(f"Loaded ResNet50 | Layers: {len(base.layers)}")
        return base

    def get_model_name(self) -> str:
        return "ResNet50"

    # ------------------------------------------------------------------
    # BUILD FULL MODEL (custom — fixes preprocessing mismatch)
    # ------------------------------------------------------------------

    def build(self, num_classes: int = None, **kwargs) -> tf.keras.Model:
        """
        Build ResNet50 with proper caffe-style preprocessing.

        The data pipeline uses rescale=1/255 (outputs [0,1]).
        ResNet50's preprocess_input expects [0,255] and applies:
          - Scale to [0,255]
          - Convert RGB → BGR
          - Subtract ImageNet channel means [103.939, 116.779, 123.68]

        We undo the 1/255 rescaling, then apply the native preprocessing.
        This matches what EfficientNetB0Builder already does for torch-style.
        """
        num_classes = num_classes or self.data_cfg.num_classes

        base_model = self.get_base_model()

        inputs = tf.keras.Input(shape=self.data_cfg.image_shape)

        # Undo ImageDataGenerator's rescale=1/255
        # ResNet50's preprocess_input expects [0, 255]
        x = layers.Rescaling(255.0)(inputs)

        # Apply ResNet50's proper caffe-style preprocessing
        # (RGB → BGR, subtract ImageNet means)
        x = tf.keras.applications.resnet50.preprocess_input(x)

        x = base_model(x, training=False)

        # Classifier head — larger than default to match ResNet's 2048-dim output
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="batch_norm")(x)
        x = layers.Dense(512, activation="relu", name="dense_1")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.Dense(256, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = tf.keras.Model(inputs, outputs, name="ResNet50")

        total_params = model.count_params()
        trainable_params = sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        )
        logger.info(
            f"Built ResNet50 (custom) | "
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
        logger.info(f"Compiled ResNet50 | LR={lr}")
        return model

    # ------------------------------------------------------------------
    # FINE TUNING — unfreeze conv5 block (deeper than default 50 layers)
    # ------------------------------------------------------------------

    def unfreeze_for_fine_tuning(
        self,
        model: tf.keras.Model,
        num_layers_to_unfreeze: int = None,
    ) -> tf.keras.Model:
        """
        Unfreeze the top layers of ResNet50 for fine-tuning.

        ResNet50 has 175 layers. We unfreeze the last 80 layers which covers
        the entire conv5_block plus part of conv4_block, giving the model
        enough capacity to adapt its high-level features to plant diseases.
        """
        n = num_layers_to_unfreeze or 80  # conv5 + partial conv4

        logger.info(f"Unfreezing top {n} layers of ResNet50 for fine-tuning...")

        # Find the base model (ResNet50 backbone) within the functional model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            logger.warning("ResNet50 base model not found! Falling back to full model unfreezing.")
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
            f"Fine-tuning ResNet50 | "
            f"Unfrozen top {n} layers | "
            f"Trainable params: {trainable_count:,}"
        )
        return model

    # ------------------------------------------------------------------
    # GRAD-CAM LAYER
    # ------------------------------------------------------------------

    def get_last_conv_layer_name(self) -> str:
        """Last conv layer in ResNet50 for Grad-CAM."""
        return "conv5_block3_out"
