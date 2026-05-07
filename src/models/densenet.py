# ==============================================================================
# src/models/densenet.py
# DenseNet121 transfer learning model builder.
# Dense connectivity pattern for feature reuse and gradient flow.
# ==============================================================================

import tensorflow as tf

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DenseNet121Builder(BaseModelBuilder):
    """Builder for DenseNet121 transfer learning model."""

    def __init__(self, config: Config):
        super().__init__(config)

    def get_base_model(self) -> tf.keras.Model:
        """Load DenseNet121 pretrained on ImageNet."""
        base = tf.keras.applications.DenseNet121(
            input_shape=self.data_cfg.image_shape,
            include_top=self.model_cfg.include_top,
            weights=self.model_cfg.weights,
        )
        logger.info(f"Loaded DenseNet121 | Layers: {len(base.layers)}")
        return base

    def get_model_name(self) -> str:
        return "DenseNet121"

    def get_last_conv_layer_name(self) -> str:
        """Last conv layer in DenseNet121 for Grad-CAM."""
        return "relu"
