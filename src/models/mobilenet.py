# ==============================================================================
# src/models/mobilenet.py
# MobileNetV2 transfer learning model builder.
# Lightweight architecture optimized for mobile/edge deployment.
# ==============================================================================

import tensorflow as tf

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MobileNetV2Builder(BaseModelBuilder):
    """Builder for MobileNetV2 transfer learning model."""

    def __init__(self, config: Config):
        super().__init__(config)

    def get_base_model(self) -> tf.keras.Model:
        """Load MobileNetV2 pretrained on ImageNet."""
        base = tf.keras.applications.MobileNetV2(
            input_shape=self.data_cfg.image_shape,
            include_top=self.model_cfg.include_top,
            weights=self.model_cfg.weights,
        )
        logger.info(f"Loaded MobileNetV2 | Layers: {len(base.layers)}")
        return base

    def get_model_name(self) -> str:
        return "MobileNetV2"

    def get_last_conv_layer_name(self) -> str:
        """Last conv layer in MobileNetV2 for Grad-CAM."""
        return "out_relu"
