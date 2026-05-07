# ==============================================================================
# src/models/efficientnet.py
# EfficientNetB0 transfer learning model builder.
# Compound-scaled CNN architecture for optimal accuracy/efficiency trade-off.
# ==============================================================================

import tensorflow as tf

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EfficientNetB0Builder(BaseModelBuilder):
    """Builder for EfficientNetB0 transfer learning model."""

    def __init__(self, config: Config):
        super().__init__(config)

    def get_base_model(self) -> tf.keras.Model:
        """Load EfficientNetB0 pretrained on ImageNet."""
        base = tf.keras.applications.EfficientNetB0(
            input_shape=self.data_cfg.image_shape,
            include_top=self.model_cfg.include_top,
            weights=self.model_cfg.weights,
        )
        logger.info(f"Loaded EfficientNetB0 | Layers: {len(base.layers)}")
        return base

    def get_model_name(self) -> str:
        return "EfficientNetB0"

    def get_last_conv_layer_name(self) -> str:
        """Last conv layer in EfficientNetB0 for Grad-CAM."""
        return "top_conv"
