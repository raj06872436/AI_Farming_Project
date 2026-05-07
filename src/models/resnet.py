# ==============================================================================
# src/models/resnet.py
# ResNet50 transfer learning model builder.
# Deep residual network with skip connections.
# ==============================================================================

import tensorflow as tf

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResNet50Builder(BaseModelBuilder):
    """Builder for ResNet50 transfer learning model."""

    def __init__(self, config: Config):
        super().__init__(config)

    def get_base_model(self) -> tf.keras.Model:
        """Load ResNet50 pretrained on ImageNet."""
        base = tf.keras.applications.ResNet50(
            input_shape=self.data_cfg.image_shape,
            include_top=self.model_cfg.include_top,
            weights=self.model_cfg.weights,
        )
        logger.info(f"Loaded ResNet50 | Layers: {len(base.layers)}")
        return base

    def get_model_name(self) -> str:
        return "ResNet50"

    def get_last_conv_layer_name(self) -> str:
        """Last conv layer in ResNet50 for Grad-CAM."""
        return "conv5_block3_out"
