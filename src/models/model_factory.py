# ==============================================================================
# src/models/model_factory.py
# Factory pattern for creating model builders by name.
# Single entry point for constructing any supported model architecture.
# ==============================================================================

from typing import Dict, Type

import tensorflow as tf

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.models.mobilenet import MobileNetV2Builder
from src.models.resnet import ResNet50Builder
from src.models.efficientnet import EfficientNetB0Builder
from src.models.densenet import DenseNet121Builder
from src.models.vit_model import ViTBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Registry of all available model builders
_MODEL_REGISTRY: Dict[str, Type[BaseModelBuilder]] = {
    "MobileNetV2": MobileNetV2Builder,
    "ResNet50": ResNet50Builder,
    "EfficientNetB0": EfficientNetB0Builder,
    "DenseNet121": DenseNet121Builder,
    "ViT": ViTBuilder,
}


class ModelFactory:
    """Factory for creating model builders and compiled models."""

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def get_available_models():
        """Return list of all available model architecture names."""
        return list(_MODEL_REGISTRY.keys())

    def get_builder(self, model_name: str) -> BaseModelBuilder:
        """
        Get a model builder instance by name.

        Args:
            model_name: Name of the model architecture (e.g., 'MobileNetV2').

        Returns:
            Model builder instance.

        Raises:
            ValueError: If model_name is not in the registry.
        """
        if model_name not in _MODEL_REGISTRY:
            available = ", ".join(_MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Available models: {available}"
            )
        builder = _MODEL_REGISTRY[model_name](self.config)
        logger.info(f"Created builder for {model_name}")
        return builder

    def build_model(self, model_name: str, **kwargs) -> tf.keras.Model:
        """
        Build and return a compiled model by name.

        Args:
            model_name: Name of the model architecture.
            **kwargs: Additional arguments passed to build() (e.g., num_classes).

        Returns:
            Compiled Keras model.
        """
        builder = self.get_builder(model_name)
        model = builder.build(**kwargs)
        model = builder.compile_model(model)
        return model

    def build_all_models(self, **kwargs) -> Dict[str, tf.keras.Model]:
        """
        Build all models specified in the configuration.

        Returns:
            Dictionary mapping model name to compiled model.
        """
        models = {}
        for name in self.config.model.model_names:
            try:
                models[name] = self.build_model(name, **kwargs)
                logger.info(f"Successfully built {name}")
            except Exception as e:
                logger.error(f"Failed to build {name}: {e}")
        return models

    def get_last_conv_layer_name(self, model_name: str) -> str:
        """Get the Grad-CAM target layer name for a given model."""
        builder = self.get_builder(model_name)
        return builder.get_last_conv_layer_name()
