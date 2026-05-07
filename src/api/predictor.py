# ==============================================================================
# src/api/predictor.py
# Prediction API: load model, preprocess image, predict, explain, recommend.
# ==============================================================================

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config.settings import Config
from src.models.model_factory import ModelFactory
from src.visualization.gradcam import GradCAMVisualizer
from src.utils.recommendations import get_recommendation, estimate_severity, Recommendation
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PlantDiseasePredictor:
    """
    High-level prediction API for plant disease detection.
    Encapsulates model loading, preprocessing, prediction,
    Grad-CAM generation, severity estimation, and recommendation.
    """

    def __init__(self, config: Config):
        self.config = config
        self.factory = ModelFactory(config)
        self.gradcam_viz = GradCAMVisualizer(config)
        self.class_names = config.data.class_names
        self.image_size = config.data.image_size

        # Cache for loaded models
        self._models: Dict[str, tf.keras.Model] = {}

        # Layer name mapping for Grad-CAM
        self._gradcam_layers = {
            "MobileNetV2": "out_relu",
            "ResNet50": "conv5_block3_out",
            "EfficientNetB0": "top_conv",
            "DenseNet121": "relu",
            "ViT": "patch_embedding",
        }

    def load_model(self, model_name: str) -> tf.keras.Model:
        """
        Load a saved model by name. Caches loaded models for reuse.

        Args:
            model_name: Model architecture name.

        Returns:
            Loaded Keras model.
        """
        if model_name in self._models:
            return self._models[model_name]

        # Try .keras format first, then .h5
        for ext in [".keras", ".h5"]:
            model_path = os.path.join(
                self.config.paths.saved_models_dir, f"{model_name}_final{ext}"
            )
            if os.path.exists(model_path):
                logger.info(f"Loading model: {model_path}")
                model = tf.keras.models.load_model(model_path)
                self._models[model_name] = model
                return model

        # Also try best checkpoint
        model_path = os.path.join(
            self.config.paths.saved_models_dir, f"{model_name}_best.keras"
        )
        if os.path.exists(model_path):
            logger.info(f"Loading model: {model_path}")
            model = tf.keras.models.load_model(model_path)
            self._models[model_name] = model
            return model

        raise FileNotFoundError(
            f"No saved model found for {model_name} in {self.config.paths.saved_models_dir}"
        )

    def get_available_models(self) -> List[str]:
        """Return list of model names that have saved checkpoints."""
        available = []
        for name in self.factory.get_available_models():
            for ext in [".keras", ".h5"]:
                path = os.path.join(
                    self.config.paths.saved_models_dir, f"{name}_final{ext}"
                )
                if os.path.exists(path):
                    available.append(name)
                    break
            else:
                path = os.path.join(
                    self.config.paths.saved_models_dir, f"{name}_best.keras"
                )
                if os.path.exists(path):
                    available.append(name)
        return available

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for model prediction.

        Args:
            image: PIL Image object.

        Returns:
            Preprocessed numpy array of shape (1, H, W, 3).
        """
        img = image.convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(
        self,
        image: Image.Image,
        model_name: str,
        top_k: int = 3,
    ) -> Dict:
        """
        Make a prediction on an image using the specified model.

        Args:
            image: PIL Image object.
            model_name: Model architecture name.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with prediction details:
            - predicted_class, confidence, top_k_predictions,
            - severity, recommendations
        """
        model = self.load_model(model_name)
        img_array = self.preprocess_image(image)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        pred_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_class = self.class_names[pred_idx]

        # Top-K predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        top_predictions = [
            {
                "class": self.class_names[idx],
                "confidence": float(predictions[0][idx]),
                "display_name": self.class_names[idx].replace("_", " ").replace("__", " — "),
            }
            for idx in top_indices
        ]

        # Grad-CAM
        gradcam_layer = self._gradcam_layers.get(model_name, "")
        heatmap, overlayed, activation_pct = self.gradcam_viz.generate_gradcam_for_prediction(
            model, img_array, gradcam_layer
        )

        # Severity
        is_healthy = "healthy" in predicted_class.lower()
        severity_info = None
        if not is_healthy:
            severity_info = estimate_severity(confidence, activation_pct)

        # Recommendation
        recommendation = get_recommendation(predicted_class, confidence, activation_pct)

        return {
            "predicted_class": predicted_class,
            "display_name": predicted_class.replace("_", " ").replace("__", " — "),
            "confidence": confidence,
            "top_k_predictions": top_predictions,
            "gradcam_heatmap": heatmap,
            "gradcam_overlay": overlayed,
            "activation_percentage": activation_pct,
            "severity": severity_info,
            "recommendation": recommendation,
            "model_name": model_name,
            "is_healthy": is_healthy,
        }
