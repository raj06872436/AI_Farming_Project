# ==============================================================================
# src/visualization/gradcam.py
# Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
# Generates visual explanations showing which regions drive predictions.
# ==============================================================================

import os
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.config.settings import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAMVisualizer:
    """
    Generates Grad-CAM heatmaps for model predictions.
    Shows which image regions the model focuses on for disease detection.
    """

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.paths.gradcam_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_gradcam(
        self,
        model: tf.keras.Model,
        image_array: np.ndarray,
        last_conv_layer_name: str,
        pred_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a given image and model.

        Args:
            model: Trained Keras model.
            image_array: Preprocessed image array of shape (1, H, W, 3).
            last_conv_layer_name: Name of the last convolutional layer.
            pred_index: Target class index. If None, uses the predicted class.

        Returns:
            Heatmap array of shape (H, W) with values in [0, 1].
        """
        # Find the target layer in the model
        last_conv_layer = None
        for layer in model.layers:
            if layer.name == last_conv_layer_name:
                last_conv_layer = layer
                break
            # Also check within nested models
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    if sub_layer.name == last_conv_layer_name:
                        last_conv_layer = sub_layer
                        break

        if last_conv_layer is None:
            logger.warning(
                f"Could not find layer '{last_conv_layer_name}'. "
                f"Attempting to find last Conv2D layer..."
            )
            # Fallback: find the last Conv2D layer
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
                if hasattr(layer, 'layers'):
                    for sub_layer in reversed(layer.layers):
                        if isinstance(sub_layer, tf.keras.layers.Conv2D):
                            last_conv_layer = sub_layer
                            break

        if last_conv_layer is None:
            logger.error("No convolutional layer found for Grad-CAM")
            return np.zeros((self.config.data.image_size, self.config.data.image_size))

        # Create a model that outputs the conv layer output and final predictions
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output],
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_output = predictions[:, pred_index]

        # Gradient of the predicted class with respect to conv layer output
        grads = tape.gradient(class_output, conv_outputs)

        if grads is None:
            logger.warning("Gradients are None — model may not be differentiable through this path")
            return np.zeros((self.config.data.image_size, self.config.data.image_size))

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv outputs by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize to image dimensions
        heatmap = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap).resize(
            (self.config.data.image_size, self.config.data.image_size),
            Image.BILINEAR,
        )
        heatmap = np.array(heatmap_img).astype(np.float32) / 255.0

        return heatmap

    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            original_image: Original image array (H, W, 3) in [0, 1].
            heatmap: Heatmap array (H, W) in [0, 1].
            alpha: Overlay transparency.

        Returns:
            Overlayed image array (H, W, 3) in [0, 1].
        """
        # Apply colormap to heatmap
        colormap = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel

        # Overlay
        overlayed = (1 - alpha) * original_image + alpha * colormap
        overlayed = np.clip(overlayed, 0, 1)

        return overlayed

    def get_activation_percentage(self, heatmap: np.ndarray, threshold: float = 0.3) -> float:
        """
        Compute the percentage of image area with significant activation.
        Used for disease severity estimation.

        Args:
            heatmap: Heatmap array (H, W) in [0, 1].
            threshold: Activation threshold.

        Returns:
            Percentage of area above threshold (0-100).
        """
        total_pixels = heatmap.size
        active_pixels = np.sum(heatmap > threshold)
        return (active_pixels / total_pixels) * 100

    def visualize_and_save(
        self,
        model: tf.keras.Model,
        image_array: np.ndarray,
        last_conv_layer_name: str,
        class_name: str,
        model_name: str,
        confidence: float,
        filename: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Generate and save Grad-CAM visualization.

        Args:
            model: Trained model.
            image_array: Preprocessed image (1, H, W, 3).
            last_conv_layer_name: Target conv layer name.
            class_name: Predicted class name.
            model_name: Model architecture name.
            confidence: Prediction confidence.
            filename: Custom output filename.

        Returns:
            Tuple of (filepath, activation_percentage).
        """
        heatmap = self.generate_gradcam(model, image_array, last_conv_layer_name)
        activation_pct = self.get_activation_percentage(heatmap)

        original = image_array[0]  # Remove batch dimension
        overlayed = self.overlay_heatmap(original, heatmap)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(overlayed)
        axes[2].set_title("Overlay", fontsize=11)
        axes[2].axis("off")

        display_cls = class_name.replace("_", " ")
        fig.suptitle(
            f"{model_name} | Prediction: {display_cls} | "
            f"Confidence: {confidence:.1%} | Active Area: {activation_pct:.1f}%",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

        if filename is None:
            safe_cls = class_name.replace(" ", "_")[:30]
            filename = f"{model_name}_{safe_cls}_gradcam.png"

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Grad-CAM saved: {filepath} | Activation: {activation_pct:.1f}%")
        return filepath, activation_pct

    def generate_gradcam_for_prediction(
        self,
        model: tf.keras.Model,
        image_array: np.ndarray,
        last_conv_layer_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate Grad-CAM data for Streamlit display (no file saving).

        Returns:
            Tuple of (heatmap, overlayed_image, activation_percentage).
        """
        heatmap = self.generate_gradcam(model, image_array, last_conv_layer_name)
        activation_pct = self.get_activation_percentage(heatmap)
        original = image_array[0]
        overlayed = self.overlay_heatmap(original, heatmap)
        return heatmap, overlayed, activation_pct
