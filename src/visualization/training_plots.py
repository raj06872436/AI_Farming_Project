# ==============================================================================
# src/visualization/training_plots.py
# Training history visualization: accuracy & loss curves.
# ==============================================================================

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config.settings import Config
from src.entity.model_entity import TrainingResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]


class TrainingPlotter:
    """Generates publication-quality training history plots."""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.paths.graphs_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_training_history(
        self,
        result: TrainingResult,
        save: bool = True,
    ) -> Optional[str]:
        """
        Plot accuracy and loss curves for a single model.

        Args:
            result: TrainingResult with history data.
            save: Whether to save the figure.

        Returns:
            Path to saved figure, or None.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(result.history.get("accuracy", [])) + 1)

        # Accuracy
        ax1.plot(epochs, result.history.get("accuracy", []),
                 "o-", color=COLORS[0], label="Train Accuracy", linewidth=2)
        ax1.plot(epochs, result.history.get("val_accuracy", []),
                 "s--", color=COLORS[1], label="Val Accuracy", linewidth=2)
        ax1.set_title(f"{result.model_name} — Accuracy vs Epoch", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(frameon=True)
        ax1.grid(True, alpha=0.3)

        # Loss
        ax2.plot(epochs, result.history.get("loss", []),
                 "o-", color=COLORS[2], label="Train Loss", linewidth=2)
        ax2.plot(epochs, result.history.get("val_loss", []),
                 "s--", color=COLORS[3], label="Val Loss", linewidth=2)
        ax2.set_title(f"{result.model_name} — Loss vs Epoch", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = None
        if save:
            filepath = os.path.join(self.output_dir, f"{result.model_name}_training_history.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"Training history plot saved: {filepath}")

        plt.close(fig)
        return filepath

    def plot_all_models_comparison(
        self,
        results: List[TrainingResult],
        save: bool = True,
    ) -> Optional[str]:
        """
        Plot accuracy and loss curves for all models on the same graph.

        Args:
            results: List of TrainingResult objects.
            save: Whether to save the figure.

        Returns:
            Path to saved figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for idx, result in enumerate(results):
            color = COLORS[idx % len(COLORS)]
            epochs = range(1, len(result.history.get("val_accuracy", [])) + 1)

            ax1.plot(epochs, result.history.get("val_accuracy", []),
                     "o-", color=color, label=result.model_name, linewidth=2, markersize=4)
            ax2.plot(epochs, result.history.get("val_loss", []),
                     "o-", color=color, label=result.model_name, linewidth=2, markersize=4)

        ax1.set_title("Validation Accuracy — All Models", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(frameon=True)
        ax1.grid(True, alpha=0.3)

        ax2.set_title("Validation Loss — All Models", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = None
        if save:
            filepath = os.path.join(self.output_dir, "all_models_training_comparison.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"All models comparison plot saved: {filepath}")

        plt.close(fig)
        return filepath
