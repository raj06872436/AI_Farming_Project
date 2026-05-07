# ==============================================================================
# src/visualization/evaluation_plots.py
# Evaluation visualization: confusion matrices, ROC curves, PR curves,
# and model comparison charts.
# ==============================================================================

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.config.settings import Config
from src.entity.model_entity import ModelMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]


class EvaluationPlotter:
    """Generates all evaluation visualization plots."""

    def __init__(self, config: Config):
        self.config = config
        self.class_names = config.data.class_names

    # ── Confusion Matrix ──────────────────────────────────────────────

    def plot_confusion_matrix(
        self, metrics: ModelMetrics, save: bool = True,
    ) -> Optional[str]:
        """Plot confusion matrix heatmap for a single model."""
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            metrics.confusion_matrix,
            annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix — {metrics.model_name}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        filepath = None
        if save:
            filepath = os.path.join(
                self.config.paths.confusion_matrix_dir,
                f"{metrics.model_name}_confusion_matrix.png"
            )
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved: {filepath}")
        plt.close(fig)
        return filepath

    # ── ROC Curves ────────────────────────────────────────────────────

    def plot_roc_curve(
        self, metrics: ModelMetrics, save: bool = True,
    ) -> Optional[str]:
        """Plot ROC curves (macro-average + selected classes)."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot macro-average
        if "macro" in metrics.fpr:
            ax.plot(
                metrics.fpr["macro"], metrics.tpr["macro"],
                color="#e74c3c", linewidth=3, linestyle="--",
                label=f'Macro-average (AUC = {metrics.roc_auc["macro"]:.3f})',
            )

        # Plot a few individual classes
        n_to_show = min(5, self.config.data.num_classes)
        for i in range(n_to_show):
            if i in metrics.fpr:
                ax.plot(
                    metrics.fpr[i], metrics.tpr[i],
                    color=COLORS[i % len(COLORS)], linewidth=1.5, alpha=0.7,
                    label=f'{self.class_names[i]} (AUC = {metrics.roc_auc.get(i, 0):.3f})',
                )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_title(f"ROC Curves — {metrics.model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = None
        if save:
            filepath = os.path.join(
                self.config.paths.roc_curves_dir,
                f"{metrics.model_name}_roc_curves.png"
            )
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curves saved: {filepath}")
        plt.close(fig)
        return filepath

    # ── Precision-Recall Curves ───────────────────────────────────────

    def plot_pr_curve(
        self, metrics: ModelMetrics, save: bool = True,
    ) -> Optional[str]:
        """Plot Precision-Recall curves for selected classes."""
        fig, ax = plt.subplots(figsize=(10, 8))

        n_to_show = min(5, self.config.data.num_classes)
        for i in range(n_to_show):
            if i in metrics.pr_precision:
                ax.plot(
                    metrics.pr_recall[i], metrics.pr_precision[i],
                    color=COLORS[i % len(COLORS)], linewidth=1.5,
                    label=f'{self.class_names[i]} (AP = {metrics.pr_auc.get(i, 0):.3f})',
                )

        ax.set_title(f"Precision-Recall Curves — {metrics.model_name}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = None
        if save:
            filepath = os.path.join(
                self.config.paths.roc_curves_dir,
                f"{metrics.model_name}_pr_curves.png"
            )
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"PR curves saved: {filepath}")
        plt.close(fig)
        return filepath

    # ── Model Comparison Bar Charts ───────────────────────────────────

    def plot_model_comparison(
        self, all_metrics: List[ModelMetrics], save: bool = True,
    ) -> Optional[str]:
        """Plot bar chart comparing key metrics across all models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        names = [m.model_name for m in all_metrics]
        colors = COLORS[:len(names)]

        metrics_data = {
            "Accuracy": [m.accuracy for m in all_metrics],
            "F1 Score": [m.f1_score for m in all_metrics],
            "AUC": [m.auc_score for m in all_metrics],
            "Precision": [m.precision for m in all_metrics],
        }

        for ax, (metric_name, values) in zip(axes.flatten(), metrics_data.items()):
            bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5)
            ax.set_title(metric_name, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
            ax.tick_params(axis="x", rotation=15)

        plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        filepath = None
        if save:
            filepath = os.path.join(
                self.config.paths.graphs_dir, "model_comparison_metrics.png"
            )
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"Model comparison chart saved: {filepath}")
        plt.close(fig)
        return filepath

    # ── Inference Time Comparison ─────────────────────────────────────

    def plot_inference_comparison(
        self, all_metrics: List[ModelMetrics], save: bool = True,
    ) -> Optional[str]:
        """Plot inference time comparison bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        names = [m.model_name for m in all_metrics]
        times = [m.inference_time_ms for m in all_metrics]
        colors = COLORS[:len(names)]

        bars = ax.barh(names, times, color=colors, edgecolor="white", height=0.5)
        ax.set_title("Inference Time Comparison", fontsize=13, fontweight="bold")
        ax.set_xlabel("Inference Time (ms)")

        for bar, val in zip(bars, times):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}ms", va="center", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        filepath = None
        if save:
            filepath = os.path.join(self.config.paths.graphs_dir, "inference_time_comparison.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"Inference time chart saved: {filepath}")
        plt.close(fig)
        return filepath

    # ── Model Size Comparison ─────────────────────────────────────────

    def plot_size_comparison(
        self, all_metrics: List[ModelMetrics], save: bool = True,
    ) -> Optional[str]:
        """Plot model size comparison bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        names = [m.model_name for m in all_metrics]
        sizes = [m.model_size_mb for m in all_metrics]
        colors = COLORS[:len(names)]

        bars = ax.barh(names, sizes, color=colors, edgecolor="white", height=0.5)
        ax.set_title("Model Size Comparison", fontsize=13, fontweight="bold")
        ax.set_xlabel("Model Size (MB)")

        for bar, val in zip(bars, sizes):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}MB", va="center", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        filepath = None
        if save:
            filepath = os.path.join(self.config.paths.graphs_dir, "model_size_comparison.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"Model size chart saved: {filepath}")
        plt.close(fig)
        return filepath

    # ── Generate All Plots ────────────────────────────────────────────

    def generate_all_plots(self, all_metrics: List[ModelMetrics]) -> None:
        """Generate all evaluation plots for all models."""
        logger.info("Generating all evaluation plots...")

        for m in all_metrics:
            self.plot_confusion_matrix(m)
            self.plot_roc_curve(m)
            self.plot_pr_curve(m)

        self.plot_model_comparison(all_metrics)
        self.plot_inference_comparison(all_metrics)
        self.plot_size_comparison(all_metrics)

        logger.info("All evaluation plots generated successfully")
