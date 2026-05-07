# ==============================================================================
# src/evaluation/compression_study.py
# Model compression analysis: size, speed, accuracy tradeoffs.
# ==============================================================================

import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from src.config.settings import Config
from src.entity.model_entity import ModelMetrics
from src.utils.logger import get_logger
from src.utils.helpers import save_csv

logger = get_logger(__name__)


class CompressionStudy:
    """
    Analyzes model compression characteristics:
    - Model size comparison
    - Inference speed benchmarking
    - Accuracy vs efficiency tradeoffs
    - Deployment readiness scoring
    """

    def __init__(self, config: Config):
        self.config = config

    def analyze(self, all_metrics: List[ModelMetrics]) -> pd.DataFrame:
        """
        Generate compression analysis comparing all models.

        Args:
            all_metrics: List of ModelMetrics for each model.

        Returns:
            DataFrame with compression analysis results.
        """
        rows = []
        for m in all_metrics:
            # Efficiency score: accuracy per MB
            acc_per_mb = m.accuracy / max(m.model_size_mb, 0.01)
            # Speed efficiency: accuracy per ms inference time
            acc_per_ms = m.accuracy / max(m.inference_time_ms, 0.01)
            # Composite deployment score (higher is better)
            # Normalized: acc (0-1), speed (inverted, normalized), size (inverted, normalized)
            deployment_score = (
                m.accuracy * 0.5 +
                (1.0 / max(m.inference_time_ms, 1)) * 100 * 0.25 +
                (1.0 / max(m.model_size_mb, 1)) * 10 * 0.25
            )

            rows.append({
                "Model": m.model_name,
                "Accuracy": round(m.accuracy, 4),
                "Model Size (MB)": round(m.model_size_mb, 2),
                "Total Parameters": m.total_params,
                "Trainable Parameters": m.trainable_params,
                "Inference Time (ms)": round(m.inference_time_ms, 2),
                "Accuracy/MB": round(acc_per_mb, 4),
                "Accuracy/ms": round(acc_per_ms, 6),
                "Deployment Score": round(deployment_score, 4),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("Deployment Score", ascending=False)

        logger.info(
            f"Compression analysis: Most deployment-efficient = {df.iloc[0]['Model']}"
        )
        return df

    def save_compression_report(
        self,
        all_metrics: List[ModelMetrics],
        output_dir: Optional[str] = None,
    ) -> str:
        """Save compression analysis to CSV."""
        output_dir = output_dir or self.config.paths.summary_dir
        os.makedirs(output_dir, exist_ok=True)
        df = self.analyze(all_metrics)
        filepath = os.path.join(output_dir, "compression_study.csv")
        df.to_csv(filepath, index=False)
        logger.info(f"Compression report saved to {filepath}")
        return filepath
