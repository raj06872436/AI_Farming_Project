# ==============================================================================
# src/evaluation/statistical_analysis.py
# Statistical analysis: mean, variance, std, confidence intervals,
# and comparative statistical tests between models.
# ==============================================================================

import os
from typing import List, Optional

import numpy as np
from scipy import stats
import pandas as pd

from src.config.settings import Config
from src.entity.model_entity import ModelMetrics, CrossValidationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis on model evaluation results."""

    def __init__(self, config: Config):
        self.config = config

    def compute_statistics(self, cv_results: List[CrossValidationResult],
                           confidence_level: float = 0.95) -> pd.DataFrame:
        """Compute descriptive statistics from cross-validation results."""
        rows = []
        for result in cv_results:
            for metric_name, data in [
                ("Accuracy", result.fold_accuracies),
                ("Precision", result.fold_precisions),
                ("Recall", result.fold_recalls),
                ("F1 Score", result.fold_f1_scores),
            ]:
                arr = np.array(data)
                ci_lo, ci_hi = (0.0, 0.0)
                if len(arr) >= 2:
                    sem = stats.sem(arr)
                    interval = stats.t.interval(confidence_level, len(arr)-1,
                                                loc=np.mean(arr), scale=sem)
                    ci_lo, ci_hi = round(interval[0], 4), round(interval[1], 4)

                rows.append({
                    "Model": result.model_name, "Metric": metric_name,
                    "Mean": round(float(np.mean(arr)), 4),
                    "Variance": round(float(np.var(arr)), 6),
                    "Std Dev": round(float(np.std(arr)), 4),
                    "CI Lower (95%)": ci_lo, "CI Upper (95%)": ci_hi,
                })
        return pd.DataFrame(rows)

    def paired_comparison(self, cv_results: List[CrossValidationResult]) -> pd.DataFrame:
        """Perform paired t-tests between all model pairs on accuracy."""
        rows = []
        for i in range(len(cv_results)):
            for j in range(i + 1, len(cv_results)):
                acc_i = np.array(cv_results[i].fold_accuracies)
                acc_j = np.array(cv_results[j].fold_accuracies)
                if len(acc_i) > 1 and len(acc_j) > 1:
                    t_stat, p_val = stats.ttest_rel(acc_i, acc_j)
                    sig = "Yes" if p_val < 0.05 else "No"
                else:
                    t_stat, p_val, sig = 0.0, 1.0, "N/A"
                rows.append({
                    "Model A": cv_results[i].model_name,
                    "Model B": cv_results[j].model_name,
                    "t-statistic": round(t_stat, 4), "p-value": round(p_val, 4),
                    "Significant (p<0.05)": sig,
                })
        return pd.DataFrame(rows)

    def model_ranking(self, all_metrics: List[ModelMetrics]) -> pd.DataFrame:
        """Rank models across multiple criteria."""
        data = [{
            "Model": m.model_name, "Accuracy": m.accuracy, "F1 Score": m.f1_score,
            "AUC": m.auc_score, "Inference (ms)": m.inference_time_ms,
            "Size (MB)": m.model_size_mb,
        } for m in all_metrics]

        df = pd.DataFrame(data)
        df["Acc Rank"] = df["Accuracy"].rank(ascending=False).astype(int)
        df["F1 Rank"] = df["F1 Score"].rank(ascending=False).astype(int)
        df["AUC Rank"] = df["AUC"].rank(ascending=False).astype(int)
        df["Speed Rank"] = df["Inference (ms)"].rank(ascending=True).astype(int)
        df["Size Rank"] = df["Size (MB)"].rank(ascending=True).astype(int)
        rank_cols = ["Acc Rank", "F1 Rank", "AUC Rank", "Speed Rank", "Size Rank"]
        df["Avg Rank"] = df[rank_cols].mean(axis=1).round(2)
        return df.sort_values("Avg Rank")

    def save_statistical_report(self, cv_results: List[CrossValidationResult],
                                all_metrics: List[ModelMetrics],
                                output_dir: Optional[str] = None) -> None:
        """Save all statistical analysis outputs."""
        output_dir = output_dir or self.config.paths.summary_dir
        os.makedirs(output_dir, exist_ok=True)

        self.compute_statistics(cv_results).to_csv(
            os.path.join(output_dir, "statistical_summary.csv"), index=False)

        if len(cv_results) >= 2:
            self.paired_comparison(cv_results).to_csv(
                os.path.join(output_dir, "paired_comparisons.csv"), index=False)

        if all_metrics:
            self.model_ranking(all_metrics).to_csv(
                os.path.join(output_dir, "model_rankings.csv"), index=False)

        logger.info(f"Statistical reports saved to {output_dir}")
