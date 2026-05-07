# ==============================================================================
# src/pipelines/evaluation_pipeline.py
# Post-training evaluation orchestrator.
# Runs cross-validation, statistical analysis, ablation, compression studies.
# ==============================================================================

import os
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from src.config.settings import Config
from src.data.dataset import DatasetManager
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.statistical_analysis import StatisticalAnalyzer
from src.evaluation.ablation_study import AblationStudy
from src.evaluation.compression_study import CompressionStudy
from src.entity.model_entity import ModelMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    """
    Post-training evaluation orchestrator.
    Loads saved models and runs the full evaluation suite:
    - Metrics computation
    - Cross-validation
    - Statistical analysis
    - Ablation study
    - Compression study
    """

    def __init__(self, config: Config):
        self.config = config
        self.dataset_mgr = DatasetManager(config)
        self.factory = ModelFactory(config)
        self.metrics_calc = MetricsCalculator(config)
        self.cv = CrossValidator(config)
        self.stats = StatisticalAnalyzer(config)
        self.ablation = AblationStudy(config)
        self.compression = CompressionStudy(config)

    def run_full_evaluation(self) -> None:
        """Run the complete evaluation pipeline."""
        logger.info("=" * 70)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info("=" * 70)

        # ── Load Models and Compute Metrics ──
        all_metrics = self._load_and_evaluate_models()

        if not all_metrics:
            logger.error("No models found for evaluation. Run training first.")
            return

        # ── Cross Validation ──
        logger.info("Running Cross Validation...")
        try:
            X, y = self.dataset_mgr.load_full_dataset()
            cv_results = self.cv.run_all_models(X, y)
            self.cv.save_cv_results(cv_results)

            # ── Statistical Analysis ──
            logger.info("Running Statistical Analysis...")
            self.stats.save_statistical_report(cv_results, all_metrics)
        except Exception as e:
            logger.error(f"Cross-validation/stats failed: {e}", exc_info=True)
            cv_results = []

        # ── Ablation Study ──
        logger.info("Running Ablation Study...")
        try:
            self._run_ablation_study()
        except Exception as e:
            logger.error(f"Ablation study failed: {e}", exc_info=True)

        # ── Compression Study ──
        logger.info("Running Compression Study...")
        try:
            self.compression.save_compression_report(all_metrics)
        except Exception as e:
            logger.error(f"Compression study failed: {e}", exc_info=True)

        logger.info("=" * 70)
        logger.info("EVALUATION PIPELINE COMPLETE")
        logger.info("=" * 70)

    def _load_and_evaluate_models(self) -> List[ModelMetrics]:
        """Load saved models and compute metrics."""
        all_metrics = []

        _, val_gen, test_gen = self.dataset_mgr.prepare_data(use_augmentation=False)

        for model_name in self.config.model.model_names:
            model_path = os.path.join(
                self.config.paths.saved_models_dir, f"{model_name}_final.keras"
            )

            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue

            logger.info(f"Loading {model_name} from {model_path}...")
            try:
                model = tf.keras.models.load_model(model_path)
                metrics = self.metrics_calc.compute_all_metrics(
                    model, model_name, test_gen, model_path
                )
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
            finally:
                tf.keras.backend.clear_session()

        return all_metrics

    def _run_ablation_study(self) -> None:
        """Run ablation study with augmented and non-augmented data."""
        train_gen_no_aug, val_gen_no_aug, _ = self.dataset_mgr.prepare_data(
            use_augmentation=False
        )
        train_gen_aug, val_gen_aug, _ = self.dataset_mgr.prepare_data(
            use_augmentation=True
        )

        all_ablation_results = []
        for model_name in self.config.model.model_names:
            try:
                results = self.ablation.run_full_ablation(
                    model_name, train_gen_no_aug, train_gen_aug, val_gen_aug
                )
                all_ablation_results.extend(results)
            except Exception as e:
                logger.error(f"Ablation failed for {model_name}: {e}")

        if all_ablation_results:
            self.ablation.save_ablation_results(all_ablation_results)
