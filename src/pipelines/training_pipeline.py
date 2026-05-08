# ==============================================================================
# src/pipelines/training_pipeline.py
# Main training orchestrator: trains all models with two-phase transfer learning.
# ==============================================================================

import os
import time
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from src.config.settings import Config
from src.data.dataset import DatasetManager
from src.models.model_factory import ModelFactory
from src.entity.model_entity import TrainingResult, EvaluationResult
from src.evaluation.metrics import MetricsCalculator
from src.visualization.training_plots import TrainingPlotter
from src.visualization.evaluation_plots import EvaluationPlotter
from src.visualization.gradcam import GradCAMVisualizer
from src.utils.logger import get_logger
from src.utils.helpers import set_seed, save_json

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Main training orchestrator. For each model:
    1. Build model with transfer learning (frozen backbone)
    2. Train Phase 1 (classifier head only)
    3. Fine-tune Phase 2 (unfreeze top layers)
    4. Evaluate on test set
    5. Generate metrics and visualizations
    6. Save model checkpoint
    """

    def __init__(self, config: Config):
        self.config = config
        self.dataset_mgr = DatasetManager(config)
        self.factory = ModelFactory(config)
        self.metrics_calc = MetricsCalculator(config)
        self.train_plotter = TrainingPlotter(config)
        self.eval_plotter = EvaluationPlotter(config)
        self.gradcam_viz = GradCAMVisualizer(config)

    def run(self) -> Dict[str, EvaluationResult]:
        """
        Execute the full training pipeline for all configured models.

        Returns:
            Dictionary mapping model name to EvaluationResult.
        """
        set_seed(self.config.data.seed)
        logger.info("=" * 70)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Config: {self.config.summary()}")

        # ── Prepare Data ──
        logger.info("Preparing dataset...")
        train_gen, val_gen, test_gen = self.dataset_mgr.prepare_data(use_augmentation=True)

        # Save dataset summary
        dataset_summary = self.dataset_mgr.get_dataset_summary()
        save_json(dataset_summary, os.path.join(
            self.config.paths.summary_dir, "dataset_summary.json"
        ))
        logger.info(f"Dataset: {dataset_summary['total_images']} images, "
                     f"{dataset_summary['num_classes']} classes")

        # ── Train Each Model ──
        results: Dict[str, EvaluationResult] = {}
        training_results: List[TrainingResult] = []
        all_metrics = []

        for model_name in self.config.model.model_names:
            logger.info("=" * 50)
            logger.info(f"TRAINING: {model_name}")
            logger.info("=" * 50)

            try:
                eval_result = self._train_single_model(
                    model_name, train_gen, val_gen, test_gen
                )
                results[model_name] = eval_result
                if eval_result.training_result:
                    training_results.append(eval_result.training_result)
                if eval_result.metrics:
                    all_metrics.append(eval_result.metrics)

            except Exception as e:
                logger.error(f"FAILED training {model_name}: {e}", exc_info=True)
                continue

            finally:
                tf.keras.backend.clear_session()

        # ── Generate Comparison Plots ──
        if training_results:
            self.train_plotter.plot_all_models_comparison(training_results)

        if all_metrics:
            self.eval_plotter.generate_all_plots(all_metrics)
            self.metrics_calc.save_metrics_report(all_metrics)
            self.metrics_calc.save_classification_reports(all_metrics)

        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"Models trained: {len(results)}/{len(self.config.model.model_names)}")
        logger.info("=" * 70)

        return results

    def _train_single_model(
        self,
        model_name: str,
        train_gen,
        val_gen,
        test_gen,
    ) -> EvaluationResult:
        """Train a single model through both phases."""

        start_time = time.time()
        builder = self.factory.get_builder(model_name)

        # ── Phase 1: Train classifier head (frozen backbone) ──
        logger.info(f"Phase 1: Training {model_name} (frozen backbone)...")
        model = builder.build(num_classes=self.dataset_mgr.num_classes)
        model = builder.compile_model(model, learning_rate=self.config.training.learning_rate)

        callbacks = builder.get_callbacks(model_name)

        history_p1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config.training.initial_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Phase 2: Fine-tune (unfreeze top layers) ──
        logger.info(f"Phase 2: Fine-tuning {model_name}...")
        model = builder.unfreeze_for_fine_tuning(model)
        model = builder.compile_model(
            model, learning_rate=self.config.training.fine_tune_lr
        )

        history_p2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config.training.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        training_time = time.time() - start_time

        # ── Merge Histories ──
        merged_history = self._merge_histories(history_p1.history, history_p2.history)

        training_result = TrainingResult(
            model_name=model_name,
            history=merged_history,
            training_time_seconds=training_time,
            initial_epochs=self.config.training.initial_epochs,
            fine_tune_epochs=self.config.training.fine_tune_epochs,
        )

        # ── Save Model ──
        model_path = os.path.join(
            self.config.paths.saved_models_dir, f"{model_name}_final.keras"
        )
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # ── Plot Training History ──
        self.train_plotter.plot_training_history(training_result)

        # ── Evaluate ──
        logger.info(f"Evaluating {model_name} on test set...")
        metrics = self.metrics_calc.compute_all_metrics(
            model, model_name, test_gen, model_path
        )
        # Add training time to metrics
        metrics.training_time_seconds = training_time

        # ── Grad-CAM (sample images) ──
        try:
            self._generate_sample_gradcam(
                model, model_name, builder.get_last_conv_layer_name(), test_gen
            )
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed for {model_name}: {e}")

        return EvaluationResult(
            model_name=model_name,
            training_result=training_result,
            metrics=metrics,
        )

    def _merge_histories(self, h1: dict, h2: dict) -> dict:
        """Merge two Keras history dictionaries."""
        merged = {}
        for key in h1:
            merged[key] = h1[key] + h2.get(key, [])
        for key in h2:
            if key not in merged:
                merged[key] = h2[key]
        return merged

    def _generate_sample_gradcam(
        self,
        model: tf.keras.Model,
        model_name: str,
        last_conv_layer: str,
        test_gen,
        n_samples: int = 5,
    ):
        """Generate Grad-CAM for a few sample test images."""
        test_gen.reset()
        images, labels = next(iter(test_gen))

        for i in range(min(n_samples, len(images))):
            img = images[i:i+1]
            pred = model.predict(img, verbose=0)
            pred_idx = int(np.argmax(pred))
            confidence = float(np.max(pred))
            class_name = self.dataset_mgr.class_names[pred_idx]

            self.gradcam_viz.visualize_and_save(
                model, img, last_conv_layer,
                class_name, model_name, confidence,
                filename=f"{model_name}_sample_{i}_{class_name}.png",
            )

