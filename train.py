# ==============================================================================
# train.py
# CLI entry point for the training pipeline.
# Usage: python train.py
# ==============================================================================

"""
Train all configured models using the PlantVillage dataset.

This script:
1. Loads and preprocesses the dataset
2. Trains each model with two-phase transfer learning
3. Evaluates on the test set
4. Generates all metrics, plots, and Grad-CAM visualizations
5. Saves trained models to saved_models/
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import Config
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.logger import get_logger
from src.utils.helpers import set_seed

logger = get_logger("train")


def main():
    """Run the full training pipeline."""
    logger.info("=" * 70)
    logger.info("AI AGRICULTURE PROJECT — TRAINING")
    logger.info("Plant Disease Detection using Transfer Learning")
    logger.info("=" * 70)

    # Initialize configuration
    config = Config()
    set_seed(config.data.seed)

    logger.info(f"Configuration: {config.summary()}")

    # Run training pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run()

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    for model_name, result in results.items():
        if result.metrics:
            m = result.metrics
            logger.info(
                f"  {model_name:20s} | Acc: {m.accuracy:.4f} | "
                f"F1: {m.f1_score:.4f} | AUC: {m.auc_score:.4f}"
            )

    logger.info("=" * 70)
    logger.info("Training complete! Check reports/ for all outputs.")
    logger.info("Run 'python evaluate.py' for advanced analysis (CV, ablation, etc.)")


if __name__ == "__main__":
    main()
