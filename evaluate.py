# ==============================================================================
# evaluate.py
# CLI entry point for the advanced evaluation pipeline.
# Usage: python evaluate.py
# ==============================================================================

"""
Run advanced evaluation on trained models:
- K-Fold Cross Validation
- Statistical Analysis with Confidence Intervals
- Ablation Study
- Compression Study
- Hyperparameter Tuning

Prerequisites: Run 'python train.py' first to train and save models.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import Config
from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.pipelines.hyperparameter_pipeline import HyperparameterPipeline
from src.utils.logger import get_logger
from src.utils.helpers import set_seed

logger = get_logger("evaluate")


def main():
    """Run the full evaluation pipeline."""
    logger.info("=" * 70)
    logger.info("AI AGRICULTURE PROJECT — EVALUATION")
    logger.info("Advanced Analysis & Research-Level Evaluation")
    logger.info("=" * 70)

    config = Config()
    set_seed(config.data.seed)

    # ── Run Evaluation Pipeline ──
    eval_pipeline = EvaluationPipeline(config)
    eval_pipeline.run_full_evaluation()

    # ── Run Hyperparameter Tuning ──
    logger.info("Starting Hyperparameter Tuning...")
    hp_pipeline = HyperparameterPipeline(config)
    hp_results = hp_pipeline.run_all_models()

    logger.info("\nBest Hyperparameters:")
    for name, params in hp_results.items():
        logger.info(f"  {name}: {params}")

    logger.info("=" * 70)
    logger.info("Evaluation complete! Check reports/ for all outputs.")
    logger.info("Run 'streamlit run app.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
