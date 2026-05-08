"""
Generate ALL reports for ALL trained models.
- Confusion matrices
- ROC curves & PR curves
- Classification reports
- Model comparison CSV
- Comparison bar charts (accuracy, F1, AUC, inference, size)
- Updated model_registry.json
"""

import sys
import os
import json
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np

from src.config.settings import Config
from src.data.dataset import DatasetManager
from src.evaluation.metrics import MetricsCalculator
from src.visualization.evaluation_plots import EvaluationPlotter
from src.utils.logger import get_logger

logger = get_logger("generate_reports")

# ============================================================
# CONFIG
# ============================================================

config = Config()

MODELS_TO_EVALUATE = [
    "MobileNetV2",
    "ResNet50",
    "EfficientNetB0",
    "DenseNet121",
]

SAVED_MODELS_DIR = config.paths.saved_models_dir

# ============================================================
# DATASET
# ============================================================

logger.info("Loading dataset...")

dataset_mgr = DatasetManager(config)
_, _, test_gen = dataset_mgr.prepare_data(use_augmentation=False)

logger.info(f"Classes: {dataset_mgr.num_classes}")
logger.info(f"Test samples: {test_gen.samples}")

# ============================================================
# COMPONENTS
# ============================================================

metrics_calc = MetricsCalculator(config)
eval_plotter = EvaluationPlotter(config)

# ============================================================
# EVALUATE ALL MODELS
# ============================================================

all_metrics = []

for model_name in MODELS_TO_EVALUATE:

    # Find model file
    model_path = None
    for suffix in ["_best.keras", "_final.keras", "_best.h5", "_final.h5"]:
        p = os.path.join(SAVED_MODELS_DIR, f"{model_name}{suffix}")
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        logger.warning(f"SKIPPING {model_name} — no saved model found")
        continue

    logger.info("=" * 60)
    logger.info(f"EVALUATING {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info("=" * 60)

    try:
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)

        # Compute all metrics
        metrics = metrics_calc.compute_all_metrics(
            model, model_name, test_gen, model_path
        )

        all_metrics.append(metrics)

        logger.info(
            f"{model_name}: "
            f"Acc={metrics.accuracy:.4f} | "
            f"F1={metrics.f1_score:.4f} | "
            f"AUC={metrics.auc_score:.4f}"
        )

        # Clean up
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    except Exception as e:
        logger.error(f"FAILED to evaluate {model_name}: {e}", exc_info=True)

# ============================================================
# GENERATE ALL PLOTS
# ============================================================

if all_metrics:

    logger.info("=" * 60)
    logger.info("GENERATING ALL REPORTS")
    logger.info("=" * 60)

    # Per-model plots (confusion matrix, ROC, PR)
    # + comparison plots (bar charts, inference, size)
    eval_plotter.generate_all_plots(all_metrics)

    # Save CSV report
    metrics_calc.save_metrics_report(all_metrics)

    # Save classification reports
    metrics_calc.save_classification_reports(all_metrics)

    # ── Update model_registry.json ──
    registry_path = os.path.join("research_bundle", "model_registry.json")

    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    for m in all_metrics:
        if m.model_name in registry:
            registry[m.model_name]["accuracy"] = round(m.accuracy, 4)
            registry[m.model_name]["status"] = "verified"
            registry[m.model_name]["size_mb"] = round(m.model_size_mb, 1)
            registry[m.model_name]["inference_ms"] = round(m.inference_time_ms, 1)
            registry[m.model_name]["params"] = f"{m.total_params / 1e6:.1f}M"
        else:
            registry[m.model_name] = {
                "architecture": m.model_name,
                "path": f"{m.model_name}_best.keras",
                "status": "verified",
                "accuracy": round(m.accuracy, 4),
                "params": f"{m.total_params / 1e6:.1f}M",
                "size_mb": round(m.model_size_mb, 1),
                "inference_ms": round(m.inference_time_ms, 1),
                "strengths": "—",
                "weaknesses": "—",
                "deployment": "—",
            }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    logger.info(f"Updated registry: {registry_path}")

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("=" * 60)

    for m in all_metrics:
        logger.info(
            f"  {m.model_name:20s} | "
            f"Acc: {m.accuracy:.4f} | "
            f"F1: {m.f1_score:.4f} | "
            f"AUC: {m.auc_score:.4f} | "
            f"Size: {m.model_size_mb:.1f}MB | "
            f"Speed: {m.inference_time_ms:.1f}ms"
        )

    logger.info("Refresh Streamlit to see updated reports!")

else:
    logger.error("No models were evaluated!")
