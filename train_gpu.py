"""
GPU Training Script — Run inside WSL2 with tf_gpu environment.
Retrains DenseNet121 and EfficientNetB0 on GPU (RTX 4050).

Usage (from WSL2 Ubuntu terminal):
    cd /mnt/c/Users/raj\ naik/Downloads/AI_Farming_Project_Final
    source ~/tf_gpu/bin/activate
    python3 train_gpu.py

Expected: ~2-3 min/epoch instead of ~25-30 min/epoch on CPU.
"""

import sys
import os
import time
import gc
import argparse

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np

# ============================================================
# GPU SETUP (must run before any model creation)
# ============================================================

def setup_gpu():
    """Configure GPU with memory growth to prevent OOM on 6GB VRAM."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU DETECTED: {gpus[0].name}")
            
            # Enable mixed precision for faster training on RTX 4050
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed precision (float16) enabled — ~2x speedup")
        except RuntimeError as e:
            print(f"⚠️ GPU config error: {e}")
    else:
        print("❌ NO GPU DETECTED — running on CPU")
        print("   Make sure you're running inside WSL2 with tf_gpu env")
        sys.exit(1)

setup_gpu()

from src.config.settings import Config
from src.data.dataset import DatasetManager
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import MetricsCalculator
from src.visualization.training_plots import TrainingPlotter
from src.visualization.evaluation_plots import EvaluationPlotter
from src.visualization.gradcam import GradCAMVisualizer
from src.entity.model_entity import TrainingResult
from src.utils.logger import get_logger
from src.utils.helpers import set_seed

logger = get_logger("train_gpu")

# ============================================================
# TRAINING CONFIG
# ============================================================

MODELS_TO_TRAIN = {
    "DenseNet121": {
        "initial_epochs": 5,
        "fine_tune_epochs": 25,
        "initial_lr": 1e-3,
        "fine_tune_lr": 1e-5,
    },
    "EfficientNetB0": {
        "initial_epochs": 5,
        "fine_tune_epochs": 25,
        "initial_lr": 1e-3,
        "fine_tune_lr": 1e-5,
    },
}

# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(model_name, phase, config):
    """Get callbacks optimized for GPU training."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    if phase == "phase2":
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    config.paths.saved_models_dir,
                    f"{model_name}_gpu_best.weights.h5"
                ),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        )

    return callbacks

# ============================================================
# TRAIN A SINGLE MODEL
# ============================================================

def train_model(model_name, model_config, config, dataset_mgr, factory):
    """Train a single model with GPU acceleration."""

    logger.info("=" * 60)
    logger.info(f"🚀 TRAINING {model_name} ON GPU")
    logger.info(f"   Phase 1: {model_config['initial_epochs']} epochs, LR={model_config['initial_lr']}")
    logger.info(f"   Phase 2: {model_config['fine_tune_epochs']} epochs, LR={model_config['fine_tune_lr']}")
    logger.info("=" * 60)

    start_time = time.time()

    # Reload data generators fresh for each model
    train_gen, val_gen, test_gen = dataset_mgr.prepare_data(use_augmentation=True)
    class_weights = dataset_mgr.get_class_weights()

    # Build model
    builder = factory.get_builder(model_name)
    model = builder.build(num_classes=dataset_mgr.num_classes)

    # --------------------------------------------------------
    # PHASE 1 — Frozen backbone
    # --------------------------------------------------------
    logger.info(f"PHASE 1: Frozen Backbone ({model_config['initial_epochs']} epochs)")

    model = builder.compile_model(model, learning_rate=model_config['initial_lr'])

    callbacks_p1 = get_callbacks(model_name, "phase1", config)

    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=model_config['initial_epochs'],
        callbacks=callbacks_p1,
        class_weight=class_weights,
        verbose=1,
    )

    p1_time = time.time() - start_time
    logger.info(f"Phase 1 done in {p1_time/60:.1f} min")

    # --------------------------------------------------------
    # PHASE 2 — Fine-tuning
    # --------------------------------------------------------
    logger.info(f"PHASE 2: Fine-Tuning ({model_config['fine_tune_epochs']} epochs)")

    model = builder.unfreeze_for_fine_tuning(model)
    model = builder.compile_model(model, learning_rate=model_config['fine_tune_lr'])

    callbacks_p2 = get_callbacks(model_name, "phase2", config)

    total_epochs = model_config['initial_epochs'] + model_config['fine_tune_epochs']

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=model_config['initial_epochs'],
        epochs=total_epochs,
        callbacks=callbacks_p2,
        class_weight=class_weights,
        verbose=1,
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.1f} minutes")

    # --------------------------------------------------------
    # Load best weights & save full model
    # --------------------------------------------------------
    weights_path = os.path.join(
        config.paths.saved_models_dir,
        f"{model_name}_gpu_best.weights.h5"
    )

    if os.path.exists(weights_path):
        logger.info("Loading best weights from checkpoint...")
        model.load_weights(weights_path)

    # Save full model
    save_path = os.path.join(
        config.paths.saved_models_dir,
        f"{model_name}_best.keras"
    )
    logger.info(f"Saving full model to: {save_path}")
    model.save(save_path)

    # Cleanup weights file
    if os.path.exists(weights_path):
        os.remove(weights_path)

    # --------------------------------------------------------
    # Merge histories
    # --------------------------------------------------------
    merged_history = {}
    for key in history_p1.history:
        merged_history[key] = (
            history_p1.history[key]
            + history_p2.history.get(key, [])
        )

    training_result = TrainingResult(
        model_name=model_name,
        history=merged_history,
        training_time_seconds=training_time,
        initial_epochs=model_config['initial_epochs'],
        fine_tune_epochs=model_config['fine_tune_epochs'],
    )

    # --------------------------------------------------------
    # Plots & evaluation
    # --------------------------------------------------------
    metrics_calc = MetricsCalculator(config)
    train_plotter = TrainingPlotter(config)
    eval_plotter = EvaluationPlotter(config)
    gradcam_viz = GradCAMVisualizer(config)

    logger.info("Generating training plots...")
    train_plotter.plot_training_history(training_result)

    logger.info("Evaluating model...")
    metrics = metrics_calc.compute_all_metrics(model, model_name, test_gen, save_path)
    metrics.training_time_seconds = training_time

    logger.info(f"  Accuracy:  {metrics.accuracy:.4f}")
    logger.info(f"  F1 Score:  {metrics.f1_score:.4f}")
    logger.info(f"  AUC:       {metrics.auc_score:.4f}")

    # Inference speed
    test_gen.reset()
    sample_images, _ = next(iter(test_gen))
    start_inf = time.time()
    _ = model.predict(sample_images[:1], verbose=0)
    inference_time = time.time() - start_inf
    logger.info(f"  Inference:  {inference_time*1000:.2f} ms")

    # Save reports
    logger.info("Saving evaluation reports...")
    eval_plotter.generate_all_plots([metrics])
    metrics_calc.save_metrics_report([metrics])
    metrics_calc.save_classification_reports([metrics])

    # Grad-CAM
    try:
        test_gen.reset()
        images, _ = next(iter(test_gen))
        conv_layer = builder.get_last_conv_layer_name()
        for i in range(min(3, len(images))):
            image = images[i:i+1]
            prediction = model.predict(image, verbose=0)
            predicted_class = dataset_mgr.class_names[int(np.argmax(prediction))]
            confidence = float(np.max(prediction))
            gradcam_viz.visualize_and_save(
                model=model,
                image=image,
                last_conv_layer_name=conv_layer,
                predicted_class=predicted_class,
                model_name=model_name,
                confidence=confidence,
                filename=f"{model_name}_gpu_sample_{i}_{predicted_class}.png",
            )
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")

    logger.info("=" * 60)
    logger.info(f"✅ {model_name} COMPLETE — Accuracy: {metrics.accuracy:.4f}")
    logger.info("=" * 60)

    # Cleanup
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GPU Training for Plant Disease Models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS_TO_TRAIN.keys()),
        help="Models to train (default: DenseNet121 EfficientNetB0)",
    )
    args = parser.parse_args()

    config = Config()
    set_seed(config.data.seed)
    dataset_mgr = DatasetManager(config)
    factory = ModelFactory(config)

    all_metrics = []
    total_start = time.time()

    for model_name in args.models:
        if model_name not in MODELS_TO_TRAIN:
            logger.error(f"Unknown model: {model_name}. Available: {list(MODELS_TO_TRAIN.keys())}")
            continue

        try:
            metrics = train_model(
                model_name,
                MODELS_TO_TRAIN[model_name],
                config,
                dataset_mgr,
                factory,
            )
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"❌ {model_name} FAILED: {e}", exc_info=True)

    total_time = time.time() - total_start

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("🏁 GPU TRAINING COMPLETE")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'AUC':<12}")
    print("-" * 56)
    for m in all_metrics:
        print(f"{m.model_name:<20} {m.accuracy:<12.4f} {m.f1_score:<12.4f} {m.auc_score:<12.4f}")
    print("-" * 56)

    # Previous CPU results for comparison
    print("\n📊 Previous CPU Results (for comparison):")
    print(f"  ResNet50:       99.50%  (already excellent — no retrain needed)")
    print(f"  DenseNet121:    95.99%  → check new result above")
    print(f"  EfficientNetB0: 95.80%  → check new result above")
    print(f"  MobileNetV2:    91.63%")


if __name__ == "__main__":
    main()
