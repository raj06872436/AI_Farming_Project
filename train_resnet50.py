"""
Train ONLY ResNet50.
Research-grade transfer learning pipeline.
Optimized for PlantVillage dataset.

Usage:  .venv\Scripts\python train_resnet50.py
"""

import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np

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

# ============================================================
# LOGGER
# ============================================================

logger = get_logger("train_resnet50")

# ============================================================
# CONFIG
# ============================================================

config = Config()

set_seed(config.data.seed)

INITIAL_EPOCHS = 10       # More warmup for ResNet50's deep backbone
FINE_TUNE_EPOCHS = 20     # More fine-tuning for larger capacity
INITIAL_LR = 3e-4         # Higher LR for Phase 1
FINE_TUNE_LR = 2e-5       # Fine-tune LR

MODEL_NAME = "ResNet50"

# ============================================================
# GPU CHECK
# ============================================================

logger.info("=" * 60)
logger.info("GPU CHECK")
logger.info("=" * 60)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    logger.info(f"GPU DETECTED: {gpus}")
else:
    logger.warning("NO GPU DETECTED — training will be slow on CPU")

logger.info("=" * 60)

# ============================================================
# DATASET
# ============================================================

logger.info("Loading dataset...")

dataset_mgr = DatasetManager(config)

train_gen, val_gen, test_gen = dataset_mgr.prepare_data(
    use_augmentation=True
)

class_weights = dataset_mgr.get_class_weights()

logger.info(f"Classes: {dataset_mgr.class_names}")
logger.info(f"Num Classes: {dataset_mgr.num_classes}")
logger.info(f"Class weights computed for {len(class_weights)} classes")

# ============================================================
# COMPONENTS
# ============================================================

factory = ModelFactory(config)

metrics_calc = MetricsCalculator(config)

train_plotter = TrainingPlotter(config)

eval_plotter = EvaluationPlotter(config)

gradcam_viz = GradCAMVisualizer(config)

# ============================================================
# TRAINING
# ============================================================

logger.info("=" * 60)
logger.info(f"TRAINING {MODEL_NAME}")
logger.info(f"Phase 1: {INITIAL_EPOCHS} epochs, LR={INITIAL_LR}")
logger.info(f"Phase 2: {FINE_TUNE_EPOCHS} epochs, LR={FINE_TUNE_LR}")
logger.info("=" * 60)

try:

    start_time = time.time()

    # --------------------------------------------------------
    # BUILDER
    # --------------------------------------------------------

    builder = factory.get_builder(MODEL_NAME)

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    logger.info("Building model...")

    model = builder.build(
        num_classes=dataset_mgr.num_classes
    )

    # --------------------------------------------------------
    # PHASE 1 - FROZEN BACKBONE
    # --------------------------------------------------------

    logger.info(
        f"PHASE 1: Frozen Backbone "
        f"({INITIAL_EPOCHS} epochs, LR={INITIAL_LR})"
    )

    model = builder.compile_model(
        model,
        learning_rate=INITIAL_LR
    )

    callbacks = builder.get_callbacks(MODEL_NAME)

    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # --------------------------------------------------------
    # PHASE 2 - FINE TUNING
    # --------------------------------------------------------

    logger.info(
        f"PHASE 2: Fine-Tuning "
        f"({FINE_TUNE_EPOCHS} epochs, LR={FINE_TUNE_LR})"
    )

    model = builder.unfreeze_for_fine_tuning(model)

    model = builder.compile_model(
        model,
        learning_rate=FINE_TUNE_LR
    )

    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=INITIAL_EPOCHS,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # --------------------------------------------------------
    # TRAINING TIME
    # --------------------------------------------------------

    training_time = time.time() - start_time

    logger.info(
        f"Training completed in "
        f"{training_time/60:.2f} minutes"
    )

    # --------------------------------------------------------
    # MERGE HISTORY
    # --------------------------------------------------------

    merged_history = {}

    for key in history_phase1.history:

        merged_history[key] = (
            history_phase1.history[key]
            +
            history_phase2.history.get(key, [])
        )

    # --------------------------------------------------------
    # TRAINING RESULT
    # --------------------------------------------------------

    training_result = TrainingResult(
        model_name=MODEL_NAME,
        history=merged_history,
        training_time_seconds=training_time,
        initial_epochs=INITIAL_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    save_path = os.path.join(
        config.paths.saved_models_dir,
        "ResNet50_final.keras"
    )

    logger.info(f"Saving model to: {save_path}")

    model.save(save_path)

    # VERIFY SAVE

    assert os.path.exists(save_path), (
        "MODEL SAVE FAILED!"
    )

    logger.info("Model saved successfully!")

    # --------------------------------------------------------
    # TRAINING PLOTS
    # --------------------------------------------------------

    logger.info("Generating training plots...")

    train_plotter.plot_training_history(
        training_result
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    logger.info("Evaluating model...")

    metrics = metrics_calc.compute_all_metrics(
        model,
        MODEL_NAME,
        test_gen,
        save_path
    )

    metrics.training_time_seconds = training_time

    logger.info(
        f"Accuracy: {metrics.accuracy:.4f}"
    )

    logger.info(
        f"F1 Score: {metrics.f1_score:.4f}"
    )

    logger.info(
        f"AUC: {metrics.auc_score:.4f}"
    )

    # --------------------------------------------------------
    # INFERENCE SPEED
    # --------------------------------------------------------

    logger.info("Benchmarking inference speed...")

    test_gen.reset()

    sample_images, _ = next(iter(test_gen))

    start_inf = time.time()

    _ = model.predict(
        sample_images[:1],
        verbose=0
    )

    inference_time = time.time() - start_inf

    logger.info(
        f"Inference Time: "
        f"{inference_time*1000:.2f} ms"
    )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    logger.info("Generating evaluation plots...")

    eval_plotter.generate_all_plots([metrics])

    metrics_calc.save_metrics_report([metrics])

    metrics_calc.save_classification_reports([metrics])

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    logger.info("Generating Grad-CAM visualizations...")

    try:

        test_gen.reset()

        images, _ = next(iter(test_gen))

        conv_layer = builder.get_last_conv_layer_name()

        for i in range(min(3, len(images))):

            image = images[i:i+1]

            prediction = model.predict(
                image,
                verbose=0
            )

            predicted_class = dataset_mgr.class_names[
                int(np.argmax(prediction))
            ]

            confidence = float(np.max(prediction))

            gradcam_viz.visualize_and_save(
                model=model,
                image=image,
                last_conv_layer_name=conv_layer,
                predicted_class=predicted_class,
                model_name=MODEL_NAME,
                confidence=confidence,
                filename=(
                    f"{MODEL_NAME}_sample_{i}_"
                    f"{predicted_class}.png"
                )
            )

    except Exception as e:

        logger.warning(
            f"Grad-CAM failed: {e}"
        )

    logger.info("=" * 60)
    logger.info("RESNET50 TRAINING COMPLETE")
    logger.info(f"Accuracy: {metrics.accuracy:.4f}")
    logger.info(f"F1 Score: {metrics.f1_score:.4f}")
    logger.info("=" * 60)

except Exception as e:

    logger.error(
        f"TRAINING FAILED: {e}",
        exc_info=True
    )

finally:

    try:
        del model
    except:
        pass

    tf.keras.backend.clear_session()

    gc.collect()

logger.info("Run: streamlit run app.py")
