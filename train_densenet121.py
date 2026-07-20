"""
Train ONLY DenseNet121.
Research-grade transfer learning pipeline.
Optimized for PlantVillage dataset.
Speed-optimized: reduced inter-epoch delays.

Usage:  .venv\Scripts\python train_densenet121.py
"""

import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

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

logger = get_logger("train_densenet121")

# ============================================================
# CONFIG
# ============================================================

config = Config()
set_seed(config.data.seed)

INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 15
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5

MODEL_NAME = "DenseNet121"

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
    logger.warning("NO GPU — training on CPU")
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
# SPEED-OPTIMIZED CALLBACKS
# ============================================================

def get_fast_callbacks(model_name, phase="phase1"):
    """
    Faster callbacks to reduce inter-epoch delay.
    - Phase 1: no checkpoint saving (we'll fine-tune anyway)
    - Phase 2: save weights only (much faster than full model)
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
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
        # Only save during fine-tuning, and save weights only (MUCH faster)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    config.paths.saved_models_dir,
                    f"{model_name}_best.weights.h5"
                ),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,   # KEY: saves in ~1s instead of ~30s
                verbose=1,
            )
        )

    return callbacks

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
    # BUILD MODEL
    # --------------------------------------------------------

    builder = factory.get_builder(MODEL_NAME)

    logger.info("Building model...")

    model = builder.build(
        num_classes=dataset_mgr.num_classes
    )

    # --------------------------------------------------------
    # PHASE 1 - FROZEN BACKBONE (no checkpoint = faster epochs)
    # --------------------------------------------------------

    logger.info(
        f"PHASE 1: Frozen Backbone "
        f"({INITIAL_EPOCHS} epochs, LR={INITIAL_LR})"
    )

    model = builder.compile_model(
        model,
        learning_rate=INITIAL_LR
    )

    callbacks_p1 = get_fast_callbacks(MODEL_NAME, phase="phase1")

    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks_p1,
        class_weight=class_weights,
        verbose=1
    )

    p1_time = time.time() - start_time
    logger.info(f"Phase 1 done in {p1_time/60:.1f} min")

    # --------------------------------------------------------
    # PHASE 2 - FINE TUNING (saves weights only = fast saves)
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

    callbacks_p2 = get_fast_callbacks(MODEL_NAME, phase="phase2")

    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=INITIAL_EPOCHS,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        callbacks=callbacks_p2,
        class_weight=class_weights,
        verbose=1
    )

    # --------------------------------------------------------
    # TRAINING TIME
    # --------------------------------------------------------

    training_time = time.time() - start_time

    logger.info(
        f"Training completed in "
        f"{training_time/60:.1f} minutes"
    )

    # --------------------------------------------------------
    # LOAD BEST WEIGHTS & SAVE FULL MODEL
    # --------------------------------------------------------

    weights_path = os.path.join(
        config.paths.saved_models_dir,
        f"{MODEL_NAME}_best.weights.h5"
    )

    if os.path.exists(weights_path):
        logger.info("Loading best weights from checkpoint...")
        model.load_weights(weights_path)

    # Save full model (once, at the end)
    save_path = os.path.join(
        config.paths.saved_models_dir,
        f"{MODEL_NAME}_final.keras"
    )
    logger.info(f"Saving full model to: {save_path}")
    model.save(save_path)

    # Also save as _best.keras for the dashboard
    best_path = os.path.join(
        config.paths.saved_models_dir,
        f"{MODEL_NAME}_best.keras"
    )
    model.save(best_path)
    logger.info("Full model saved!")

    # Clean up weights file
    if os.path.exists(weights_path):
        os.remove(weights_path)

    # --------------------------------------------------------
    # MERGE HISTORY
    # --------------------------------------------------------

    merged_history = {}
    for key in history_phase1.history:
        merged_history[key] = (
            history_phase1.history[key]
            + history_phase2.history.get(key, [])
        )

    training_result = TrainingResult(
        model_name=MODEL_NAME,
        history=merged_history,
        training_time_seconds=training_time,
        initial_epochs=INITIAL_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS
    )

    # --------------------------------------------------------
    # TRAINING PLOTS
    # --------------------------------------------------------

    logger.info("Generating training plots...")
    train_plotter.plot_training_history(training_result)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    logger.info("Evaluating model...")

    metrics = metrics_calc.compute_all_metrics(
        model, MODEL_NAME, test_gen, save_path
    )
    metrics.training_time_seconds = training_time

    logger.info(f"Accuracy:  {metrics.accuracy:.4f}")
    logger.info(f"F1 Score:  {metrics.f1_score:.4f}")
    logger.info(f"AUC:       {metrics.auc_score:.4f}")

    # --------------------------------------------------------
    # INFERENCE SPEED
    # --------------------------------------------------------

    logger.info("Benchmarking inference speed...")
    test_gen.reset()
    sample_images, _ = next(iter(test_gen))

    start_inf = time.time()
    _ = model.predict(sample_images[:1], verbose=0)
    inference_time = time.time() - start_inf

    logger.info(f"Inference Time: {inference_time*1000:.2f} ms")

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
            prediction = model.predict(image, verbose=0)
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
        logger.warning(f"Grad-CAM failed: {e}")

    logger.info("=" * 60)
    logger.info("DENSENET121 TRAINING COMPLETE")
    logger.info(f"Accuracy:  {metrics.accuracy:.4f}")
    logger.info(f"F1 Score:  {metrics.f1_score:.4f}")
    logger.info("=" * 60)

except Exception as e:
    logger.error(f"TRAINING FAILED: {e}", exc_info=True)

finally:
    try:
        del model
    except:
        pass
    tf.keras.backend.clear_session()
    gc.collect()

logger.info("Run: streamlit run app.py")
