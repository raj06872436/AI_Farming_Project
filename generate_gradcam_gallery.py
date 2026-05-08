"""
Generate Grad-CAM gallery images for EfficientNetB0 and DenseNet121.
Saves sample visualizations to reports/gradcam/
"""

import os, sys, gc
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Config ──
SAVED_MODELS_DIR = "saved_models"
GRADCAM_DIR = os.path.join("reports", "gradcam")
DATASET_DIR = "PlantVillage"
os.makedirs(GRADCAM_DIR, exist_ok=True)

MODELS = {
    "EfficientNetB0": {"layer": "top_conv"},
    "DenseNet121": {"layer": "relu"},
}

NUM_SAMPLES = 5


def find_model_path(name):
    for ext in ["_best.keras", "_final.keras", "_best.h5", "_final.h5"]:
        p = os.path.join(SAVED_MODELS_DIR, f"{name}{ext}")
        if os.path.exists(p):
            return p
    return None


def get_sample_images(dataset_dir, num=5):
    """Get random sample images from different classes."""
    samples = []
    classes = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])
    np.random.seed(42)
    chosen_classes = np.random.choice(classes, min(num, len(classes)), replace=False)

    for cls in chosen_classes:
        cls_dir = os.path.join(dataset_dir, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            img_name = np.random.choice(imgs)
            img_path = os.path.join(cls_dir, img_name)
            samples.append((img_path, cls))
    return samples


def generate_gradcam(model, img_array, layer_name):
    """Generate Grad-CAM heatmap — handles nested backbone models."""
    target = None
    backbone = None

    # Find target layer
    for layer in model.layers:
        if layer.name == layer_name:
            target = layer
            break
        if hasattr(layer, 'layers'):
            try:
                target = layer.get_layer(layer_name)
                backbone = layer
                break
            except (ValueError, KeyError):
                continue

    # Fallback: last Conv2D
    if target is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target = layer
                break
            if hasattr(layer, 'layers'):
                for sl in reversed(layer.layers):
                    if isinstance(sl, tf.keras.layers.Conv2D):
                        target = sl
                        backbone = layer
                        break
                if target:
                    break

    if target is None:
        print(f"  WARNING: No target layer found")
        return None

    # Build gradient model
    try:
        if backbone is None:
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[target.output, model.output]
            )
        else:
            backbone_dual = tf.keras.models.Model(
                inputs=backbone.input,
                outputs=[target.output, backbone.output]
            )
            inp = model.input
            x = inp
            for l in model.layers:
                if isinstance(l, tf.keras.layers.InputLayer):
                    continue
                if l is backbone:
                    break
                x = l(x)
            conv_out_t, bb_out = backbone_dual(x)
            y = bb_out
            past_backbone = False
            for l in model.layers:
                if l is backbone:
                    past_backbone = True
                    continue
                if past_backbone and not isinstance(l, tf.keras.layers.InputLayer):
                    y = l(y)
            grad_model = tf.keras.models.Model(inputs=inp, outputs=[conv_out_t, y])

        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(img_array)
            pred_idx = tf.argmax(predictions[0])
            class_out = predictions[:, pred_idx]
        grads = tape.gradient(class_out, conv_out)
        if grads is None:
            return None

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def save_gradcam_figure(img_array, heatmap, model_name, sample_idx, class_name):
    """Save a Grad-CAM overlay figure."""
    hm_uint8 = np.uint8(255 * heatmap)
    hm_img = Image.fromarray(hm_uint8).resize((224, 224), Image.BILINEAR)
    heatmap_resized = np.array(hm_img).astype(np.float32) / 255.0

    colormap = cm.jet(heatmap_resized)[:, :, :3]
    original = img_array[0]
    overlay = np.clip(0.6 * original + 0.4 * colormap, 0, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.imshow(original)
    ax1.set_title("Original", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(heatmap_resized, cmap="jet")
    ax2.set_title("Heatmap", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3.imshow(overlay)
    ax3.set_title("Grad-CAM Overlay", fontsize=11, fontweight="bold")
    ax3.axis("off")

    clean_class = class_name.replace(" ", "_")
    fig.suptitle(f"{model_name} — {class_name.replace('_', ' ')}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    filename = f"{model_name}_sample_{sample_idx}_{clean_class}.png"
    filepath = os.path.join(GRADCAM_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


# ============================================================
# MAIN
# ============================================================

samples = get_sample_images(DATASET_DIR, NUM_SAMPLES)
print(f"Got {len(samples)} sample images from {len(set(s[1] for s in samples))} classes")

for model_name, config in MODELS.items():
    model_path = find_model_path(model_name)
    if model_path is None:
        print(f"SKIP {model_name} — no saved model")
        continue

    print(f"\n{'='*60}")
    print(f"Generating Grad-CAM for {model_name}")
    print(f"Model: {model_path}")
    print(f"Target layer: {config['layer']}")
    print(f"{'='*60}")

    model = tf.keras.models.load_model(model_path, compile=False)

    for idx, (img_path, class_name) in enumerate(samples):
        print(f"\n  Sample {idx}: {class_name}")

        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)

        heatmap = generate_gradcam(model, arr, config["layer"])
        if heatmap is not None:
            save_gradcam_figure(arr, heatmap, model_name, idx, class_name)
        else:
            print(f"  FAILED for sample {idx}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()

print(f"\n{'='*60}")
print("DONE! All Grad-CAM images saved to reports/gradcam/")
print(f"{'='*60}")
