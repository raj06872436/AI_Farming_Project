"""
How much did the old Grad-CAM preprocessing bug change the heatmaps?

The old gradient model replayed only `model.layers` before the backbone, which in
Keras 3 skips `preprocess_input` operations (ResNet50 caffe-style mean subtraction,
DenseNet121 ImageNet normalisation). This script recomputes Grad-CAM++ both ways on
the same images and reports how similar the maps are and whether the old version
still predicted the same class.
"""
import glob
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

import disease_heatmap as dh  # noqa: E402


def old_forward(model, target, backbone, penult):
    dual = tf.keras.models.Model(inputs=backbone.input, outputs=[target.output, backbone.output])
    before, after, past = [], [], False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer is backbone:
            past = True
            continue
        (after if past else before).append(layer)
        if past and layer is penult:
            break

    def f(x):
        for layer in before:
            x = layer(x)
        conv, y = dual(x)
        for layer in after:
            y = layer(y, training=False)
        return conv, y
    return f


def campp(forward, dense, x):
    with tf.GradientTape() as tape:
        conv, pen = forward(x)
        logits = tf.matmul(pen, dense.kernel) + dense.bias
        idx = int(tf.argmax(logits[0]))
        s = logits[:, idx]
    g = tape.gradient(s, conv)[0].numpy()
    c = conv[0].numpy()
    pg = np.maximum(g, 0)
    a = pg ** 2 / (2 * pg ** 2 + (c * pg ** 3).sum(axis=(0, 1), keepdims=True) + 1e-8)
    cam = np.maximum((c * (a * pg).sum(axis=(0, 1))).sum(-1), 0)
    return idx, cam / (cam.max() + 1e-8)


def main(data=os.path.join(ROOT, "PlantVillage"), n=60):
    random.seed(1)
    files = random.sample(sorted(glob.glob(os.path.join(data, "*", "*"))), int(n))
    for name in ("ResNet50", "DenseNet121", "EfficientNetB0"):
        model = tf.keras.models.load_model(os.path.join(ROOT, "saved_models", f"{name}_best.keras"), compile=False)
        target, backbone = dh._find_target_layer(model, dh.GRADCAM_LAYERS[name])
        dense, penult = dh._find_output_and_penultimate(model)
        new_f = dh._build_grad_model(model, target, backbone, penult)
        old_f = old_forward(model, target, backbone, penult)
        corrs, same_cls, same_peak = [], 0, 0
        for f in files:
            x = np.expand_dims(np.array(Image.open(f).convert("RGB").resize((224, 224)), np.float32) / 255.0, 0)
            i_new, c_new = campp(new_f, dense, x)
            i_old, c_old = campp(old_f, dense, x)
            same_cls += i_new == i_old
            same_peak += np.unravel_index(c_new.argmax(), c_new.shape) == np.unravel_index(c_old.argmax(), c_old.shape)
            if c_new.std() > 0 and c_old.std() > 0:
                corrs.append(np.corrcoef(c_new.ravel(), c_old.ravel())[0, 1])
        print(f"{name:15s} heatmap correlation old vs fixed: mean {np.mean(corrs):.2f} (min {np.min(corrs):.2f}) | "
              f"old pipeline predicted the same class: {same_cls}/{len(files)} | same peak cell: {same_peak}/{len(files)}")
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main(*sys.argv[1:])
