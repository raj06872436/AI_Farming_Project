"""
Attention-vs-lesion study: do the classifiers look at the disease?

For every leaf photo and every trained model this script compares the model's
Grad-CAM / Grad-CAM++ attention map with a pixel-level map of diseased tissue
(from lesion_segmentation.py) and records:

  e_lesion / e_healthy / e_bg   share of attention "energy" on diseased tissue,
                                healthy leaf tissue and background (sum to 1)
  focus_ratio                   mean attention on lesions ÷ mean attention on the
                                whole leaf (1 = no preference, >1 = looks at lesions)
  pointing_hit                  does the attention peak fall on a lesion (±15 px)?
  iou_05                        IoU between attention > 0.5 and the lesion mask
  cam_area_pct                  % of the leaf with attention > 0.25 — what the app
                                used to report as "affected area"
  lesion_pct                    measured diseased share of the leaf

Lesion masks are produced automatically (colour-based, see lesion_segmentation.py),
so they are pseudo ground truth, not hand annotations — keep that in mind when
reporting. Swap in hand-drawn masks (e.g. PlantSeg) via --masks for a stricter test.

Usage (from the project folder):
    python research/attention_lesion_study.py --data PlantVillage --per-class 40
    python research/analyze_attention_study.py
"""
import argparse
import csv
import glob
import os
import random
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

import disease_heatmap as dh  # noqa: E402
import lesion_segmentation as ls  # noqa: E402

MODELS = ["MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121"]
SIZE = 224


def _load_compat(p):
    """
    Load a model saved by a NEWER Keras than the one installed. Newer Keras stores extra
    BatchNormalization options ('renorm', ...) that older versions don't accept, so drop
    any option the installed BatchNormalization doesn't know about.
    """
    import inspect
    BN = tf.keras.layers.BatchNormalization
    allowed = set(inspect.signature(BN.__init__).parameters) | {"name", "trainable", "dtype"}
    original = BN.__dict__.get("from_config")

    @classmethod
    def from_config(cls, config):
        config = {k: v for k, v in config.items() if k in allowed or k not in
                  ("renorm", "renorm_clipping", "renorm_momentum")}
        return cls(**config)

    BN.from_config = from_config
    try:
        return tf.keras.models.load_model(p, compile=False)
    finally:
        if original is not None:
            BN.from_config = original
        else:
            del BN.from_config


def load_model(name):
    for ext in ("_best.keras", "_final.keras"):
        p = os.path.join(ROOT, "saved_models", name + ext)
        if os.path.exists(p):
            try:
                return tf.keras.models.load_model(p, compile=False)
            except TypeError:  # saved with a newer Keras (e.g. BatchNormalization 'renorm' options)
                return _load_compat(p)
    raise FileNotFoundError(name)


class CamEngine:
    """Grad-CAM and Grad-CAM++ from one gradient pass, using the app's (fixed) gradient model."""

    def __init__(self, model, name):
        target, backbone = dh._find_target_layer(model, dh.GRADCAM_LAYERS[name])
        self.dense, penult = dh._find_output_and_penultimate(model)
        self.forward = dh._build_grad_model(model, target, backbone, penult)

    def __call__(self, x):
        with tf.GradientTape() as tape:
            conv, pen = self.forward(x)
            tape.watch(conv)
            logits = tf.matmul(pen, self.dense.kernel) + self.dense.bias
            idx = int(tf.argmax(logits[0]))
            score = logits[:, idx]
        grads = tape.gradient(score, conv)[0].numpy()
        conv = conv[0].numpy()
        probs = tf.nn.softmax(logits)[0].numpy()

        # Grad-CAM: channel weights = mean gradient
        w = grads.mean(axis=(0, 1))
        cam = np.maximum((conv * w).sum(-1), 0)
        # Grad-CAM++ (same formulation as the app)
        pg = np.maximum(grads, 0)
        a_num = pg ** 2
        a_den = 2 * a_num + (conv * pg ** 3).sum(axis=(0, 1), keepdims=True) + 1e-8
        wpp = ((a_num / a_den) * pg).sum(axis=(0, 1))
        campp = np.maximum((conv * wpp).sum(-1), 0)

        def up(c):
            c = cv2.resize(c.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            return c / (c.max() + 1e-8)

        return idx, float(probs[idx]), up(cam), up(campp)


def metrics(cam, lesion, leaf):
    total = cam.sum() + 1e-8
    healthy = leaf & ~lesion
    bg = ~leaf
    out = {
        "e_lesion": float(cam[lesion].sum() / total),
        "e_healthy": float(cam[healthy].sum() / total),
        "e_bg": float(cam[bg].sum() / total),
        "cam_area_pct": float(((cam > 0.25) & leaf).sum() / max(1, leaf.sum()) * 100),
    }
    if lesion.sum() >= 20:
        out["focus_ratio"] = float(cam[lesion].mean() / (cam[leaf].mean() + 1e-8))
        y, x = np.unravel_index(np.argmax(cam), cam.shape)
        tol = cv2.dilate(lesion.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
        out["pointing_hit"] = int(tol[y, x])
        hot = cam > 0.5
        out["iou_05"] = float((hot & lesion).sum() / max(1, (hot | lesion).sum()))
    else:
        out.update(focus_ratio="", pointing_hit="", iou_05="")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "PlantVillage"))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "attention_study.csv"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    classes = [l.strip() for l in open(os.path.join(ROOT, "class_names.txt")) if l.strip()]
    random.seed(args.seed)
    files = []
    for c in classes:
        fs = sorted(glob.glob(os.path.join(args.data, c, "*")))
        files += random.sample(fs, min(args.per_class, len(fs))) if len(fs) > args.per_class else fs
    print(f"{len(files)} images")

    # Lesion + leaf masks once per image (model independent)
    masks = {}
    t0 = time.time()
    for f in files:
        rgb = np.array(Image.open(f).convert("RGB"))
        cls = os.path.basename(os.path.dirname(f))
        r = ls.detect_lesions(rgb, include_pale=any(k in cls for k in ("Late_blight", "Leaf_Mold")))
        lesion = cv2.resize(r["lesion_mask"], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
        leaf = cv2.resize(r["leaf_mask"], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
        masks[f] = (lesion & leaf, leaf, r["affected_pct"], r["reliable"])
    print(f"lesion masks: {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["image", "true_class", "healthy", "model", "method", "pred_class", "correct", "confidence",
              "lesion_pct", "lesion_reliable", "e_lesion", "e_healthy", "e_bg", "focus_ratio",
              "pointing_hit", "iou_05", "cam_area_pct"]
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for name in args.models.split(","):
            t0 = time.time()
            model = load_model(name)
            engine = CamEngine(model, name)
            for f in files:
                cls = os.path.basename(os.path.dirname(f))
                x = np.expand_dims(np.array(Image.open(f).convert("RGB").resize((SIZE, SIZE)), np.float32) / 255.0, 0)
                idx, conf, cam, campp = engine(x)
                lesion, leaf, pct, reliable = masks[f]
                for method, c in (("Grad-CAM", cam), ("Grad-CAM++", campp)):
                    row = {"image": os.path.relpath(f, args.data), "true_class": cls,
                           "healthy": int("healthy" in cls), "model": name, "method": method,
                           "pred_class": classes[idx], "correct": int(classes[idx] == cls),
                           "confidence": round(conf, 4), "lesion_pct": round(pct, 2),
                           "lesion_reliable": int(reliable)}
                    row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics(c, lesion, leaf).items()})
                    wr.writerow(row)
            print(f"{name}: {time.time() - t0:.0f}s")
            del model, engine
            tf.keras.backend.clear_session()
    print("saved", args.out)


if __name__ == "__main__":
    main()
