"""
Step 3 — compare LesionNet with the four existing classifiers on the held-out test split.

Tests (all on the same test images):
  1. Clean accuracy / macro-F1.
  2. Background-only: the leaf is blanked out, only the background is left.
     A model that truly uses the leaf should drop to chance (1/15 = 6.7 %).
     Accuracy above chance = it learned shortcuts from the background.
  3. Background swap: the leaf is kept but its background is replaced
     (a) with flat grey, (b) with the photo of a leaf from a DIFFERENT class.
     A robust model keeps its accuracy.
  4. Attention on disease (diseased leaves only): share of the explanation on lesions /
     healthy leaf / background, lesion focus ratio, pointing game, IoU.
     Existing models are explained with Grad-CAM++; LesionNet with its own lesion map
     (the map its decision is computed from).
     NOTE: lesion masks are the same automatic pseudo-labels LesionNet was trained to
     imitate, so this test favours LesionNet. Tests 2 and 3 do not use lesion masks at
     all — they are the fair comparison. Hand-drawn masks (e.g. PlantSeg) are needed for
     a fully independent lesion test.
  5. Severity: LesionNet's predicted diseased share of the leaf vs the measured one,
     and on healthy leaves (should be ~0).

Usage (WSL, project folder):
    python research/lesion_net/evaluate.py
    python research/lesion_net/evaluate.py --lesionnets saved_models/LesionNet_*.keras --limit-per-class 60
"""
import argparse
import csv
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from attention_lesion_study import CamEngine, load_model, metrics  # noqa: E402
from lesionnet import load_lesionnet  # noqa: E402
from train_lesionnet import _abs, read_manifest  # noqa: E402

SIZE = 224
BASELINES = ["MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121"]


def load_test(manifest, limit):
    rows = read_manifest(manifest, "test", limit)
    imgs, leafs, lesions = [], [], []
    for r in rows:
        img = np.array(Image.open(_abs(r["image"])).convert("RGB").resize((SIZE, SIZE)), np.uint8)
        lbl = np.array(Image.open(_abs(r["label"])).resize((SIZE, SIZE), Image.NEAREST))
        imgs.append(img)
        leafs.append(lbl > 0)
        lesions.append(lbl == 2)
    return rows, np.stack(imgs), np.stack(leafs), np.stack(lesions)


class Variants:
    """
    Builds test-set variants batch by batch (keeps memory small):
      clean   – the photo
      bg_only – leaf blanked to grey, only the background left (leaf silhouette still visible)
      bg_fill – leaf area filled with this photo's own background pixels: no leaf, no silhouette
      grey_bg – leaf kept, background flat grey
      swap_bg – leaf kept, background taken from a photo of a DIFFERENT class
    """
    KINDS = ("clean", "bg_only", "bg_fill", "grey_bg", "swap_bg")

    def __init__(self, imgs, leafs, ys, seed=0):
        rng = np.random.default_rng(seed)
        self.imgs, self.leafs = imgs, leafs
        self.other = np.array([rng.choice(np.flatnonzero(ys != y)) for y in ys])

    def __len__(self):
        return len(self.imgs)

    def batch(self, kind, i, j):
        x = self.imgs[i:j].astype(np.float32) / 255.0
        if kind == "clean":
            return x
        leaf3 = self.leafs[i:j, ..., None]
        if kind == "bg_only":
            return np.where(leaf3, 0.5, x).astype(np.float32)
        if kind == "bg_fill":
            import cv2
            out = x.copy()
            rng = np.random.default_rng(i)
            for b in range(len(out)):
                leaf = self.leafs[i + b]
                bg_px = out[b][~leaf]
                if len(bg_px) < 50:
                    out[b][:] = 0.5
                    continue
                out[b][leaf] = bg_px[rng.integers(0, len(bg_px), int(leaf.sum()))]
                smooth = cv2.blur(out[b], (7, 7))
                out[b][leaf] = smooth[leaf]
            return out
        if kind == "grey_bg":
            return np.where(leaf3, x, 0.5).astype(np.float32)
        other = self.imgs[self.other[i:j]].astype(np.float32) / 255.0
        return np.where(leaf3, x, other).astype(np.float32)


def macro_f1(y, p, k):
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def predict_cls(model, sets, kind, is_lesionnet, batch=16):
    out = []
    for i in range(0, len(sets), batch):
        o = model(sets.batch(kind, i, i + batch), training=False)
        out.append((o["cls"] if is_lesionnet else o).numpy())
    return np.concatenate(out)


def spearman(a, b):
    import pandas as pd
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def evaluate_model(name, model, is_lesionnet, rows, sets, leafs, lesions, ys, k):
    res = {"Model": name}
    for key in Variants.KINDS:
        p = predict_cls(model, sets, key, is_lesionnet).argmax(1)
        res[f"acc_{key}"] = float(np.mean(p == ys))
        if key == "clean":
            res["macro_f1"] = macro_f1(ys, p, k)

    # Attention on disease
    att = []
    sev_pred, sev_true, sev_healthy = [], [], []
    engine = None if is_lesionnet else CamEngine(model, name)
    for i, r in enumerate(rows):
        x = sets.batch("clean", i, i + 1)
        diseased_ok = r["healthy"] == "0" and float(r["seg_weight"]) > 0 and float(r["lesion_pct"]) >= 1
        if is_lesionnet:
            seg = model(x, training=False)["seg"][0].numpy()
            les_map = tf.image.resize(seg[..., 2:3], (SIZE, SIZE)).numpy()[..., 0]
            leaf_p = seg[..., 1].sum() + seg[..., 2].sum()
            pct = float(seg[..., 2].sum() / max(leaf_p, 1e-6) * 100)
            if r["healthy"] == "1":
                sev_healthy.append(pct)
            elif diseased_ok:
                sev_pred.append(pct)
                sev_true.append(float(r["lesion_pct"]))
            explanation = les_map / (les_map.max() + 1e-8)
        else:
            if not diseased_ok:
                continue
            explanation = engine(x)[3]
        if diseased_ok:
            att.append(metrics(explanation, lesions[i], leafs[i]))
    for key in ("e_lesion", "e_healthy", "e_bg", "focus_ratio", "pointing_hit", "iou_05"):
        vals = [a[key] for a in att if a[key] != ""]
        res[key] = float(np.mean(vals)) if vals else np.nan
    if is_lesionnet:
        res["severity_spearman"] = spearman(sev_pred, sev_true) if len(sev_pred) > 2 else np.nan
        res["severity_mae_pct"] = float(np.mean(np.abs(np.array(sev_pred) - np.array(sev_true)))) if sev_pred else np.nan
        res["lesion_pct_on_healthy"] = float(np.mean(sev_healthy)) if sev_healthy else np.nan
    res["explanation"] = "own lesion map" if is_lesionnet else "Grad-CAM++"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "pseudo_masks", "manifest.csv"))
    ap.add_argument("--lesionnets", nargs="*", default=None)
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    classes = [l.strip() for l in open(os.path.join(ROOT, "class_names.txt")) if l.strip()]
    rows, imgs, leafs, lesions = load_test(args.manifest, args.limit_per_class)
    ys = np.array([classes.index(r["class"]) for r in rows])
    sets = Variants(imgs, leafs, ys)
    print(f"test images: {len(rows)}")

    nets = args.lesionnets if args.lesionnets is not None else sorted(
        glob.glob(os.path.join(ROOT, "saved_models", "LesionNet_*.keras")))
    results = []
    for name in [b for b in args.baselines.split(",") if b]:
        t0 = time.time()
        model = load_model(name)
        results.append(evaluate_model(name, model, False, rows, sets, leafs, lesions, ys, len(classes)))
        print(f"{name}: {time.time() - t0:.0f}s  acc {results[-1]['acc_clean']:.3f}")
        tf.keras.backend.clear_session()
    for path in nets:
        t0 = time.time()
        model = load_lesionnet(path)
        name = os.path.basename(path).replace(".keras", "")
        results.append(evaluate_model(name, model, True, rows, sets, leafs, lesions, ys, len(classes)))
        print(f"{name}: {time.time() - t0:.0f}s  acc {results[-1]['acc_clean']:.3f}")
        tf.keras.backend.clear_session()

    os.makedirs(args.out, exist_ok=True)
    keys = ["Model", "explanation", "acc_clean", "macro_f1", "acc_bg_only", "acc_bg_fill", "acc_grey_bg", "acc_swap_bg",
            "e_lesion", "e_healthy", "e_bg", "focus_ratio", "pointing_hit", "iou_05",
            "severity_spearman", "severity_mae_pct", "lesion_pct_on_healthy"]
    with open(os.path.join(args.out, "eval_summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in results:
            wr.writerow({k: r.get(k, "") for k in keys})
    nice = {"acc_clean": "Accuracy", "macro_f1": "Macro-F1", "acc_bg_only": "Background + leaf silhouette acc.",
            "acc_bg_fill": "Background-only acc. (chance 0.067)",
            "acc_grey_bg": "Acc. grey background", "acc_swap_bg": "Acc. swapped background",
            "e_lesion": "Explanation on lesions", "e_bg": "Explanation on background", "focus_ratio": "Lesion focus",
            "pointing_hit": "Pointing hit", "iou_05": "IoU", "severity_spearman": "Severity ρ",
            "lesion_pct_on_healthy": "Lesion % on healthy"}
    with open(os.path.join(args.out, "eval_summary.md"), "w") as fh:
        cols = ["Model", "explanation"] + list(nice)
        fh.write("| " + " | ".join(["Model", "Explanation"] + list(nice.values())) + " |\n")
        fh.write("|" + "---|" * len(cols) + "\n")
        for r in results:
            cells = []
            for c in cols:
                v = r.get(c, "")
                cells.append(f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else ("—" if v == "" or (isinstance(v, float) and np.isnan(v)) else str(v)))
            fh.write("| " + " | ".join(cells) + " |\n")
    print(open(os.path.join(args.out, "eval_summary.md")).read())
    make_figure(results, args.out)


def make_figure(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["Model"].replace("LesionNet_", "LesionNet\n") for r in results]
    panels = [("acc_clean", "Accuracy (clean)"), ("acc_bg_fill", "Accuracy from background only\n(lower is better, chance 0.067)"),
              ("acc_swap_bg", "Accuracy with swapped background"), ("e_lesion", "Explanation share on lesions")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), layout="constrained")
    colors = ["#9AA59C" if not r["Model"].startswith("LesionNet") else "#2E7D32" for r in results]
    for ax, (key, title) in zip(axes, panels):
        vals = [r.get(key, np.nan) for r in results]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)), names, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        if key == "acc_bg_fill":
            ax.axhline(1 / 15, color="black", ls="--", lw=1)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(out, "fig_lesionnet_vs_baselines.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
