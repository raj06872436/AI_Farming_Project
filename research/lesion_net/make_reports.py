"""
Standard evaluation reports for LesionNet, in the same format as the four classic models
(reports/metrics, reports/confusion_matrix, reports/roc_curves, reports/graphs), so the
Research Dashboard can show them side by side.

Run (project folder, after training):
    python research/lesion_net/make_reports.py
    python research/lesion_net/make_reports.py --data-root /path/with/PlantVillage
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (auc, average_precision_score, classification_report, confusion_matrix,  # noqa: E402
                             precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize  # noqa: E402

from lesionnet import load_lesionnet  # noqa: E402

MODELS = {"LesionNet": "LesionNet_A3_full", "LesionNet-Mobile": "LesionNet_A5_mobilenet"}
REPORTS = os.path.join(ROOT, "reports")


def load_test(manifest, data_root, classes):
    rows = [r for r in csv.DictReader(open(manifest)) if r["split"] == "test"]
    xs, ys = [], []
    for r in rows:
        p = r["image"] if os.path.isabs(r["image"]) else os.path.join(data_root, r["image"])
        xs.append(np.asarray(Image.open(p).convert("RGB").resize((224, 224)), np.uint8))
        ys.append(classes.index(r["class"]))
    return np.stack(xs), np.array(ys)


def predict(model, xs, batch=32):
    out = []
    for i in range(0, len(xs), batch):
        out.append(np.asarray(model(xs[i:i + batch].astype(np.float32) / 255.0, training=False)["cls"], np.float32))
    return np.concatenate(out)


def plot_confusion(cm, classes, name, path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="YlGnBu")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)), classes, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=17, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_curves(ys, probs, classes, name, roc_path, pr_path):
    yb = label_binarize(ys, classes=range(len(classes)))
    cmap = plt.get_cmap("tab20")
    for kind, path in (("roc", roc_path), ("pr", pr_path)):
        fig, ax = plt.subplots(figsize=(9, 7))
        for c, cname in enumerate(classes):
            if kind == "roc":
                fpr, tpr, _ = roc_curve(yb[:, c], probs[:, c])
                ax.plot(fpr, tpr, color=cmap(c), lw=1.3, label=f"{cname} (AUC={auc(fpr, tpr):.3f})")
            else:
                pr, rc, _ = precision_recall_curve(yb[:, c], probs[:, c])
                ax.plot(rc, pr, color=cmap(c), lw=1.3,
                        label=f"{cname} (AP={average_precision_score(yb[:, c], probs[:, c]):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curves — {name}", fontweight="bold")
        else:
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curves — {name}", fontweight="bold")
        ax.legend(fontsize=6.5, loc="lower right" if kind == "roc" else "lower left")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


def plot_history(hist_path, name, path):
    h = json.load(open(hist_path))
    H = h["history"]
    ep = np.arange(1, len(H["cls_accuracy"]) + 1)
    fig, ax = plt.subplots(1, 3 if "seg_loss" in H else 2, figsize=(15, 4.5))
    ax[0].plot(ep, H["cls_accuracy"], label="train")
    ax[0].plot(ep, H["val_cls_accuracy"], label="validation")
    ax[0].set_title("Classification accuracy")
    ax[1].plot(ep, H["cls_loss"], label="train")
    ax[1].plot(ep, H["val_cls_loss"], label="validation")
    ax[1].set_title("Classification loss")
    if "seg_loss" in H and len(ax) > 2:
        ax[2].plot(ep, H["seg_loss"], label="train")
        ax[2].plot(ep, H["val_seg_loss"], label="validation")
        ax[2].set_title("Segmentation loss (lesion map)")
    for a in ax:
        a.axvline(3.5, color="grey", ls=":", lw=1)
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(f"{name} — Training History (frozen backbone for epochs 1–3, then fine-tuning; "
                 f"{h.get('minutes', '?')} min on GPU)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "pseudo_masks", "manifest.csv"))
    ap.add_argument("--data-root", default=ROOT)
    args = ap.parse_args()
    classes = [l.strip() for l in open(os.path.join(ROOT, "class_names.txt")) if l.strip()]
    for d in ("metrics", "confusion_matrix", "roc_curves", "graphs"):
        os.makedirs(os.path.join(REPORTS, d), exist_ok=True)
    xs, ys = load_test(args.manifest, args.data_root, classes)
    print("test images", len(xs), flush=True)

    rows = []
    for name, tag in MODELS.items():
        path = os.path.join(ROOT, "saved_models", f"{tag}.keras")
        if not os.path.exists(path):
            continue
        model = load_lesionnet(path)
        probs = predict(model, xs)
        pred = probs.argmax(1)
        p, r, f1, _ = precision_recall_fscore_support(ys, pred, average="weighted", zero_division=0)
        yb = label_binarize(ys, classes=range(len(classes)))
        row = {"Model": name, "Accuracy": float(np.mean(pred == ys)), "Precision": float(p), "Recall": float(r),
               "F1 Score": float(f1), "AUC": float(roc_auc_score(yb, probs, average="macro")),
               "Model Size (MB)": round(os.path.getsize(path) / 2 ** 20, 2), "Total Params": int(model.count_params()),
               "Test Images": len(ys)}
        rows.append(row)
        rep = classification_report(ys, pred, target_names=classes, digits=2, zero_division=0)
        with open(os.path.join(REPORTS, "metrics", f"{name}_classification_report.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"Classification Report — {name}\n{'=' * 60}\n\n{rep}\n")
        plot_confusion(confusion_matrix(ys, pred), classes, name,
                       os.path.join(REPORTS, "confusion_matrix", f"{name}_confusion_matrix.png"))
        plot_curves(ys, probs, classes, name, os.path.join(REPORTS, "roc_curves", f"{name}_roc_curves.png"),
                    os.path.join(REPORTS, "roc_curves", f"{name}_pr_curves.png"))
        hist = os.path.join(ROOT, "saved_models", f"{tag}_history.json")
        if os.path.exists(hist):
            plot_history(hist, name, os.path.join(REPORTS, "graphs", f"{name}_training_history.png"))
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        tf.keras.backend.clear_session()

    with open(os.path.join(REPORTS, "metrics", "lesionnet_metrics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
