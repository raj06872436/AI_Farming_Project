"""
Qualitative figure: photo, measured lesions, and Grad-CAM++ of each model for a few leaves.
Usage: python research/make_example_panel.py [PlantVillage folder]
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

import lesion_segmentation as ls  # noqa: E402
from attention_lesion_study import MODELS, CamEngine, load_model  # noqa: E402

CLASSES = ["Potato___Early_blight", "Tomato_Septoria_leaf_spot", "Tomato_Bacterial_spot",
           "Tomato__Tomato_YellowLeaf__Curl_Virus"]


def main(data=os.path.join(ROOT, "PlantVillage")):
    files = [sorted(glob.glob(os.path.join(data, c, "*")))[3] for c in CLASSES]
    imgs = [np.array(Image.open(f).convert("RGB").resize((224, 224))) for f in files]
    lesions = []
    for f in files:
        r = ls.detect_lesions(np.array(Image.open(f).convert("RGB")))
        lesions.append(cv2.resize(r["lesion_mask"], (224, 224), interpolation=cv2.INTER_NEAREST) > 0)
    cams = {}
    for m in MODELS:
        model = load_model(m)
        eng = CamEngine(model, m)
        cams[m] = [eng(np.expand_dims(im.astype(np.float32) / 255.0, 0))[3] for im in imgs]
        tf.keras.backend.clear_session()

    cols = 2 + len(MODELS)
    fig, axes = plt.subplots(len(files), cols, figsize=(2.1 * cols, 2.1 * len(files)))
    for i, (im, les) in enumerate(zip(imgs, lesions)):
        axes[i, 0].imshow(im)
        ov = im.copy()
        ov[les] = (0.45 * ov[les] + 0.55 * np.array([220, 38, 38])).astype(np.uint8)
        axes[i, 1].imshow(ov)
        for j, m in enumerate(MODELS):
            axes[i, 2 + j].imshow(im)
            axes[i, 2 + j].imshow(cams[m][i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[i, 2 + j].contour(les, levels=[0.5], colors="white", linewidths=0.7)
        axes[i, 0].set_ylabel(CLASSES[i].replace("___", " ").replace("__", " ").replace("_", " "), fontsize=7)
    titles = ["Photo", "Measured lesions"] + [f"{m}\nGrad-CAM++" for m in MODELS]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(HERE, "results", "fig_examples.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
