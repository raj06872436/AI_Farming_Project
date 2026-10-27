"""
LesionNet in the app: one forward pass gives the disease class AND a lesion map.

The class is computed only from features pooled over the pixels the network marked as
leaf / lesion (see research/lesion_net/lesionnet.py), so the lesion map shown to the user
is exactly the evidence the decision used — not an after-the-fact heatmap.
"""
import glob
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
sys.path.insert(0, os.path.join(BASE_DIR, "research", "lesion_net"))

NAME = "LesionNet"           # family name; also used to tag lesion results
MOBILE = "LesionNet-Mobile"
SIZE = 224
FILES = {NAME: "LesionNet_A3_full.keras", MOBILE: "LesionNet_A5_mobilenet.keras"}
# Measured on the 3,109-photo held-out test split (research/lesion_net/results/eval_summary.md).
# inference_ms: CPU estimate on the same scale as the classic models' registry numbers
# (LesionNet ≈ EfficientNetB0 cost, LesionNet-Mobile ≈ 1.26× MobileNetV2, see the paper).
METAS = {
    NAME: {"accuracy": 0.995, "grey_bg_accuracy": 0.987, "swap_bg_accuracy": 0.818, "size_mb": 94.2,
           "params": "4.9M", "inference_ms": 104,
           "description": "Finds the diseased spots first, then names the disease from those spots only."},
    MOBILE: {"accuracy": 0.960, "grey_bg_accuracy": 0.942, "swap_bg_accuracy": 0.732, "size_mb": 63.7,
             "params": "3.4M", "inference_ms": 79,
             "description": "A lighter LesionNet for slower computers and phones: same idea, a little less accurate."},
}
META = METAS[NAME]


def model_path(name=NAME):
    p = os.path.join(SAVED_MODELS_DIR, FILES.get(name, ""))
    return p if os.path.exists(p) else None


def available_models():
    return [n for n in (NAME, MOBILE) if model_path(n)]


def find_lesionnet_path():
    """Prefer the main model (A3_full); fall back to any other trained LesionNet."""
    preferred = model_path(NAME)
    if preferred:
        return preferred
    others = sorted(p for p in glob.glob(os.path.join(SAVED_MODELS_DIR, "LesionNet_*.keras"))
                    if "smoke" not in os.path.basename(p) and "A0" not in os.path.basename(p))
    return others[0] if others else None


def load(path=None):
    from lesionnet import load_lesionnet
    return load_lesionnet(path or find_lesionnet_path())


def predict(model, image, class_names):
    """
    image: PIL image. Returns a dict with the class prediction (same shape as the app's
    predict_image) plus the lesion map.
    """
    img = image.convert("RGB")
    arr = np.asarray(img.resize((SIZE, SIZE)), np.float32)[None] / 255.0
    out = model(arr, training=False)
    probs = np.asarray(out["cls"])[0].astype(np.float64)
    seg = np.asarray(out["seg"])[0].astype(np.float32)      # 56×56×3: background, leaf, lesion
    idx = int(np.argmax(probs))
    top5 = [{"class": class_names[i], "confidence": float(probs[i])}
            for i in np.argsort(probs)[::-1][:5] if i < len(class_names)]
    return {"idx": idx, "confidence": float(probs[idx]), "top5": top5, "probs": probs, "seg": seg, "arr": arr}


def _main_leaf(leaf_mask):
    """Keep the main leaf (plus pieces at least a quarter of its size); drop stray blobs such as a
    shadow or a fold of cloth that the outline head mistook for leaf."""
    leaf = leaf_mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(leaf, 8)
    if n > 2:
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = np.flatnonzero(areas >= 0.25 * areas.max()) + 1
        leaf = np.isin(lab, keep).astype(np.uint8)
    return leaf


def lesion_result(image, seg, threshold=0.5, confidence=None):
    """
    Turn the 56×56 segmentation into full-resolution masks and the same fields the
    colour-based analysis returns (affected_pct, regions, leaf_mask, reliable, quality_notes).
    """
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    # Work at a display size that keeps big photos fast
    scale = min(1.0, 900 / max(h, w))
    dh, dw = max(1, int(h * scale)), max(1, int(w * scale))
    view = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()
    up = cv2.resize(seg, (dw, dh), interpolation=cv2.INTER_LINEAR)
    lesion_p = up[..., 2]
    leaf_p = up[..., 1] + up[..., 2]
    leaf_mask = _main_leaf(leaf_p > 0.5)
    lesion_mask = ((lesion_p > threshold) & (leaf_mask > 0)).astype(np.uint8)

    # Diseased share of the (cleaned) leaf, from the soft probabilities
    leaf_px = max(int(leaf_mask.sum()), 1)
    affected_pct = float(np.clip(lesion_p[leaf_mask > 0].sum() / leaf_px * 100, 0, 100))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(lesion_mask, 8)
    regions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < leaf_px * 0.002:
            continue
        conf = float(lesion_p[labels == i].mean())
        regions.append({"area_pct": area / leaf_px * 100, "confidence": conf,
                        "severity": "severe" if conf > 0.8 else "moderate" if conf > 0.65 else "mild"})
    regions.sort(key=lambda r: -r["area_pct"])
    for k, r in enumerate(regions, 1):
        r["id"] = k

    # When to warn: LesionNet learned from single tomato, potato and pepper leaves on plain
    # backgrounds. Other plants, dried leaves, shadows or patterned cloth are outside that.
    notes = []
    if leaf_mask.mean() < 0.08:
        notes.append("very little leaf was found in the photo")
    if affected_pct > 60:
        notes.append("almost the whole leaf is marked, which usually means the photo is unlike the training "
                     "photos (another plant, a dried leaf, strong shadows or a patterned background)")
    if confidence is not None and confidence < 0.85:
        notes.append(f"the AI is only {confidence:.0%} sure of the disease")
    return {
        "affected_pct": affected_pct, "regions": regions, "leaf_mask": leaf_mask * 255,
        "lesion_mask": lesion_mask, "lesion_prob": lesion_p, "view": view,
        "reliable": not notes, "quality_notes": notes, "breakdown": {}, "source": NAME,
    }


def make_overlay(result, alpha=0.6):
    """Red = diseased tissue (stronger red = surer), background dimmed, spot outlines drawn."""
    view = result["view"].astype(np.float32)
    p = np.clip((result["lesion_prob"] - 0.3) / 0.5, 0, 1) * (result["leaf_mask"] > 0)
    red = np.array([220, 38, 38], np.float32)
    out = view * (1 - alpha * p[..., None]) + red * (alpha * p[..., None])
    bg = result["leaf_mask"] == 0
    out[bg] = out[bg] * 0.55
    out = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(result["lesion_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thick = max(1, int(round(min(out.shape[:2]) / 320)))
    cv2.drawContours(out, contours, -1, (255, 255, 255), thick + 1)
    cv2.drawContours(out, contours, -1, (153, 27, 27), thick)
    return out


def heat_image(result):
    """Lesion probability as a colour map (for the 'Disease map' tab)."""
    import matplotlib.cm as cm
    return (cm.inferno(result["lesion_prob"])[:, :, :3] * 255).astype(np.uint8)
