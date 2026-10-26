"""
AGRI-X AI — Leaf & lesion segmentation (how much of the leaf is actually diseased).

Why this exists
---------------
Grad-CAM tells us *where the model looked*, on a coarse 7×7 grid. Measuring the
"affected area" from it gave ~40–55 % on every leaf — even perfectly healthy ones —
because the model looks at most of the leaf. Here we measure the diseased tissue
itself, pixel by pixel, from colour:

  1. Find the leaf: separate it from the background using the colours seen at the
     image border (background) versus leaf-like colours.
  2. Learn what *healthy* tissue looks like on this particular leaf (its typical
     green), so lighting and variety differences don't matter.
  3. Mark tissue that departs from that healthy green:
       • necrotic  — brown / dark / reddish spots and dead tissue   (severe)
       • chlorotic — yellowing, mottling, mosaic                    (mild)
       • pale      — whitish mould / powdery or bleached patches    (moderate)
  4. Clean up (remove specks, ignore the leaf's outer rim) and report the
     affected area as a percentage of the LEAF, not of the photo.

Plain-language analogy: the old method was like asking "where did the doctor's eyes
go?" — this one is like actually measuring the size of the rash.
"""
import cv2
import numpy as np

ANALYSIS_MAX_SIDE = 640   # analyse at up to this resolution (fast, still sharp)

# Lesion classes in the returned label map
HEALTHY, CHLOROTIC, PALE, NECROTIC = 0, 1, 2, 3
CLASS_NAMES = {CHLOROTIC: "yellowing", PALE: "pale / mould", NECROTIC: "dead / spotted"}


def _resize_max(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img, 1.0
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s


def _odd(n):
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


# ─────────────────────────────────────────────────────────────────────
# 1. Leaf segmentation
# ─────────────────────────────────────────────────────────────────────
def segment_leaf(rgb):
    """
    Return a binary leaf mask (uint8 0/255) for an RGB uint8 image.

    Background colours are learned from the image border; border colours that look
    like plant tissue (close-up photos where the leaf fills the frame) are NOT
    treated as background.
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, S, V = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)

    # Plant-tissue colours: green / yellow-green / brown with some saturation
    plant_like = (((H >= 8) & (H <= 95) & (S >= 35) & (V >= 30))
                  | ((H < 8) & (S >= 70) & (V >= 30) & (V <= 200)))   # red-brown lesions

    # Background model from a band around the border
    b = max(2, int(0.04 * min(h, w)))
    border = np.zeros((h, w), bool)
    border[:b, :] = border[-b:, :] = True
    border[:, :b] = border[:, -b:] = True
    # Learn background from border pixels that are not green leaf (beige/brown tables are
    # "plant-coloured" too, so only clearly green pixels are left out of the model)
    border_green = (H >= 30) & (H <= 95) & (S >= 40)
    bg_pixels = lab[border & ~border_green]

    background = np.zeros((h, w), bool)
    chroma_dist = None
    if len(bg_pixels) > 0.15 * border.sum():
        k = 3 if len(bg_pixels) > 50 else 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(bg_pixels.astype(np.float32), k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
        flat = lab.reshape(-1, 3)
        d = np.min(np.stack([np.linalg.norm(flat - c, axis=1) for c in centers]), axis=0).reshape(h, w)
        background = d < 18
        ab = lab[..., 1:].reshape(-1, 2)
        chroma_dist = np.min(np.stack([np.linalg.norm(ab - c[1:], axis=1) for c in centers]), axis=0).reshape(h, w)
    # Very dark or very bright, unsaturated pixels are background too
    background |= (V < 25) | ((S < 25) & (V > 200))

    # Only plant-coloured pixels seed the leaf (shadows on the table are grey, not
    # plant-coloured, so they stay out); dark spots inside the leaf come back via hole filling.
    mask = plant_like & ~(background & (S < 60))
    # Pale / grey-green leaves (e.g. hairy undersides) have little saturation but are
    # still greener than the background: Otsu threshold on "greenness" (−a* in LAB).
    greenness = 128.0 - lab[..., 1]
    g8 = np.clip((greenness + 20) * 4, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    green_mask = (g8 > thr) & (greenness > 3) & (V >= 30)
    mask = (mask & ~(background & ~green_mask)) | green_mask
    mask = mask.astype(np.uint8) * 255

    k = _odd(min(h, w) * 0.02)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))

    # Refine with GrabCut: it learns this photo's leaf vs background colours, which
    # recovers pale / grey-green leaf areas and drops coloured table tops.
    mask = _grabcut_refine(rgb, mask, background, plant_like)

    # Dead (dark brown / black) tissue is easily mistaken for background. Add back
    # regions whose colour tint differs from the background and that touch the leaf.
    # (Shadows keep the background's tint, so they are not added.)
    if chroma_dist is not None and mask.any():
        dead_like = (chroma_dist > 10) & (V >= 15) & ~((S < 25) & (V > 200)) & ~background
        cand = (dead_like | (mask > 0)).astype(np.uint8)
        n_c, lab_c = cv2.connectedComponents(cand, connectivity=8)
        touching = np.unique(lab_c[mask > 0])
        grown = np.isin(lab_c, touching[touching > 0])
        if grown.sum() < 1.8 * (mask > 0).sum():        # don't let it swallow the background
            mask = (grown.astype(np.uint8) * 255)

    # Keep the main leaf (plus any other large pieces of the same leaf)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.full((h, w), 255, np.uint8)  # nothing found → assume the leaf fills the photo
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = areas.max()
    keep = [i + 1 for i, a in enumerate(areas) if a >= 0.15 * biggest]
    mask = np.isin(labels, keep).astype(np.uint8) * 255

    # Fill holes (lesions with odd colours inside the leaf belong to the leaf)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    if filled.sum() < 0.03 * 255 * h * w:      # implausibly small → leaf fills the photo
        return np.full((h, w), 255, np.uint8)
    return filled


def _grabcut_refine(rgb, coarse, background, plant_like):
    """GrabCut seeded by the colour-rule mask. Returns uint8 0/255 (falls back to `coarse`)."""
    h, w = rgb.shape[:2]
    if coarse.sum() < 0.02 * 255 * h * w:
        return coarse
    s = min(1.0, 320 / max(h, w))
    small = cv2.resize(rgb, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    sh, sw = small.shape[:2]
    rs = lambda m: cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    c = rs(coarse > 0)
    bg = rs(background) & ~rs(plant_like)
    k = _odd(min(sh, sw) * 0.06)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sure_fg = cv2.erode(c.astype(np.uint8), ker) > 0
    far_bg = cv2.dilate(c.astype(np.uint8), ker) == 0

    gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    gc[c] = cv2.GC_PR_FGD
    gc[bg & far_bg] = cv2.GC_BGD
    b = max(1, int(0.02 * min(sh, sw)))
    edge = np.zeros((sh, sw), bool)
    edge[:b, :] = edge[-b:, :] = True
    edge[:, :b] = edge[:, -b:] = True
    gc[edge & ~c] = cv2.GC_BGD
    gc[sure_fg] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any() or not ((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD)).any():
        return coarse
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse
    out = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    out = (out > 127).astype(np.uint8) * 255
    k2 = _odd(min(h, w) * 0.015)
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))


# ─────────────────────────────────────────────────────────────────────
# 2–4. Lesion detection
# ─────────────────────────────────────────────────────────────────────
def detect_lesions(rgb, leaf_mask=None, include_pale=False):
    """
    Detect diseased tissue inside the leaf.

    Returns dict:
        lesion_mask : uint8 (H, W) 0/255 — all diseased tissue
        label_map   : uint8 (H, W)       — HEALTHY / CHLOROTIC / PALE / NECROTIC
        leaf_mask   : uint8 (H, W) 0/255
        affected_pct: float — diseased share of the leaf area (0–100)
        breakdown   : {"yellowing": %, "pale / mould": %, "dead / spotted": %} of leaf
        regions     : list of per-spot stats (largest first)
    """
    h, w = rgb.shape[:2]
    if leaf_mask is None:
        leaf_mask = segment_leaf(rgb)
    leaf = leaf_mask > 0
    leaf_px = int(leaf.sum())
    empty = {"lesion_mask": np.zeros((h, w), np.uint8), "label_map": np.zeros((h, w), np.uint8),
             "leaf_mask": leaf_mask, "affected_pct": 0.0, "breakdown": {}, "regions": [],
             "reliable": False, "quality_notes": ["no leaf could be found in the photo"]}
    if leaf_px < 50:
        return empty

    # Light denoise so JPEG noise and leaf texture don't look like spots
    smooth = cv2.bilateralFilter(rgb, d=5, sigmaColor=30, sigmaSpace=5)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, S, V = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    A, B = lab[..., 1] - 128.0, lab[..., 2] - 128.0

    # Search area: the leaf minus a thin outer rim (edges are often shadowed)
    rim = _odd(min(h, w) * 0.025)
    inner = cv2.erode(leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim, rim))) > 0
    if inner.sum() < 0.5 * leaf_px:
        inner = leaf

    # Healthy reference = clearly green, well-lit tissue of THIS leaf
    green = inner & (H >= 34) & (H <= 88) & (S >= 50) & (V >= 50)
    if green.sum() < 0.05 * leaf_px:   # very sick / unusual leaf → most-green third of the leaf
        thresh = np.percentile(A[inner], 33)
        green = inner & (A <= thresh)
    ref_v = float(np.median(V[green]))
    ref_s = float(np.median(S[green]))
    ref_a = float(np.median(A[green]))
    ref_b = float(np.median(B[green]))

    # How far each pixel's colour (a*, b*) sits from healthy green
    da = A - ref_a                       # + → redder / browner than healthy
    db = B - ref_b                       # + → yellower than healthy

    necrotic = inner & (
        ((H < 20) | (H > 165)) & (S >= 45) & (V <= 0.92 * ref_v)       # brown / red-brown
        | ((V < 0.50 * ref_v) & ((H < 30) | (H > 95) | (S < 45)))         # dark / black spots (not shaded green)
        | ((da > 14) & (V < 0.95 * ref_v))                                 # clearly browner than leaf
    )
    ref_h = float(np.median(H[green]))
    chlorotic = inner & ~necrotic & (
        ((H >= 18) & (H < 31) & (S >= 70) & (V >= 0.75 * ref_v))         # yellow
        | ((da > 7) & (db > 4) & (H < 34))                                 # yellow-brown tint
        | ((H < ref_h - 10) & (S >= 45) & (V >= 0.70 * ref_v))            # clearly yellower than this leaf
    )
    # Whitish mould. Off by default: on hairy or shiny leaves glare looks identical, so it
    # is only used for diseases that actually produce white growth (late blight, leaf mould).
    pale = inner & ~necrotic & ~chlorotic & (S < 0.45 * ref_s) & (V > 0.80 * ref_v) & (V < 250)
    if not include_pale:
        pale[:] = False

    labels = np.zeros((h, w), np.uint8)
    labels[pale] = PALE
    labels[chlorotic] = CHLOROTIC
    labels[necrotic] = NECROTIC

    # Clean-up: remove specks and close tiny gaps inside spots
    lesion = (labels > 0).astype(np.uint8) * 255
    k1 = _odd(min(h, w) * 0.006)
    k2 = _odd(min(h, w) * 0.012)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
    lesion = cv2.bitwise_and(lesion, inner.astype(np.uint8) * 255)

    n, cc, stats, cents = cv2.connectedComponentsWithStats(lesion, 8)
    min_area = max(6, 0.0008 * leaf_px)
    # Pixels along the leaf outline — used to spot shadows hugging the edge
    edge_band = cv2.dilate(leaf_mask, np.ones((3, 3), np.uint8), iterations=rim + 2) & ~cv2.erode(
        leaf_mask, np.ones((3, 3), np.uint8), iterations=rim + 2)
    edge_band = edge_band > 0
    # Colour of whatever is just outside the leaf (table, soil, shadow) — strips along
    # the outline that match it are background bleeding into the mask, not disease
    outside = (cv2.dilate(leaf_mask, np.ones((3, 3), np.uint8), iterations=rim * 2 + 4) > 0) & ~leaf
    out_lab = np.median(lab[outside], axis=0) if outside.sum() > 30 else None
    keep = np.zeros(n, bool)
    regions = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp = cc == i
        kinds = labels[comp]
        nec_share = float(np.mean(kinds == NECROTIC))
        # A pale patch on its own is often just glare — keep only sizeable ones
        if np.all(kinds[kinds > 0] == PALE) and area < 0.01 * leaf_px:
            continue
        # Dark, grey strips along the leaf outline are cast shadows, not disease
        on_edge = float(np.mean(edge_band[comp]))
        if on_edge > 0.45 and np.mean(S[comp]) < 60 and np.mean(V[comp]) < 0.75 * ref_v:
            continue
        if on_edge > 0.35 and out_lab is not None:
            med = np.median(lab[comp], axis=0)
            # same colour as the surface (or its shadow: same a*/b*, lower lightness)
            if np.linalg.norm(med - out_lab) < 22 or np.linalg.norm(med[1:] - out_lab[1:]) < 9:
                continue
        keep[i] = True
        regions.append({
            "area_px": int(area),
            "area_pct": float(area / leaf_px * 100),
            "center": (int(cents[i][0]), int(cents[i][1])),
            "severity": "severe" if nec_share > 0.5 else "moderate" if nec_share > 0.15 else "mild",
            "mean_intensity": nec_share,
            "max_intensity": min(1.0, nec_share + 0.3),
        })
    lesion = keep[cc].astype(np.uint8) * 255
    labels[lesion == 0] = HEALTHY
    # Fill closed gaps inside kept spots with the spot's dominant kind
    gap = (lesion > 0) & (labels == HEALTHY)
    labels[gap] = NECROTIC if regions and np.mean([r["severity"] == "severe" for r in regions]) > 0.5 else CHLOROTIC

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    for i, r in enumerate(regions, 1):
        r["id"] = i

    affected = float((lesion > 0).sum() / leaf_px * 100)
    breakdown = {name: float((labels == code).sum() / leaf_px * 100) for code, name in CLASS_NAMES.items()}

    # How much to trust the number: a leaf that fills almost none (or all) of the photo,
    # or very little healthy green to compare against, makes the measurement shaky.
    leaf_share = leaf_px / float(h * w)
    notes = []
    if leaf_share < 0.12:
        notes.append("the leaf is small in the photo")
    if leaf_share > 0.97:
        notes.append("the leaf fills the whole photo, so its edges are not visible")
    if green.sum() < 0.15 * leaf_px:
        notes.append("very little healthy tissue is left to compare against")
    return {"lesion_mask": lesion, "label_map": labels, "leaf_mask": leaf_mask,
            "affected_pct": affected, "breakdown": breakdown, "regions": regions,
            "reliable": not notes, "quality_notes": notes}


# ─────────────────────────────────────────────────────────────────────
# Overlay for display
# ─────────────────────────────────────────────────────────────────────
_COLORS = {CHLOROTIC: (250, 204, 21), PALE: (56, 189, 248), NECROTIC: (220, 38, 38)}


def make_overlay(rgb, result, attention=None, alpha=0.55):
    """
    Colour diseased tissue on the photo: red = dead/spotted, yellow = yellowing,
    blue = pale/mould. Spot outlines are drawn so small lesions stay visible.
    `attention` (optional Grad-CAM, any size, 0–1) slightly strengthens colour where
    the model was looking.
    """
    out = rgb.astype(np.float32).copy()
    labels = result["label_map"]
    strength = np.full(labels.shape, alpha, np.float32)
    if attention is not None:
        att = cv2.resize(attention.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
        strength = np.clip(alpha * (0.75 + 0.5 * att), 0, 0.8)
    for code, col in _COLORS.items():
        m = labels == code
        if m.any():
            s = strength[m][:, None]
            out[m] = out[m] * (1 - s) + np.array(col, np.float32) * s
    # Dim the background slightly so the leaf stands out
    bg = result["leaf_mask"] == 0
    out[bg] = out[bg] * 0.55
    out = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(result["lesion_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thick = max(1, int(round(min(rgb.shape[:2]) / 320)))
    cv2.drawContours(out, contours, -1, (255, 255, 255), thick + 1)
    cv2.drawContours(out, contours, -1, (153, 27, 27), thick)
    return out


# Diseases whose symptoms include whitish fungal growth
_PALE_DISEASES = ("Late_blight", "Leaf_Mold")


def analyze_leaf(pil_image, attention=None, disease_class=""):
    """
    Convenience wrapper: PIL image → (result dict at analysis size, overlay RGB uint8, analysed RGB).
    `disease_class` (the model's prediction) switches on mould detection where it applies.
    """
    rgb = np.array(pil_image.convert("RGB"))
    rgb, _ = _resize_max(rgb, ANALYSIS_MAX_SIDE)
    include_pale = any(k in (disease_class or "") for k in _PALE_DISEASES)
    result = detect_lesions(rgb, include_pale=include_pale)
    overlay = make_overlay(rgb, result, attention)
    return result, overlay, rgb
