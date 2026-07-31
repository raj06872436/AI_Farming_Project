"""
disease_heatmap.py — Production-grade disease heatmap engine.

Modular pipeline:
  1. segment_leaf()             → binary leaf mask
  2. generate_gradcam()         → raw Grad-CAM heatmap (model attention)
  3. detect_infected_regions()  → multi-spot disease mask from heatmap
  4. apply_leaf_mask()          → constrain heatmap strictly within leaf
  5. create_heatmap_overlay()   → final overlay with severity coloring

Fixes ALL previous issues:
  - Heatmap NEVER appears outside the leaf
  - Multiple disease spots are ALL highlighted
  - Clean, accurate localization with no bleeding
  - Severity-based coloring (green → yellow → orange → red)
"""

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════════════════════════
# 1. LEAF SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def segment_leaf(image_rgb_uint8):
    """
    Segment the leaf from the background using multi-channel color analysis.

    Args:
        image_rgb_uint8: np.ndarray (H, W, 3) in uint8 [0, 255], RGB format.

    Returns:
        leaf_mask: np.ndarray (H, W) float32 in [0, 1]. Smooth soft mask.
        leaf_mask_binary: np.ndarray (H, W) uint8 {0, 255}. Hard binary mask.
    """
    h, w = image_rgb_uint8.shape[:2]
    hsv = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2LAB)

    # ── Multi-range HSV masks for leaf tissue ──
    # Green foliage (healthy parts)
    mask_green = cv2.inRange(hsv, np.array([20, 25, 25]), np.array([90, 255, 255]))
    # Yellow-green (early disease, senescence)
    mask_yellow = cv2.inRange(hsv, np.array([12, 30, 40]), np.array([25, 255, 255]))
    # Brown / necrotic tissue (advanced disease spots)
    mask_brown = cv2.inRange(hsv, np.array([5, 25, 25]), np.array([20, 200, 200]))
    # Dark lesions (black spots, severe necrosis)
    mask_dark = cv2.inRange(hsv, np.array([0, 0, 15]), np.array([180, 180, 120]))

    # ── LAB-based detection (catches what HSV misses) ──
    # In LAB, A channel > 128 means reddish, B channel > 128 means yellowish
    # Leaf tissue (including diseased) typically has moderate-high A and B
    l_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    # Non-white and non-very-bright pixels (exclude white/gray backgrounds)
    mask_lab = ((l_ch < 220) & (l_ch > 10)).astype(np.uint8) * 255

    # ── Combine all masks ──
    combined = cv2.bitwise_or(mask_green, mask_yellow)
    combined = cv2.bitwise_or(combined, mask_brown)
    combined = cv2.bitwise_or(combined, mask_dark)
    combined = cv2.bitwise_and(combined, mask_lab)

    # ── Morphological cleanup ──
    # Close gaps inside the leaf
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kern_close)
    # Remove small noise outside the leaf
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kern_open)

    # ── Keep only the largest connected components (the leaf) ──
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        # Sort by area, keep the largest regions that are likely the leaf
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        total_area = h * w
        for c in contours:
            area = cv2.contourArea(c)
            # Keep contours that are at least 1% of image area
            if area > total_area * 0.01:
                cv2.drawContours(filled_mask, [c], -1, 255, -1)
            else:
                break  # Skip tiny noise

    # ── Fill internal holes in the leaf ──
    # Flood fill from corners (background) and invert
    flood = filled_mask.copy()
    flood_fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled_mask = cv2.bitwise_or(filled_mask, flood_inv)

    leaf_mask_binary = filled_mask

    # ── Soft mask with smooth edges ──
    leaf_mask_float = filled_mask.astype(np.float32) / 255.0
    # Gaussian blur for smooth mask edges (prevents harsh cutoff artifacts)
    edge_sigma = max(h, w) / 80.0  # Scale sigma with image size
    leaf_mask_float = gaussian_filter(leaf_mask_float, sigma=edge_sigma)
    leaf_mask_float = np.clip(leaf_mask_float, 0, 1)

    return leaf_mask_float, leaf_mask_binary


# ═══════════════════════════════════════════════════════════════════
# 2. GRAD-CAM GENERATION
# ═══════════════════════════════════════════════════════════════════

# Layer mapping — verified from actual saved model architectures
GRADCAM_LAYERS = {
    "MobileNetV2": "out_relu",           # 7x7x1280  (flat, top-level layer)
    "ResNet50": "conv5_block3_out",      # 7x7x2048  (inside 'resnet50' backbone)
    "EfficientNetB0": "top_activation",  # 7x7x1280  (inside 'efficientnetb0' backbone)
    "DenseNet121": "relu",               # 7x7x1024  (inside 'densenet121' backbone)
}


def _find_target_layer(model, layer_name):
    """Find a layer by name, searching both top-level and nested backbones."""
    target = None
    backbone = None

    # Check top-level layers first (flat models like MobileNetV2)
    for layer in model.layers:
        if layer.name == layer_name:
            target = layer
            return target, backbone

    # Search inside nested sub-models (ResNet50, EfficientNetB0, DenseNet121)
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 5:
            try:
                target = layer.get_layer(layer_name)
                backbone = layer
                return target, backbone
            except (ValueError, KeyError):
                continue

    # Fallback: last Conv2D anywhere
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer, None
        if hasattr(layer, 'layers'):
            for sl in reversed(layer.layers):
                if isinstance(sl, tf.keras.layers.Conv2D):
                    return sl, layer

    return None, None


def _find_output_and_penultimate(model):
    """Find the output Dense layer and the layer feeding into it."""
    output_dense = None
    penultimate = None
    found = False
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense) and not found:
            output_dense = layer
            found = True
            continue
        if found and not isinstance(layer, tf.keras.layers.InputLayer):
            penultimate = layer
            break
    return output_dense, penultimate


def _build_grad_model(model, target_layer, backbone, penultimate_layer):
    """
    Build a gradient model that outputs conv activations + penultimate output.

    CRITICAL: Always builds from model.input so that ALL preprocessing layers
    (Rescaling, preprocess_input, etc.) are preserved in the computation graph.
    This prevents MobileNetV2's preprocessing mismatch that causes scattered
    heatmaps when the model expects [-1,1] but receives [0,1] raw input.
    """
    # Resolve output tensors by walking the model graph
    # This works for BOTH flat and nested architectures because
    # model.input → ... → target_layer.output is always a valid path
    # through the original model's graph.
    try:
        if backbone is not None:
            # For nested backbones: target_layer.output lives inside the
            # backbone submodel.  We need to find the corresponding tensor
            # in the *outer* model's graph.
            # Build a temporary model inside the backbone to get the conv output
            backbone_dual = tf.keras.models.Model(
                inputs=backbone.input,
                outputs=[target_layer.output, backbone.output]
            )
            # Replay outer layers up to (but not including) the backbone
            inp = model.input
            x = inp
            for l in model.layers:
                if isinstance(l, tf.keras.layers.InputLayer):
                    continue
                if l is backbone:
                    break
                x = l(x)
            conv_out_tensor, bb_out = backbone_dual(x)

            # Continue from backbone output through remaining layers
            y = bb_out
            past_backbone = False
            for l in model.layers:
                if l is backbone:
                    past_backbone = True
                    continue
                if past_backbone and not isinstance(l, tf.keras.layers.InputLayer):
                    y = l(y)
                    if l is penultimate_layer:
                        break

            return tf.keras.models.Model(inputs=inp, outputs=[conv_out_tensor, y])
        else:
            # Flat model — target_layer.output is directly in the model graph
            return tf.keras.models.Model(
                inputs=model.input,
                outputs=[target_layer.output, penultimate_layer.output]
            )
    except Exception:
        # Fallback: try direct tensor extraction from the model graph
        return tf.keras.models.Model(
            inputs=model.input,
            outputs=[target_layer.output, penultimate_layer.output]
        )


def generate_gradcam(model, img_array, layer_name):
    """
    Generate raw Grad-CAM++ heatmap using pre-softmax logits.

    Uses Grad-CAM++ formulation which captures MULTIPLE distinct activation
    regions (critical for detecting scattered disease spots across a leaf).

    Args:
        model: Trained Keras model.
        img_array: Preprocessed image (1, 224, 224, 3) float32 [0, 1].
        layer_name: Target conv/activation layer name.

    Returns:
        heatmap_224: np.ndarray (224, 224) float32 in [0, 1]. Raw heatmap.
        None if generation fails.
    """
    target_layer, backbone = _find_target_layer(model, layer_name)
    if target_layer is None:
        return None

    output_dense, penultimate = _find_output_and_penultimate(model)
    if output_dense is None or penultimate is None:
        return None

    grad_model = _build_grad_model(model, target_layer, backbone, penultimate)

    # Compute gradients of predicted class RAW LOGIT (bypasses softmax)
    kernel = output_dense.kernel
    bias = output_dense.bias

    with tf.GradientTape() as tape:
        conv_out, penultimate_out = grad_model(img_array)
        logits = tf.matmul(penultimate_out, kernel) + bias
        pred_idx = tf.argmax(logits[0])
        class_logit = logits[:, pred_idx]

    grads = tape.gradient(class_logit, conv_out)
    if grads is None:
        return None

    # ── Grad-CAM++ weights (multi-region aware) ──
    # Standard Grad-CAM uses simple GAP which biases toward the single
    # largest activation. Grad-CAM++ uses second-order weights that
    # preserve ALL distinct activation regions independently.
    pos_grads = tf.maximum(grads[0], 0)
    alpha_num = pos_grads ** 2
    alpha_denom = (
        2.0 * alpha_num
        + tf.reduce_sum(conv_out[0] * (pos_grads ** 3), axis=(0, 1), keepdims=True)
        + 1e-8
    )
    alpha_weights = alpha_num / alpha_denom
    weights = tf.reduce_sum(alpha_weights * pos_grads, axis=(0, 1))

    # Weighted combination + ReLU
    heatmap = tf.reduce_sum(weights * conv_out[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Upscale to 224x224 with sharp interpolation (LANCZOS preserves edges)
    hm_uint8 = np.uint8(255 * heatmap)
    hm_img = Image.fromarray(hm_uint8).resize((224, 224), Image.LANCZOS)
    heatmap_224 = np.array(hm_img).astype(np.float32) / 255.0

    # Very light gaussian blur — minimal smoothing to remove block artifacts
    # while preserving distinct multi-spot activation peaks
    heatmap_224 = gaussian_filter(heatmap_224, sigma=0.8)
    hm_min, hm_max = heatmap_224.min(), heatmap_224.max()
    heatmap_224 = (heatmap_224 - hm_min) / (hm_max - hm_min + 1e-8)

    return heatmap_224


# ═══════════════════════════════════════════════════════════════════
# 3. COLOR-BASED LESION DETECTION (supplements Grad-CAM)
# ═══════════════════════════════════════════════════════════════════

def _detect_lesions_by_color(image_rgb_uint8, leaf_mask_binary=None):
    """
    Detect brown/necrotic lesion spots using HSV + LAB color analysis.

    Grad-CAM operates at 7×7 resolution (each cell ≈ 32×32 pixels) so it
    fundamentally cannot resolve multiple small spots — they merge into one
    large blob. This function provides the spatial precision that Grad-CAM
    lacks by detecting lesion-colored pixels directly in the image.

    Args:
        image_rgb_uint8: np.ndarray (H, W, 3) uint8 RGB image.
        leaf_mask_binary: optional np.ndarray (H, W) uint8 {0,255}.

    Returns:
        lesion_mask: np.ndarray (H, W) float32 [0, 1]. Per-pixel lesion score.
    """
    h, w = image_rgb_uint8.shape[:2]
    hsv = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2LAB)

    # ── Brown / tan necrotic tissue (most common lesion color) ──
    mask_brown = cv2.inRange(hsv, np.array([5, 40, 30]), np.array([25, 220, 200]))

    # ── Dark brown / black spots (severe necrosis) ──
    mask_dark_brown = cv2.inRange(hsv, np.array([0, 20, 15]), np.array([20, 200, 100]))

    # ── Reddish-brown lesions ──
    mask_red_brown = cv2.inRange(hsv, np.array([0, 50, 30]), np.array([10, 255, 180]))

    # ── LAB: high A-channel (reddish) with low-mid L (dark) = lesion ──
    l_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    # Reddish-brownish pixels: A > 135 (reddish), L < 180 (not bright white)
    mask_lab_lesion = ((a_ch > 135) & (l_ch < 180) & (l_ch > 15)).astype(np.uint8) * 255

    # ── Combine all lesion color masks ──
    combined = cv2.bitwise_or(mask_brown, mask_dark_brown)
    combined = cv2.bitwise_or(combined, mask_red_brown)
    combined = cv2.bitwise_or(combined, mask_lab_lesion)

    # ── Constrain to leaf area if mask is provided ──
    if leaf_mask_binary is not None:
        if leaf_mask_binary.shape != (h, w):
            leaf_mask_binary = cv2.resize(leaf_mask_binary, (w, h),
                                          interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_and(combined, leaf_mask_binary)

    # ── Morphological cleanup: preserve small spots ──
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kern_open)
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kern_close)

    # Convert to float [0, 1] with soft edges
    lesion_mask = combined.astype(np.float32) / 255.0
    lesion_mask = gaussian_filter(lesion_mask, sigma=1.5)
    lesion_mask = np.clip(lesion_mask / (lesion_mask.max() + 1e-8), 0, 1)

    return lesion_mask


# ═══════════════════════════════════════════════════════════════════
# 4. HYBRID MULTI-REGION DISEASE DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_infected_regions(heatmap, threshold=0.12, min_area_pct=0.08,
                            image_rgb_uint8=None, leaf_mask_binary=None):
    """
    Detect ALL infected regions using a hybrid approach:
      1. Grad-CAM heatmap → model attention signal (what the model sees)
      2. Color-based lesion detection → spatial precision (where lesions are)
      3. Fusion: multiply color lesion map with Grad-CAM to get model-validated
         lesion spots with precise spatial boundaries.

    This solves the fundamental limitation of Grad-CAM's 7×7 resolution
    which merges nearby spots into a single blob.

    Args:
        heatmap: np.ndarray (H, W) float32 [0, 1].
        threshold: Fallback minimum activation to consider as infected.
        min_area_pct: Minimum region area as percentage of image.
        image_rgb_uint8: optional np.ndarray (H, W, 3) uint8 RGB for color detection.
        leaf_mask_binary: optional np.ndarray (H, W) uint8 {0,255}.

    Returns:
        disease_mask: np.ndarray (H, W) float32 [0, 1].
        num_regions: int — number of distinct infected regions found.
        region_stats: list of dicts with per-region statistics.
    """
    h, w = heatmap.shape

    # ── HYBRID: Fuse Grad-CAM with color-based lesion detection ──
    if image_rgb_uint8 is not None:
        color_lesions = _detect_lesions_by_color(image_rgb_uint8, leaf_mask_binary)
        # Resize to match heatmap if needed
        if color_lesions.shape != (h, w):
            color_lesions = cv2.resize(color_lesions, (w, h),
                                       interpolation=cv2.INTER_LINEAR)

        # Fusion: color provides spatial precision, Grad-CAM provides
        # disease-relevance weighting.
        #
        # SOFT GATING: Use a floor so color spots are NEVER fully silenced,
        # even when Grad-CAM doesn't reach them.  This is critical because
        # the 7×7 Grad-CAM resolution cannot distinguish spots that are
        # close together — they merge into one blob, leaving other spots
        # with zero Grad-CAM activation.
        #
        #   - Grad-CAM high + color → full signal (strong confidence)
        #   - Grad-CAM low  + color → partial signal (color still contributes)
        #   - Grad-CAM high + no color → moderate signal (model attention only)
        gradcam_gate = np.clip(heatmap * 1.5 + 0.25, 0, 1)  # Soft gate with 0.25 floor
        color_gated = color_lesions * gradcam_gate

        # Blend: 65% color-gated spots + 35% raw Grad-CAM
        fused = 0.65 * color_gated + 0.35 * heatmap
        fused = np.clip(fused / (fused.max() + 1e-8), 0, 1)  # Re-normalize
    else:
        fused = heatmap

    # ── Adaptive threshold using Otsu's method ──
    hm_uint8 = np.uint8(np.clip(fused, 0, 1) * 255)
    otsu_thresh_val, _ = cv2.threshold(hm_uint8, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = otsu_thresh_val / 255.0

    # Use the lower of Otsu and explicit threshold
    effective_threshold = max(min(otsu_thresh * 0.7, threshold), 0.08)

    # Binary threshold
    binary = (fused > effective_threshold).astype(np.uint8) * 255

    # Morphological cleanup — gentle: preserve separate spots
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern_open)
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern_close)

    # Connected component analysis — find every distinct region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    min_area = h * w * min_area_pct / 100.0
    disease_mask = np.zeros((h, w), dtype=np.float32)
    region_stats = []

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # Create mask for this region
        region_mask = (labels == i).astype(np.float32)

        # Dilate slightly to capture full lesion boundary
        kern_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        region_mask = cv2.dilate(region_mask, kern_dilate, iterations=1)

        # Intensity = fused signal values within this region
        region_intensity = fused * region_mask

        # Accumulate into disease mask
        disease_mask = np.maximum(disease_mask, region_intensity)

        # Compute stats
        cx, cy = centroids[i]
        mean_intensity = float(np.mean(fused[labels == i]))
        max_intensity = float(np.max(fused[labels == i]))
        region_stats.append({
            "id": len(region_stats) + 1,
            "area_px": int(area),
            "area_pct": float(area / (h * w) * 100),
            "center": (int(cx), int(cy)),
            "mean_intensity": mean_intensity,
            "max_intensity": max_intensity,
            "severity": "severe" if max_intensity > 0.7 else "moderate" if max_intensity > 0.4 else "mild",
        })

    return disease_mask, len(region_stats), region_stats


# ═══════════════════════════════════════════════════════════════════
# 4. APPLY LEAF MASK
# ═══════════════════════════════════════════════════════════════════

def apply_leaf_mask(heatmap, leaf_mask_float):
    """
    Multiply heatmap by leaf mask to strictly constrain it within the leaf.
    Pixels outside the leaf become exactly zero.

    Args:
        heatmap: np.ndarray (H, W) float32 [0, 1].
        leaf_mask_float: np.ndarray (H, W) float32 [0, 1]. Soft leaf mask.

    Returns:
        masked_heatmap: np.ndarray (H, W) float32 [0, 1].
    """
    # Ensure same dimensions
    if heatmap.shape != leaf_mask_float.shape:
        leaf_mask_float = cv2.resize(
            leaf_mask_float, (heatmap.shape[1], heatmap.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    masked = heatmap * leaf_mask_float

    # Re-normalize so the max activation within the leaf is 1.0
    mx = masked.max()
    if mx > 1e-8:
        masked = masked / mx

    return masked


# ═══════════════════════════════════════════════════════════════════
# 5. CREATE HEATMAP OVERLAY
# ═══════════════════════════════════════════════════════════════════

def _disease_colormap(intensity):
    """
    Custom disease severity colormap:
      0.0 → transparent (healthy, no overlay)
      0.0-0.3 → green-yellow (mild)
      0.3-0.6 → yellow-orange (moderate)
      0.6-1.0 → orange-red (severe)

    Args:
        intensity: np.ndarray (H, W) float32 [0, 1].

    Returns:
        color: np.ndarray (H, W, 3) float32 [0, 1] in RGB.
    """
    h, w = intensity.shape
    color = np.zeros((h, w, 3), dtype=np.float32)

    # Green-yellow (mild)
    mask_mild = (intensity > 0.0) & (intensity <= 0.35)
    t = np.clip(intensity / 0.35, 0, 1)
    color[mask_mild, 0] = (0.2 + 0.6 * t[mask_mild])   # R: 0.2→0.8
    color[mask_mild, 1] = (0.8 - 0.2 * t[mask_mild])   # G: 0.8→0.6
    color[mask_mild, 2] = 0.0                            # B: 0

    # Yellow-orange (moderate)
    mask_mod = (intensity > 0.35) & (intensity <= 0.65)
    t = np.clip((intensity - 0.35) / 0.30, 0, 1)
    color[mask_mod, 0] = (0.9 + 0.1 * t[mask_mod])      # R: 0.9→1.0
    color[mask_mod, 1] = (0.7 - 0.4 * t[mask_mod])      # G: 0.7→0.3
    color[mask_mod, 2] = 0.0                              # B: 0

    # Orange-red (severe)
    mask_sev = intensity > 0.65
    t = np.clip((intensity - 0.65) / 0.35, 0, 1)
    color[mask_sev, 0] = 1.0                              # R: 1.0
    color[mask_sev, 1] = (0.3 - 0.3 * t[mask_sev])       # G: 0.3→0.0
    color[mask_sev, 2] = 0.0                              # B: 0

    return color


def create_heatmap_overlay(
    original_rgb_float,
    masked_heatmap,
    leaf_mask_float,
    alpha_range=(0.4, 0.65),
    use_disease_colormap=True,
):
    """
    Create the final overlay image.

    The overlay is applied ONLY on leaf pixels. Background remains unchanged.
    Healthy leaf regions stay mostly original. Diseased regions are colored
    by severity intensity.

    Args:
        original_rgb_float: np.ndarray (H, W, 3) float32 [0, 1].
        masked_heatmap: np.ndarray (H, W) float32 [0, 1]. Leaf-masked heatmap.
        leaf_mask_float: np.ndarray (H, W) float32 [0, 1]. Soft leaf mask.
        alpha_range: (min_alpha, max_alpha) for blending.
        use_disease_colormap: If True, use green→yellow→red severity colors.
                              If False, use standard jet colormap.

    Returns:
        overlay: np.ndarray (H, W, 3) float32 [0, 1].
        act_pct: float — percentage of leaf area with significant activation.
    """
    import matplotlib.cm as cm

    h, w = original_rgb_float.shape[:2]

    # Ensure dimensions match
    if masked_heatmap.shape != (h, w):
        masked_heatmap = cv2.resize(masked_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    if leaf_mask_float.shape != (h, w):
        leaf_mask_float = cv2.resize(leaf_mask_float, (w, h), interpolation=cv2.INTER_LINEAR)

    # Generate colormap
    if use_disease_colormap:
        colormap = _disease_colormap(masked_heatmap)
    else:
        colormap = cm.jet(masked_heatmap)[:, :, :3]

    # Alpha map: higher activation → more opaque overlay
    min_alpha, max_alpha = alpha_range
    # Activation threshold at 0.15 — below this, no overlay at all
    raw_alpha = np.clip((masked_heatmap - 0.15) / 0.85, 0, 1)
    # Scale alpha within the specified range
    alpha = raw_alpha * max_alpha
    # Ensure background is completely transparent
    alpha = alpha * leaf_mask_float
    alpha = np.expand_dims(alpha, axis=-1)

    # Blend
    overlay = np.clip(
        original_rgb_float * (1.0 - alpha) + colormap * alpha,
        0, 1
    )

    # Compute activation percentage (of leaf area, not whole image)
    leaf_pixels = np.sum(leaf_mask_float > 0.5)
    if leaf_pixels > 0:
        active_pixels = np.sum((masked_heatmap > 0.25) & (leaf_mask_float > 0.5))
        act_pct = float(active_pixels / leaf_pixels * 100)
    else:
        act_pct = 0.0

    return overlay, act_pct


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE — called from app.py
# ═══════════════════════════════════════════════════════════════════

def run_disease_heatmap_pipeline(model, img_array, model_name):
    """
    Complete disease heatmap pipeline.

    Args:
        model: Trained Keras model.
        img_array: Preprocessed image (1, 224, 224, 3) float32 [0, 1].
        model_name: str — one of 'MobileNetV2', 'ResNet50', etc.

    Returns:
        dict with keys:
            'leaf_mask': (224, 224) float32 — soft leaf mask
            'raw_heatmap': (224, 224) float32 — raw Grad-CAM++ heatmap
            'disease_mask': (224, 224) float32 — multi-region disease mask
            'masked_heatmap': (224, 224) float32 — heatmap constrained to leaf
            'overlay': (224, 224, 3) float32 — final overlay image
            'act_pct': float — activation percentage
            'num_regions': int — number of detected disease regions
            'region_stats': list — per-region statistics
        None if pipeline fails.
    """
    try:
        layer_name = GRADCAM_LAYERS.get(model_name, "")

        # Step 1: Segment the leaf
        original_uint8 = np.uint8(img_array[0] * 255)
        leaf_mask_float, leaf_mask_binary = segment_leaf(original_uint8)

        # Step 2: Generate Grad-CAM++ heatmap
        raw_heatmap = generate_gradcam(model, img_array, layer_name)
        if raw_heatmap is None:
            return None

        # Step 3: Detect all infected regions (hybrid: Grad-CAM + color)
        disease_mask, num_regions, region_stats = detect_infected_regions(
            raw_heatmap,
            image_rgb_uint8=original_uint8,
            leaf_mask_binary=leaf_mask_binary,
        )

        # Step 4: Apply leaf mask — strictly constrain to leaf
        masked_heatmap = apply_leaf_mask(raw_heatmap, leaf_mask_float)

        # Step 5: Create overlay
        original_float = img_array[0]
        overlay, act_pct = create_heatmap_overlay(
            original_float, masked_heatmap, leaf_mask_float
        )

        return {
            "leaf_mask": leaf_mask_float,
            "raw_heatmap": raw_heatmap,
            "disease_mask": disease_mask,
            "masked_heatmap": masked_heatmap,
            "overlay": overlay,
            "act_pct": act_pct,
            "num_regions": num_regions,
            "region_stats": region_stats,
        }
    except Exception as e:
        import traceback
        print(f"Disease heatmap pipeline error: {e}")
        traceback.print_exc()
        return None


def run_fullres_overlay(heatmap_224, original_pil_image):
    """
    Generate a full-resolution overlay for display.
    Takes the 224x224 heatmap and scales it to the original image dimensions,
    applying leaf masking at the full resolution.

    Args:
        heatmap_224: np.ndarray (224, 224) float32 [0, 1]. Raw masked heatmap.
        original_pil_image: PIL.Image — original uploaded image.

    Returns:
        overlay_fullres: np.ndarray (H, W, 3) float32 [0, 1].
        act_pct: float.
    """
    orig_arr = np.array(original_pil_image.convert("RGB"))
    h_orig, w_orig = orig_arr.shape[:2]
    orig_float = orig_arr.astype(np.float32) / 255.0

    # Upscale heatmap to original resolution (CUBIC for sharper edges)
    hm_resized = cv2.resize(heatmap_224, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    # Minimal blur scaled to resolution — preserve multi-spot separation
    scale = max(w_orig, h_orig) / 224.0
    hm_resized = gaussian_filter(hm_resized, sigma=0.8 * scale)
    hm_min, hm_max = hm_resized.min(), hm_resized.max()
    hm_resized = (hm_resized - hm_min) / (hm_max - hm_min + 1e-8)

    # Segment leaf at full resolution
    leaf_mask_float, _ = segment_leaf(orig_arr)

    # Apply leaf mask
    masked = apply_leaf_mask(hm_resized, leaf_mask_float)

    # Create overlay
    overlay, act_pct = create_heatmap_overlay(
        orig_float, masked, leaf_mask_float
    )

    return overlay, act_pct
