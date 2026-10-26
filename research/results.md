# Do the AGRI-X classifiers look at the disease?

Attention-vs-lesion study on the four trained models
(MobileNetV2, ResNet50, EfficientNetB0, DenseNet121).

## Setup

- **Images:** 600 PlantVillage photos, 40 per class, all 15 classes, random sample (seed 7).
  Lesion metrics use the 419 diseased photos whose measured lesion area is at least 1 % of the leaf.
- **Attention:** Grad-CAM and Grad-CAM++ for the predicted class, at the last convolutional
  block (7×7), upsampled bilinearly to 224×224.
- **Lesion masks:** produced automatically by `lesion_segmentation.py` (colour-based, compared
  with each leaf's own healthy green). These are **pseudo ground truth**, not hand-drawn
  annotations. 410 of the 419 masks passed the tool's own reliability check, and every
  conclusion below also holds on that subset.
- **Metrics:**
  - **Attention share:** fraction of total heatmap "energy" on diseased tissue, healthy leaf
    and background.
  - **Lesion focus ratio:** mean attention on lesions ÷ mean attention on the whole leaf.
    1.0 means no preference for lesions.
  - **Pointing game:** does the heatmap peak land within 15 px of a lesion?
  - **IoU:** overlap between heatmap > 0.5 and the lesion mask.
  - **Spearman ρ:** between the heatmap's "affected area" (% of leaf with attention > 0.25,
    which is what the app used to report) and the measured lesion area.
- **Scripts:** `research/attention_lesion_study.py` → `research/analyze_attention_study.py`.
  Per-image results: `results/attention_study.csv`.

## Main results (diseased leaves, n = 419)

| Model | Method | Acc. (sample) | Attention on lesions | on background | Focus ratio | Pointing hit | IoU | ρ(CAM area, lesion area) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | Grad-CAM | 0.920 | 0.107 | 0.275 | 0.95 | 0.51 | 0.11 | 0.06 |
| MobileNetV2 | Grad-CAM++ | 0.920 | 0.104 | 0.308 | 0.97 | 0.54 | 0.12 | −0.03 |
| ResNet50 | Grad-CAM | 0.997 | 0.105 | 0.350 | 1.07 | 0.64 | 0.12 | 0.13 |
| ResNet50 | Grad-CAM++ | 0.997 | 0.096 | 0.397 | 1.07 | 0.69 | 0.13 | 0.04 |
| EfficientNetB0 | Grad-CAM | 0.957 | 0.114 | 0.335 | 1.09 | 0.62 | 0.13 | 0.05 |
| EfficientNetB0 | Grad-CAM++ | 0.957 | 0.131 | 0.257 | 1.20 | 0.66 | 0.14 | 0.02 |
| DenseNet121 | Grad-CAM | 0.962 | 0.101 | 0.370 | 1.07 | 0.63 | 0.12 | −0.04 |
| DenseNet121 | Grad-CAM++ | 0.962 | 0.078 | 0.499 | 1.04 | 0.65 | 0.09 | −0.11 |

**Chance levels to compare against:**

- Lesions cover 14.5 % of the leaf and 6.0 % of the image.
- Background is 58 % of the image.
- A random point on the leaf "hits" a lesion (with 15 px tolerance) 53 % of the time; a random
  point anywhere in the image hits 28 % of the time.

Bootstrap 95 % confidence intervals for every number are in `results/summary_table.csv`.

## Findings

1. **High accuracy, little lesion focus.**
   - All models reach 92–99.7 % accuracy on this sample, yet put only **8–13 % of their
     attention on diseased tissue**, which covers 14.5 % of the leaf.
   - Per unit area they attend to lesions about as much as to healthy tissue
     (focus ratio 0.95–1.20). The best is EfficientNetB0 with Grad-CAM++ at 1.20.
2. **A large share of attention lands off the leaf.**
   - 26–50 % of attention falls on the background. That is less than the background's 58 %
     area share, but still substantial.
   - This is consistent with the known background bias of PlantVillage.
3. **Peaks only modestly better than chance.** The heatmap peak lands on a lesion in 51–69 %
   of images, against 53 % for a random point on the leaf.
4. **Heatmap area is not a severity measure.**
   - The heatmap "affected area" has essentially no relationship with the measured lesion
     area (|ρ| ≤ 0.13).
   - It is 66–100 % on perfectly **healthy** leaves.
5. **Results depend on the disease.** For ResNet50 with Grad-CAM++, the hit rate is high for
   blights (Potato late blight 1.00, early blight 0.88). It is low for faint or diffuse
   diseases: spider mites 0.29, mosaic 0.37, yellow-leaf-curl 0.03.
   - Part of this gap comes from the lesion masks, not the models. The colour-based masks
     under-detect diffuse yellowing, as in the yellow-leaf-curl example in
     `results/fig_examples.png`, where the models attend to the mottled interior that the
     mask misses.
   - Treat the faint-disease rows with caution until hand-drawn masks are available.

## Implementation issue found during the study (fixed)

Under Keras 3, `preprocess_input(...)` applied to a symbolic tensor becomes graph operations,
not layers. The original gradient-model builder replayed only `model.layers` before the
backbone, so it fed ResNet50 / DenseNet121 an incorrectly normalised image.

Measured on 60 images (`research/gradcam_bug_check.py`):

| Model | Old pipeline chose the same class as the model | Heatmap correlation old vs fixed |
|---|---|---|
| ResNet50 | 22 / 60 | 0.87 (min 0.61) |
| DenseNet121 | 4 / 60 | 0.15 (min −0.65) |
| EfficientNetB0 | 60 / 60 | 1.00 (unaffected: its preprocessing is built into the backbone) |
| MobileNetV2 | — | unaffected (flat model, graph used directly) |

The fixed builder reproduces each model's softmax output exactly (max difference 0.0). It is
worth checking for this when comparing explanations across Keras backbones.

## Limitations

- Lesion masks are automatic, so absolute numbers depend on them; they under-detect diffuse yellowing (viral diseases, spider mites). Hand-drawn masks (for
  example PlantSeg) are needed before making claims about pixel accuracy.
- The study uses 40 images per class from a single dataset (PlantVillage, lab backgrounds).
  Field photos may behave differently.
- Grad-CAM-family maps are 7×7 before upsampling, which limits IoU for small spots. A
  higher-resolution explanation method would be a fair additional baseline.
