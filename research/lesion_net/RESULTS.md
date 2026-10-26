# LesionNet — results

LesionNet first segments every pixel into background / healthy leaf / lesion, then
classifies using only features pooled over the predicted leaf and lesion pixels
(lesion-gated pooling, gate detached from the classification loss). It is trained
without hand annotation, from automatic lesion masks (`make_pseudo_masks.py`), with
background-swap augmentation. Backbone: EfficientNetB0 (ImageNet), 3 + 15 epochs,
RTX 4050 Laptop GPU, 76 min.

All numbers below come from `results/eval_summary.csv` (PlantVillage test split) and
`results/plantdoc_summary.csv` (PlantDoc). Re-running `evaluate.py` / `eval_external.py`
regenerates them.

## 1. PlantVillage held-out test split (3,109 photos, 15 classes)

| Model | Accuracy | Macro-F1 | Grey background | Swapped background | Background only (chance 6.7 %) |
|---|---|---|---|---|---|
| MobileNetV2 | 93.0 % | 0.919 | 76.0 % | 71.6 % | 19.1 % |
| ResNet50 | **99.8 %** | **0.998** | 45.1 % | 60.0 % | 37.4 % |
| EfficientNetB0 | 96.4 % | 0.963 | 81.2 % | 62.3 % | 17.8 % |
| DenseNet121 | 96.1 % | 0.962 | 56.3 % | 57.6 % | 28.9 % |
| **LesionNet** | 99.5 % | 0.995 | **98.7 %** | **81.8 %** | 19.5 % |

- "Grey / swapped background": the leaf is kept and its background replaced by flat grey or by
  the background of a photo from another class. These tests need no lesion masks.
- "Background only": the leaf is filled with its own background texture (no leaf, no
  silhouette). Every model, LesionNet included, stays above chance — PlantVillage backgrounds
  carry class information, and LesionNet's segmentation does not remove all of it.

**Takeaway:** same accuracy as the best classic model on normal photos, but it keeps 98.7 %
when the background is changed to grey, where the classic models fall to 45–81 %.

## 2. Where the evidence comes from (diseased test leaves with a reliable lesion mask)

| Model | Explanation | On lesions | On background | Lesion focus ratio | Pointing hit | IoU |
|---|---|---|---|---|---|---|
| MobileNetV2 | Grad-CAM++ | 9.4 % | 28.8 % | 0.96 | 49 % | 0.11 |
| ResNet50 | Grad-CAM++ | 8.7 % | 38.1 % | 1.06 | 64 % | 0.12 |
| EfficientNetB0 | Grad-CAM++ | 12.1 % | 24.7 % | 1.22 | 63 % | 0.13 |
| DenseNet121 | Grad-CAM++ | 7.1 % | 48.7 % | 1.03 | 60 % | 0.08 |
| **LesionNet** | own lesion map | **44.1 %** | **6.7 %** | **6.06** | **97 %** | **0.50** |

Caveat: the lesion masks used here are the automatic masks LesionNet was trained to imitate,
so this table favours LesionNet. Table 1's background tests and table 4 do not use them.

## 3. Severity

LesionNet's predicted diseased share of the leaf vs the measured share: Spearman
**ρ = 0.90**, mean absolute error 5.7 percentage points; on healthy leaves it predicts
0.5 % diseased. The Grad-CAM "affected area" used before had |ρ| ≤ 0.13
(`research/results.md`).

## 4. Second dataset: PlantDoc field photos (cross-dataset, no training on PlantDoc)

1,012 photos from fields / the internet (`research/datasets/fetch_plantdoc.py`), the 12
classes PlantDoc shares with our 15. Chance: 8.3 %.

| Model | Accuracy (12 classes) | Macro-F1 | Right crop | Lesion % healthy / diseased leaves |
|---|---|---|---|---|
| MobileNetV2 | **23.6 %** | **0.189** | 68.9 % | — |
| ResNet50 | 19.6 % | 0.147 | 65.2 % | — |
| EfficientNetB0 | 20.2 % | 0.148 | 68.9 % | — |
| DenseNet121 | 20.1 % | 0.158 | 67.4 % | — |
| LesionNet | 20.8 % | 0.172 | **70.7 %** | 16.5 % / 24.7 % |

**Takeaway (negative result, worth reporting):** every model trained only on PlantVillage
lab photos collapses on field photos (20–24 %, chance 8 %), LesionNet included. Removing the
background shortcut is necessary but not sufficient: field photos also differ in lighting,
several overlapping leaves, blur and scale. This matches the drop reported by the PlantDoc
authors. Per-class numbers: `results/plantdoc_per_class.csv`.

Next step for the paper: fine-tune on the PlantDoc *train* split and test on its *test* split
(LesionNet's pseudo-mask pipeline can label PlantDoc images too), and compare how fast each
model adapts.

## 5. Ablations (run by `run_overnight.sh`)

| Tag | What changes | Purpose |
|---|---|---|
| A0_plain | ordinary global pooling, no segmentation loss, no background swap | fair baseline: same backbone, same split, same training |
| A3_full | — | LesionNet |
| A5_mobilenet | MobileNetV2 backbone | phone-sized LesionNet |

After `run_overnight.sh` finishes, `results/eval_summary.md` contains all of them.

## Honest limitations

1. Lesion masks are automatic (colour-based); they under-detect diffuse yellowing (viruses,
   mites). Pixel-level claims need hand-drawn masks (e.g. PlantSeg).
2. The four classic models were trained on a different split and may have seen some of
   these test photos; A0_plain is the clean like-for-like baseline.
3. Single run, single seed. Report mean ± std over 3 seeds before submission.
4. Field generalisation (table 4) is unsolved for all models.
