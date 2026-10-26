# LesionNet — a classifier that has to point at the disease

Your four existing models are accurate, but the attention study (`research/results.md`)
showed they put only 8–13 % of their attention on diseased tissue, and a lot on the
background. LesionNet is built so it **can't** do that:

1. **Outline first:** a segmentation head marks every pixel as background, healthy leaf
   or lesion.
2. **Then decide, using only the outlined areas:** the classifier only sees features
   averaged over the predicted *lesion* pixels and *leaf* pixels, plus the lesion area.
   Background pixels never reach the decision.
3. **No hand annotation:** the outlines it learns from are generated automatically by
   `lesion_segmentation.py`. Healthy-class photos become clean "no disease" examples.
4. **Background swap during training:** half of the training photos get a random new
   background, so table colour can't be used as a clue.

The predicted lesion map is the explanation: the class is computed from exactly those
pixels. Analogy: a doctor who must circle the rash before naming the illness.

## Run it (WSL2, your `tf_gpu` environment, RTX 4050)

```bash
cd /mnt/c/Users/raj\ naik/Downloads/AI_Farming_Project_Final
source ~/tf_gpu/bin/activate
pip install opencv-python-headless pandas matplotlib   # if not already installed

# 1. Training labels for all ~20,600 photos (≈10–15 min, CPU)
python research/lesion_net/make_pseudo_masks.py --data PlantVillage --workers 8

# 2. Train LesionNet (≈45–60 min on the RTX 4050)
python research/lesion_net/train_lesionnet.py --tag A3_full

# 3. Compare with your 4 existing models on the held-out test split (≈15–30 min)
python research/lesion_net/evaluate.py
```

Results land in `research/lesion_net/results/`: `eval_summary.md`, `eval_summary.csv`
and `fig_lesionnet_vs_baselines.png`. The trained model is saved as
`saved_models/LesionNet_A3_full.keras`.

If the GPU runs out of memory, add `--batch 16` to the training command.

## Ablations for the paper (≈45–60 min each)

Each row adds one idea, so you can show what each part contributes:

| Tag | Command flags | What it tests |
|---|---|---|
| `A0_plain` | `--pool gap --lambda-seg 0 --bg-swap 0` | Same backbone, ordinary classifier (fair baseline) |
| `A1_bgswap` | `--pool gap --lambda-seg 0 --bg-swap 0.5` | + background swap only |
| `A2_multitask` | `--pool gap --lambda-seg 1 --bg-swap 0.5` | + segmentation head, but the classifier still sees everything |
| `A3_full` | *(defaults)* | + lesion-gated pooling = **LesionNet** |
| `A4_nodetach` | `--no-detach` | gate trained by the classifier too (is detaching important?) |
| `A5_mobilenet` | `--backbone MobileNetV2` | lighter backbone for phones |

```bash
python research/lesion_net/train_lesionnet.py --pool gap --lambda-seg 0 --bg-swap 0 --tag A0_plain
python research/lesion_net/train_lesionnet.py --pool gap --lambda-seg 0 --bg-swap 0.5 --tag A1_bgswap
python research/lesion_net/train_lesionnet.py --pool gap --lambda-seg 1 --bg-swap 0.5 --tag A2_multitask
python research/lesion_net/train_lesionnet.py --no-detach --tag A4_nodetach
python research/lesion_net/evaluate.py      # evaluates every saved_models/LesionNet_*.keras
```

## What the evaluation measures

| Test | Why it matters |
|---|---|
| Accuracy, macro-F1 | LesionNet should match the existing models |
| **Background-only accuracy** | Leaf replaced by its own background texture. Above chance (6.7 %) = the model learned shortcuts from the background |
| **Grey / swapped background accuracy** | Same leaf, new background. A model that uses the leaf keeps its accuracy |
| Explanation on lesions / background, pointing game, IoU | Where the evidence comes from |
| Severity ρ, lesion % on healthy leaves | Can its lesion map measure how sick the leaf is? |

## Honest caveats

- **Lesion-attention test favours LesionNet.** It is judged against the same automatic
  masks it was trained on. The **background tests don't use lesion masks**, so they are
  the fair comparison. For an independent lesion test, evaluate on hand-drawn masks
  (e.g. PlantSeg).
- **Baselines may have seen test photos.** The existing four models were trained with a
  different split, so their test accuracy here may be slightly optimistic. LesionNet's
  test photos were never used in its training. The `A0_plain` ablation is the clean
  like-for-like baseline.
- **Smoke test only so far.** On 420 photos, CPU, 4 short epochs, starting from your
  EfficientNetB0 backbone, the pipeline ran end to end (val accuracy 21 % → 70 %).
  Real numbers need the full GPU run above.
- **Use ImageNet weights for the paper.** `--init-backbone` exists only for quick tests:
  that backbone may already have seen your test photos.
