"""
Step 1 — create training labels for LesionNet (no hand annotation needed).

For every PlantVillage photo this writes a small label image:
    0 = background, 1 = healthy leaf tissue, 2 = diseased tissue (lesion)
using the colour-based segmentation in lesion_segmentation.py.

  • Healthy-class photos: any "lesion" pixels are set to healthy leaf — we KNOW
    these leaves are healthy, so they become clean negative examples.
  • Diseased photos whose mask looks unreliable (or where < 1 % lesion was found)
    get seg_weight = 0, so the network is not taught "no disease here" on a sick leaf.

It also writes a stratified 70 / 15 / 15 train / val / test split (fixed seed) that
train_lesionnet.py and evaluate.py both use.

Usage (WSL / any shell, from the project folder):
    python research/lesion_net/make_pseudo_masks.py --data PlantVillage --workers 8
"""
import argparse
import csv
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import lesion_segmentation as ls  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _rel(path):
    """Store paths relative to the project so the manifest works from Windows and WSL alike."""
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT).replace(os.sep, "/") if path.startswith(ROOT) else path


def process(job):
    src, cls, out_path = job
    try:
        rgb = np.array(Image.open(src).convert("RGB"))
        r = ls.detect_lesions(rgb, include_pale=any(k in cls for k in ("Late_blight", "Leaf_Mold")))
        label = np.zeros(rgb.shape[:2], np.uint8)
        label[r["leaf_mask"] > 0] = 1
        healthy = "healthy" in cls.lower()
        if not healthy:
            label[(r["lesion_mask"] > 0) & (r["leaf_mask"] > 0)] = 2
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(label).save(out_path)
        lesion_pct = 0.0 if healthy else r["affected_pct"]
        ok = healthy or (r["reliable"] and lesion_pct >= 1.0)
        return {"image": src, "label": out_path, "class": cls, "healthy": int(healthy),
                "lesion_pct": round(lesion_pct, 2), "leaf_pct": round(float((label > 0).mean() * 100), 1),
                "seg_weight": 1.0 if ok else 0.0}
    except Exception as e:  # unreadable image etc.
        return {"image": src, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "PlantVillage"))
    ap.add_argument("--out", default=os.path.join(HERE, "pseudo_masks"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit-per-class", type=int, default=0, help="for quick tests; 0 = all images")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = [l.strip() for l in open(os.path.join(ROOT, "class_names.txt")) if l.strip()]
    jobs = []
    for cls in classes:
        folder = os.path.join(args.data, cls)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
        if args.limit_per_class:
            files = files[:args.limit_per_class]
        for f in files:
            jobs.append((os.path.join(folder, f), cls,
                         os.path.join(args.out, cls, os.path.splitext(f)[0] + ".png")))
    print(f"{len(jobs)} images, {args.workers} workers")

    rows, done = [], 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(process, jobs, chunksize=16):
            done += 1
            if "error" in r:
                print("skip", r["image"], r["error"])
                continue
            r["image"], r["label"] = _rel(r["image"]), _rel(r["label"])
            rows.append(r)
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)}")

    # Stratified split
    rng = random.Random(args.seed)
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["class"], []).append(r)
    for cls, items in by_cls.items():
        items.sort(key=lambda r: r["image"])
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        for i, r in enumerate(items):
            r["split"] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "label", "class", "healthy", "lesion_pct", "leaf_pct",
                                            "seg_weight", "split"])
        wr.writeheader()
        for r in sorted(rows, key=lambda r: r["image"]):
            wr.writerow(r)
    used = sum(r["seg_weight"] > 0 for r in rows)
    print(f"saved {manifest}: {len(rows)} images, {used} with segmentation supervision")


if __name__ == "__main__":
    main()
