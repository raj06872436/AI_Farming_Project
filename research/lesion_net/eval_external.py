"""
Cross-dataset test: models trained on PlantVillage (lab photos, plain backgrounds),
tested on PlantDoc (real field / internet photos, cluttered backgrounds).

Only the 12 PlantDoc classes that also exist in AGRI-X's 15 classes are used
(PlantDoc has no potato-healthy, tomato target-spot or spider-mite class).
No model is trained or tuned on PlantDoc.

Get the data (needs internet, ~300 MB):
    python research/datasets/fetch_plantdoc.py
Run:
    python research/lesion_net/eval_external.py
    python research/lesion_net/eval_external.py --data research/datasets/PlantDoc

Metrics
  acc15       – normal 15-way prediction
  acc12       – prediction restricted to the 12 classes PlantDoc has (fairer: the model
                 cannot lose points by naming a class that is not in the test set)
  macro_f1_12 – macro-F1 of the restricted prediction
  crop_acc    – right crop (pepper / potato / tomato), disease ignored
  sick_acc    – right "healthy vs diseased" call
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

from attention_lesion_study import load_model  # noqa: E402
from lesionnet import load_lesionnet  # noqa: E402

SIZE = 224
BASELINES = ["MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def crop_of(c):
    return c.split("_")[0]


def load_images(data, classes):
    xs, ys, names = [], [], []
    for ci, c in enumerate(classes):
        folder = os.path.join(data, c)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith(EXTS):
                continue
            try:
                img = Image.open(os.path.join(folder, f)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            except Exception:
                continue
            xs.append(np.asarray(img, np.uint8))
            ys.append(ci)
            names.append(f"{c}/{f}")
    return np.stack(xs), np.array(ys), names


def predict(model, xs, is_lesionnet, batch=16):
    cls, les = [], []
    for i in range(0, len(xs), batch):
        x = xs[i:i + batch].astype(np.float32) / 255.0
        o = model(x, training=False)
        if is_lesionnet:
            cls.append(o["cls"].numpy())
            seg = o["seg"].numpy()
            leaf = seg[..., 1].sum((1, 2)) + seg[..., 2].sum((1, 2))
            les.append(seg[..., 2].sum((1, 2)) / np.maximum(leaf, 1e-6) * 100)
        else:
            cls.append(o.numpy())
    return np.concatenate(cls), (np.concatenate(les) if les else None)


def macro_f1(y, p, labels):
    f1s = []
    for c in labels:
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


def score(name, probs, les, ys, classes):
    present = sorted(set(ys.tolist()))
    p15 = probs.argmax(1)
    masked = np.full_like(probs, -1.0)
    masked[:, present] = probs[:, present]
    p12 = masked.argmax(1)
    crops = np.array([crop_of(c) for c in classes])
    healthy = np.array(["healthy" in c for c in classes])
    res = {
        "Model": name,
        "acc15": float(np.mean(p15 == ys)),
        "acc12": float(np.mean(p12 == ys)),
        "macro_f1_12": macro_f1(ys, p12, present),
        "crop_acc": float(np.mean(crops[p15] == crops[ys])),
        "sick_acc": float(np.mean(healthy[p15] == healthy[ys])),
    }
    if les is not None:
        res["lesion_pct_healthy"] = float(np.mean(les[healthy[ys]]))
        res["lesion_pct_diseased"] = float(np.mean(les[~healthy[ys]]))
    per_class = {classes[c]: float(np.mean(p12[ys == c] == c)) for c in present}
    return res, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "research", "datasets", "PlantDoc"))
    ap.add_argument("--lesionnets", nargs="*", default=None)
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    classes = [l.strip() for l in open(os.path.join(ROOT, "class_names.txt")) if l.strip()]
    xs, ys, _ = load_images(args.data, classes)
    print(f"PlantDoc images: {len(xs)} in {len(set(ys.tolist()))} classes")

    nets = args.lesionnets if args.lesionnets is not None else sorted(
        glob.glob(os.path.join(ROOT, "saved_models", "LesionNet_*.keras")))
    results, per_class = [], {}
    jobs = [(b, False) for b in args.baselines.split(",") if b] + [(p, True) for p in nets]
    for ref, is_ln in jobs:
        t0 = time.time()
        model = load_lesionnet(ref) if is_ln else load_model(ref)
        name = os.path.basename(ref).replace(".keras", "") if is_ln else ref
        probs, les = predict(model, xs, is_ln)
        r, pc = score(name, probs, les, ys, classes)
        results.append(r)
        per_class[name] = pc
        print(f"{name}: {time.time() - t0:.0f}s  acc12 {r['acc12']:.3f}  crop {r['crop_acc']:.3f}")
        tf.keras.backend.clear_session()

    os.makedirs(args.out, exist_ok=True)
    keys = ["Model", "acc15", "acc12", "macro_f1_12", "crop_acc", "sick_acc", "lesion_pct_healthy", "lesion_pct_diseased"]
    with open(os.path.join(args.out, "plantdoc_summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in results:
            wr.writerow({k: r.get(k, "") for k in keys})
    with open(os.path.join(args.out, "plantdoc_per_class.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "n"] + list(per_class))
        for c in sorted({c for pc in per_class.values() for c in pc}):
            wr.writerow([c, int(np.sum(ys == classes.index(c)))] + [f"{per_class[m].get(c, float('nan')):.3f}" for m in per_class])
    nice = ["Model", "Accuracy (15-way)", "Accuracy (12 PlantDoc classes)", "Macro-F1 (12)", "Right crop",
            "Healthy vs diseased", "Lesion % healthy leaves", "Lesion % diseased leaves"]
    with open(os.path.join(args.out, "plantdoc_summary.md"), "w") as fh:
        healthy = np.array(["healthy" in c for c in classes])
        always_sick = float(np.mean(~healthy[ys]))
        fh.write(f"PlantDoc field photos, {len(xs)} images, 12 classes shared with PlantVillage. "
                 f"Chance: 15-way 0.067, 12-way 0.083. Answering 'diseased' every time scores "
                 f"{always_sick:.3f} on 'Healthy vs diseased'.\n\n")
        fh.write("| " + " | ".join(nice) + " |\n|" + "---|" * len(nice) + "\n")
        for r in results:
            cells = [r["Model"]] + [f"{r[k]:.3f}" if k in r else "—" for k in keys[1:]]
            fh.write("| " + " | ".join(cells) + " |\n")
    print(open(os.path.join(args.out, "plantdoc_summary.md")).read())


if __name__ == "__main__":
    main()
