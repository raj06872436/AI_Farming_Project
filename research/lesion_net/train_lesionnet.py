"""
Step 2 — train LesionNet (or one of its ablations) on the GPU.

Run inside WSL2 with your tf_gpu environment, from the project folder:
    python research/lesion_net/train_lesionnet.py                           # full LesionNet, EfficientNetB0
    python research/lesion_net/train_lesionnet.py --pool gap --lambda-seg 0 --bg-swap 0 --tag A0_plain
    (see research/lesion_net/README.md for the full ablation list)

Two phases: (1) backbone frozen, heads learn; (2) whole network fine-tuned with a
cosine learning-rate decay. The best epoch (validation accuracy) is saved to
saved_models/LesionNet_<tag>.keras together with its training history.
"""
import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from lesionnet import backbone_of, build_lesionnet, seg_loss  # noqa: E402

IMG, SEG = 224, 56


def read_manifest(path, split, limit_per_class=0):
    rows = [r for r in csv.DictReader(open(path)) if r["split"] == split]
    if limit_per_class:
        seen, keep = {}, []
        for r in rows:
            seen[r["class"]] = seen.get(r["class"], 0) + 1
            if seen[r["class"]] <= limit_per_class:
                keep.append(r)
        rows = keep
    return rows


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def make_dataset(rows, classes, batch, training, bg_swap=0.5, class_weights=None, seed=0):
    idx = {c: i for i, c in enumerate(classes)}
    imgs = [_abs(r["image"]) for r in rows]
    lbls = [_abs(r["label"]) for r in rows]
    ys = [idx[r["class"]] for r in rows]
    sw_seg = [float(r["seg_weight"]) for r in rows]
    sw_cls = [float(class_weights[idx[r["class"]]]) if class_weights is not None else 1.0 for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((imgs, lbls, ys, sw_cls, sw_seg))
    if training:
        ds = ds.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)

    def load(ip, lp, y, wc, ws):
        img = tf.io.decode_image(tf.io.read_file(ip), channels=3, expand_animations=False)
        img = tf.image.resize(tf.cast(img, tf.float32) / 255.0, (IMG, IMG))
        lbl = tf.io.decode_png(tf.io.read_file(lp), channels=1)
        lbl = tf.image.resize(lbl, (IMG, IMG), method="nearest")
        return img, lbl, y, wc, ws

    def augment(img, lbl, y, wc, ws):
        both = tf.concat([img, tf.cast(lbl, tf.float32)], -1)
        both = tf.image.random_flip_left_right(both)
        both = tf.image.random_flip_up_down(both)
        both = tf.image.rot90(both, tf.random.uniform([], 0, 4, tf.int32))
        img, lbl = both[..., :3], tf.cast(both[..., 3:], tf.uint8)
        img = tf.image.random_brightness(img, 0.12)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        img = tf.image.random_hue(img, 0.02)
        img = tf.clip_by_value(img, 0.0, 1.0)
        # Background swap: replace everything that is not leaf with a random surface,
        # so the network cannot learn "this table colour means this disease".
        if bg_swap > 0:
            def swap():
                base = tf.random.uniform([1, 1, 3], 0.05, 0.95)
                grad = tf.linspace(0.0, 1.0, IMG)[:, None, None] * tf.random.uniform([1, 1, 3], -0.3, 0.3)
                noise = tf.random.normal([IMG, IMG, 3], 0.0, tf.random.uniform([], 0.0, 0.08))
                new_bg = tf.clip_by_value(base + grad + noise, 0.0, 1.0)
                bg = tf.cast(lbl == 0, tf.float32)
                return img * (1.0 - bg) + new_bg * bg
            img = tf.cond(tf.random.uniform([]) < bg_swap, swap, lambda: img)
        return img, lbl, y, wc, ws

    def finish(img, lbl, y, wc, ws):
        seg = tf.image.resize(lbl, (SEG, SEG), method="nearest")
        return img, {"cls": y, "seg": seg}, {"cls": wc, "seg": ws}

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(finish, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=training).prefetch(tf.data.AUTOTUNE)


JIT = False  # XLA + mixed precision hits a dtype bug in ReluGrad on some TF builds; --xla turns it on


def compile_model(model, lr, lambda_seg):
    model.compile(
        jit_compile=JIT,
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={"cls": tf.keras.losses.SparseCategoricalCrossentropy(), "seg": seg_loss()},
        loss_weights={"cls": 1.0, "seg": lambda_seg},
        metrics={"cls": ["accuracy"]},
    )


def set_backbone_trainable(model, trainable):
    bb = backbone_of(model)
    bb.trainable = trainable
    if trainable:  # keep BatchNorm statistics frozen for stable fine-tuning with small batches
        for layer in bb.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


def init_backbone_from(model, path):
    """Copy backbone weights from an existing trained classifier (e.g. saved_models/EfficientNetB0_best.keras)."""
    src = tf.keras.models.load_model(path, compile=False)
    src_bb = next(l for l in src.layers if hasattr(l, "layers") and len(l.layers) > 50)
    copied = 0
    for layer in backbone_of(model).layers:
        if not layer.weights:
            continue
        try:
            sl = src_bb.get_layer(layer.name)
        except ValueError:
            continue
        if len(sl.weights) == len(layer.weights):
            layer.set_weights(sl.get_weights())
            copied += 1
    print(f"initialised {copied} backbone layers from {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "pseudo_masks", "manifest.csv"))
    ap.add_argument("--backbone", default="EfficientNetB0", choices=["EfficientNetB0", "MobileNetV2"])
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    ap.add_argument("--pool", default="gated", choices=["gated", "gap"])
    ap.add_argument("--no-detach", action="store_true", help="let classification gradients move the gate")
    ap.add_argument("--lambda-seg", type=float, default=1.0)
    ap.add_argument("--bg-swap", type=float, default=0.5, help="probability of replacing the background")
    ap.add_argument("--epochs1", type=int, default=3)
    ap.add_argument("--epochs2", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--limit-per-class", type=int, default=0, help="quick tests only")
    ap.add_argument("--init-backbone", default="", help="copy backbone weights from a trained classifier .keras "
                    "(note: that model may have seen your test images — use ImageNet weights for the paper)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-mixed-precision", action="store_true")
    ap.add_argument("--xla", action="store_true", help="enable XLA compilation (faster if it works on your TF build)")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    global JIT
    JIT = args.xla
    if gpus and not args.no_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"GPU: {gpus[0].name if gpus else 'none (CPU — will be slow)'}")

    classes = [l.strip() for l in open(os.path.join(ROOT, "class_names.txt")) if l.strip()]
    tr = read_manifest(args.manifest, "train", args.limit_per_class)
    va = read_manifest(args.manifest, "val", args.limit_per_class)
    counts = np.bincount([classes.index(r["class"]) for r in tr], minlength=len(classes)).astype(float)
    cw = np.clip(counts.sum() / (len(classes) * np.maximum(counts, 1)), 0.5, 3.0)
    print(f"train {len(tr)}  val {len(va)}  | seg-supervised train images: {sum(float(r['seg_weight']) > 0 for r in tr)}")

    ds_tr = make_dataset(tr, classes, args.batch, True, args.bg_swap, cw)
    ds_va = make_dataset(va, classes, args.batch, False)

    model = build_lesionnet(len(classes), args.backbone, None if args.weights == "none" else args.weights,
                            detach_gate=not args.no_detach, pool=args.pool)
    if args.init_backbone:
        init_backbone_from(model, args.init_backbone)
    tag = args.tag or f"{args.backbone}_{args.pool}"
    out = os.path.join(ROOT, "saved_models", f"LesionNet_{tag}.keras")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(out, monitor="val_cls_accuracy", mode="max", save_best_only=True,
                                              verbose=1)
    history = {}
    t0 = time.time()

    # Phase 1 — heads only
    set_backbone_trainable(model, False)
    compile_model(model, args.lr1, args.lambda_seg)
    h1 = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs1, callbacks=[ckpt], verbose=2)
    for k, v in h1.history.items():
        history.setdefault(k, []).extend([float(x) for x in v])

    # Phase 2 — fine-tune everything, cosine decay
    if args.epochs2 > 0:
        set_backbone_trainable(model, True)
        steps = max(1, len(tr) // args.batch) * args.epochs2
        sched = tf.keras.optimizers.schedules.CosineDecay(args.lr2, steps, alpha=0.01)
        compile_model(model, sched, args.lambda_seg)
        early = tf.keras.callbacks.EarlyStopping(monitor="val_cls_accuracy", mode="max", patience=5,
                                                 restore_best_weights=False)
        h2 = model.fit(ds_tr, validation_data=ds_va, initial_epoch=args.epochs1,
                       epochs=args.epochs1 + args.epochs2, callbacks=[ckpt, early], verbose=2)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend([float(x) for x in v])

    meta = {"tag": tag, "args": vars(args), "minutes": round((time.time() - t0) / 60, 1),
            "best_val_cls_accuracy": max(history.get("val_cls_accuracy", [0])), "history": history}
    with open(out.replace(".keras", "_history.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"saved {out}  (best val accuracy {meta['best_val_cls_accuracy']:.4f}, {meta['minutes']} min)")


if __name__ == "__main__":
    main()
