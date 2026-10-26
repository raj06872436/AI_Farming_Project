"""
LesionNet — a plant-disease classifier that must point at the disease first.

    photo ──► backbone (EfficientNetB0 / MobileNetV2, ImageNet-pretrained)
                 │  multi-scale features (stride 8, 16, 32)
                 ├──► FPN-lite decoder ──► segmentation: background / healthy leaf / lesion  (56×56)
                 │                                │
                 └──► class features (14×14) ─────┤  lesion-gated pooling
                                                  ▼
                       [ features pooled over LESION pixels ‖ features pooled over LEAF pixels ‖ lesion area ]
                                                  ▼
                                           disease class (15)

The classifier only ever sees features averaged over pixels the segmentation head
marked as leaf or lesion, so background pixels cannot contribute to the decision —
the predicted lesion map *is* the explanation, faithful by construction. By default
the gate is detached (stop-gradient), so the classification loss cannot move the gate
towards convenient background regions; only the segmentation labels shape it.

Works with tf.keras on TensorFlow 2.13+ (Keras 2 or Keras 3).
"""
import tensorflow as tf

layers = tf.keras.layers

BACKBONES = {
    # name: (constructor, feature layer names at stride 8 / 16 / 32, input scaling from [0,1])
    "EfficientNetB0": (tf.keras.applications.EfficientNetB0,
                       ("block3b_add", "block5c_add", "top_activation"), (255.0, 0.0)),
    "MobileNetV2": (tf.keras.applications.MobileNetV2,
                    ("block_6_expand_relu", "block_13_expand_relu", "out_relu"), (2.0, -1.0)),
}
NUM_SEG = 3  # 0 background, 1 healthy leaf, 2 lesion


@tf.keras.utils.register_keras_serializable(package="agrix")
class LesionGatedPool(layers.Layer):
    """
    Pools class features over (a) predicted lesion pixels and (b) predicted leaf pixels.
    Inputs: [features (B,h,w,C), seg_probs (B,H,W,3)] → output (B, 2C + 1).
    """

    def __init__(self, detach_gate=True, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.detach_gate = detach_gate
        self.eps = eps

    def call(self, inputs):
        feats, probs = inputs
        feats = tf.cast(feats, tf.float32)
        probs = tf.cast(probs, tf.float32)
        if self.detach_gate:
            probs = tf.stop_gradient(probs)
        # Shrink the 56x56 seg map to the 14x14 feature grid by averaging 4x4 blocks.
        # (Same result as an "area" resize, but avg-pool also works under XLA on GPU.)
        fh, ph = feats.shape[1], probs.shape[1]
        if fh and ph and ph % fh == 0:
            if ph != fh:
                k = ph // fh
                probs = tf.nn.avg_pool2d(probs, ksize=k, strides=k, padding="VALID")
        else:
            probs = tf.image.resize(probs, tf.shape(feats)[1:3], method="bilinear")
        lesion = probs[..., 2:3]
        leaf = probs[..., 1:2] + lesion
        def pool(w):
            return tf.reduce_sum(feats * w, axis=[1, 2]) / (tf.reduce_sum(w, axis=[1, 2]) + self.eps)
        area = tf.reduce_sum(lesion, axis=[1, 2]) / (tf.reduce_sum(leaf, axis=[1, 2]) + self.eps)
        return tf.concat([pool(lesion), pool(leaf), area], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"detach_gate": self.detach_gate, "eps": self.eps})
        return cfg


def _conv_bn(x, filters, k, name):
    x = layers.Conv2D(filters, k, padding="same", use_bias=False, name=name + "_conv")(x)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    return layers.Activation("relu", name=name + "_relu")(x)


def build_lesionnet(num_classes=15, backbone="EfficientNetB0", weights="imagenet", input_size=224,
                    detach_gate=True, dropout=0.3, pool="gated"):
    """
    pool="gated" → LesionNet (classifier sees only leaf/lesion-pooled features).
    pool="gap"   → ablation: same network, ordinary global average pooling over the whole image.
    """
    ctor, feat_names, (scale, offset) = BACKBONES[backbone]
    inp = layers.Input((input_size, input_size, 3), name="image")          # RGB in [0, 1]
    x = layers.Rescaling(scale, offset=offset, name="to_backbone_range")(inp)
    base = ctor(include_top=False, weights=weights, input_shape=(input_size, input_size, 3))
    feat_model = tf.keras.Model(base.input, [base.get_layer(n).output for n in feat_names], name="backbone_features")
    f8, f16, f32 = feat_model(x)

    # FPN-lite decoder → segmentation
    p32 = layers.Conv2D(128, 1, name="lat32")(f32)
    p16 = layers.Add(name="fpn16")([layers.Conv2D(128, 1, name="lat16")(f16),
                                    layers.UpSampling2D(2, interpolation="bilinear", name="up32")(p32)])
    p8 = layers.Add(name="fpn8")([layers.Conv2D(128, 1, name="lat8")(f8),
                                  layers.UpSampling2D(2, interpolation="bilinear", name="up16")(p16)])
    s = _conv_bn(p8, 128, 3, "seg1")
    s = _conv_bn(s, 64, 3, "seg2")
    s = layers.UpSampling2D(2, interpolation="bilinear", name="seg_up")(s)              # 56×56
    seg = layers.Conv2D(NUM_SEG, 1, name="seg_logits")(s)
    seg = layers.Activation("softmax", dtype="float32", name="seg")(seg)

    # Class features at stride 16 (14×14) with context from stride 32
    c = layers.Concatenate(name="cls_concat")([f16, layers.UpSampling2D(2, interpolation="bilinear", name="cls_up")(f32)])
    c = _conv_bn(c, 256, 1, "cls_feat")
    if pool == "gated":
        pooled = LesionGatedPool(detach_gate=detach_gate, name="lesion_gated_pool")([c, seg])
    else:
        pooled = layers.GlobalAveragePooling2D(name="global_avg_pool")(c)
    h = layers.Dropout(dropout, name="cls_dropout")(pooled)
    h = layers.Dense(256, activation="relu", name="cls_hidden")(h)
    out = layers.Dense(num_classes, name="cls_logits")(h)
    out = layers.Activation("softmax", dtype="float32", name="cls")(out)

    return tf.keras.Model(inp, {"cls": out, "seg": seg}, name=f"LesionNet_{backbone}")


def backbone_of(model):
    return model.get_layer("backbone_features")


def seg_loss(lesion_weight=3.0):
    """Pixel cross-entropy on (bg, leaf, lesion); lesion pixels count `lesion_weight`× (they are rare)."""
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true, -1), tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        w = tf.where(y_true == 2, lesion_weight, 1.0)
        return tf.reduce_mean(ce * w, axis=[1, 2])
    loss.__name__ = "seg_ce"
    return loss


def load_lesionnet(path):
    return tf.keras.models.load_model(path, compile=False,
                                      custom_objects={"LesionGatedPool": LesionGatedPool})
