# ==============================================================================
# src/models/vit_model.py
# Vision Transformer (ViT) model builder — OPTIONAL.
# Uses a custom lightweight ViT implementation built with Keras layers
# to avoid external dependency issues.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config.settings import Config
from src.models.base_model import BaseModelBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PatchEmbedding(layers.Layer):
    """Extract patches from images and project them into an embedding space."""

    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size
        )

    def call(self, x):
        # x shape: (batch, height, width, channels)
        x = self.projection(x)  # (batch, num_patches_h, num_patches_w, embed_dim)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])  # (batch, num_patches, embed_dim)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return config


class TransformerBlock(layers.Layer):
    """Single transformer encoder block with multi-head attention and FFN."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Multi-head self-attention with residual connection
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return config


class ViTBuilder(BaseModelBuilder):
    """
    Builder for a lightweight Vision Transformer (ViT).
    Custom Keras implementation — no external dependencies required.

    Architecture:
    - Patch embedding (16x16 patches from 224x224 input = 196 patches)
    - Positional embedding (learnable)
    - 6 Transformer encoder blocks
    - Classification head
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.patch_size = 16
        self.embed_dim = 256
        self.num_heads = 8
        self.ff_dim = 512
        self.num_transformer_blocks = 6
        self.dropout_rate = 0.1

    def get_base_model(self) -> tf.keras.Model:
        """
        Build custom ViT backbone. Since this is not a pretrained model,
        it trains from scratch (no ImageNet weights available for custom ViT).
        """
        img_size = self.data_cfg.image_size
        num_patches = (img_size // self.patch_size) ** 2

        inputs = layers.Input(shape=self.data_cfg.image_shape)

        # Patch embedding
        patch_embed = PatchEmbedding(self.patch_size, self.embed_dim)(inputs)

        # Learnable positional embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embed = layers.Embedding(
            input_dim=num_patches, output_dim=self.embed_dim
        )(positions)
        x = patch_embed + pos_embed

        # Transformer encoder blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate
            )(x)

        # Global average over patch dimension
        x = layers.GlobalAveragePooling1D()(x)

        model = models.Model(inputs=inputs, outputs=x, name="ViT_backbone")
        logger.info(f"Built custom ViT backbone | Patches: {num_patches} | "
                    f"Embed dim: {self.embed_dim} | Blocks: {self.num_transformer_blocks}")
        return model

    def build(self, num_classes=None, dense_units_1=None, dense_units_2=None, dropout_rate=None):
        """
        Override build to handle custom ViT architecture differently.
        ViT backbone outputs a 1D vector, not spatial feature maps.
        """
        num_classes = num_classes or self.data_cfg.num_classes
        dense_units_1 = dense_units_1 or self.training_cfg.dense_units_1
        dropout_rate = dropout_rate or self.training_cfg.dropout_rate

        base = self.get_base_model()
        # Base model already outputs pooled features

        x = base.output
        x = layers.BatchNormalization(name="batch_norm")(x)
        x = layers.Dense(dense_units_1, activation="relu", name="dense_1")(x)
        x = layers.Dropout(dropout_rate, name="dropout_1")(x)
        output = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = models.Model(inputs=base.input, outputs=output, name="ViT")

        logger.info(
            f"Built ViT | Total params: {model.count_params():,} | "
            f"Note: Training from scratch (no pretrained weights)"
        )
        return model

    def get_model_name(self) -> str:
        return "ViT"

    def get_last_conv_layer_name(self) -> str:
        """ViT doesn't have traditional conv layers — return patch embedding."""
        return "patch_embedding"
