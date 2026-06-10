import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import os

# ✅ Check GPU
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Dataset path
data_dir = "PlantVillage"

# Settings
img_size = (224, 224)
batch_size = 32

# 🔥 Data Augmentation (improved)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=(0.8, 1.2),
    shear_range=0.15,
    fill_mode="nearest"
)

# Validation datagen (no augmentation, only rescale)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    seed=42,
    shuffle=True
)

# Validation data (using separate datagen without augmentation)
val_data = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    seed=42,
    shuffle=False
)

# ✅ Compute Class Weights (handles imbalance)
print("\n📊 Computing class weights for imbalanced classes...")
class_counts = {}
for cls_name in sorted(os.listdir(data_dir)):
    cls_path = os.path.join(data_dir, cls_name)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[cls_name] = count

total_samples = sum(class_counts.values())
n_classes = len(class_counts)
class_weights = {}
for idx, (cls_name, count) in enumerate(sorted(class_counts.items())):
    weight = total_samples / (n_classes * count)
    class_weights[idx] = weight
    if weight > 2.0:
        print(f"  ⚠️ Underrepresented: {cls_name} ({count} images, weight={weight:.2f})")
    else:
        print(f"  ✅ {cls_name} ({count} images, weight={weight:.2f})")

# ✅ Load MobileNetV2 (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# 🔒 Freeze base model (fast training)
base_model.trainable = False

# Add custom layers (stronger classifier head)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(train_data.num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="saved_models/MobileNetV2_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
]

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 🚀 Train (Phase 1)
print("\n🚀 Training Phase 1 (Frozen Base Model)...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    callbacks=callbacks,
    class_weight=class_weights
)

# 🔥 Fine-tuning (unfreeze top layers)
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile with lower LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 🚀 Train (Phase 2)
print("\n🔥 Fine-tuning Phase...")
history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weights
)

# 💾 Save model
os.makedirs("saved_models", exist_ok=True)
model.save("plant_disease_model.h5")
model.save("saved_models/MobileNetV2_final.keras")
print("\n✅ Model saved successfully!")

# 💾 Save class names
class_names = list(train_data.class_indices.keys())

with open("class_names.txt", "w") as f:
    for item in class_names:
        f.write(item + "\n")

print("\n✅ Class names saved!")

# 📊 Final Evaluation
print("\n📊 Final Evaluation on Validation Set:")
val_loss, val_acc = model.evaluate(val_data, verbose=1)
print(f"\n🎯 Validation Accuracy: {val_acc:.4f}")
print(f"📉 Validation Loss: {val_loss:.4f}")

print("\nClasses:", class_names)