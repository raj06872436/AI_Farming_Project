import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# ✅ Check GPU
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Dataset path
data_dir = "PlantVillage"

# Settings
img_size = (224, 224)
batch_size = 32

# 🔥 Data Augmentation (improves accuracy)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Training data
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset="training"
)

# Validation data
val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset="validation"
)

# ✅ Load MobileNetV2 (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# 🔒 Freeze base model (fast training)
base_model.trainable = False

# Add custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(train_data.num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

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
    epochs=5
)

# 🔥 Fine-tuning (unfreeze top layers)
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile
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
    epochs=15
)

# 💾 Save model
model.save("plant_disease_model.h5")
print("\n✅ Model saved successfully!")

# 💾 Save class names
class_names = list(train_data.class_indices.keys())

with open("class_names.txt", "w") as f:
    for item in class_names:
        f.write(item + "\n")

print("\n✅ Class names saved!")

print("\nClasses:", class_names)