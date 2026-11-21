# ==========================================================================
#  PEANUT GOOD/BAD  --  MobileNetV2 training (Colab)
#  Har "# ===== CELL N =====" ko ALAG Colab cell mein paste karo.
#  Pehle: Runtime -> Change runtime type -> GPU (T4)
# ==========================================================================


# ===== CELL 1 : dataset load karo =========================================
# dataset_processed.zip ko Google Drive pe daalo, phir mount + unzip.

from google.colab import drive
drive.mount('/content/drive')

# apne zip ka sahi path do:
!cp "/content/drive/MyDrive/dataset_processed.zip" /content/
!unzip -q -o /content/dataset_processed.zip -d /content/

import os
data_dir = "/content/dataset_processed"
n_good = len(os.listdir(f"{data_dir}/good"))
n_bad  = len(os.listdir(f"{data_dir}/bad"))
print("good:", n_good, " bad:", n_bad)


# ===== CELL 2 : train / val / test split ==================================
import tensorflow as tf

IMG = 224
BATCH = 16

# 70% train, 30% baaki (us 30% ko aage val+test mein todenge)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.30, subset="training", seed=123,
    image_size=(IMG, IMG), batch_size=BATCH, label_mode="binary")

rest_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.30, subset="validation", seed=123,
    image_size=(IMG, IMG), batch_size=BATCH, label_mode="binary")

# IMPORTANT: labels alphabetical -> bad=0, good=1
class_names = train_ds.class_names
print("class order:", class_names, " (index 0 = bad, 1 = good)")

# rest ko aadha val, aadha test
n_rest = tf.data.experimental.cardinality(rest_ds)
test_ds = rest_ds.take(n_rest // 2)
val_ds  = rest_ds.skip(n_rest // 2)

# speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)


# ===== CELL 3 : model banao (MobileNetV2 + head) ==========================
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# data augmentation (training mein thoda vary -> overfitting kam)
augment = tf.keras.Sequential([
    layers.RandomRotation(0.3),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
])

base = MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False   # Phase 1: base freeze

inputs = layers.Input(shape=(IMG, IMG, 3))
x = augment(inputs)
x = preprocess_input(x)           # pixels -> -1..1  (yahi normalization)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


# ===== CELL 4 : class_weight (halka imbalance safety) =====================
total = n_good + n_bad
class_weight = {0: total / (2 * n_bad),    # bad
                1: total / (2 * n_good)}   # good
print("class_weight:", class_weight)


# ===== CELL 5 : PHASE 1 -- head train (base frozen) =======================
hist1 = model.fit(train_ds, validation_data=val_ds,
                  epochs=15, class_weight=class_weight)


# ===== CELL 6 : PHASE 2 -- fine-tune (base unfreeze, low LR) ===============
base.trainable = True
# sabse pehli layers freeze rakho (general features), aakhri fine-tune karo
for layer in base.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),   # bahut low LR
              loss="binary_crossentropy", metrics=["accuracy"])

hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=10, class_weight=class_weight)


# ===== CELL 7 : EVALUATE (test set -- jo kabhi nahi dekha) ================
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

loss, acc = model.evaluate(test_ds)
print(f"\nTEST accuracy: {acc:.4f}   loss: {loss:.4f}\n")

y_true, y_pred = [], []
for imgs, labels in test_ds:
    probs = model.predict(imgs, verbose=0)
    y_pred.extend((probs > 0.5).astype(int).flatten())
    y_true.extend(labels.numpy().astype(int).flatten())

print("Confusion matrix (rows=true, cols=pred)  [0=bad, 1=good]:")
print(confusion_matrix(y_true, y_pred))
print("\nReport:")
print(classification_report(y_true, y_pred, target_names=["bad", "good"]))


# ===== CELL 8 : SAVE (version-proof: weights only) ========================
model.save_weights("/content/peanut.weights.h5")
!cp /content/peanut.weights.h5 "/content/drive/MyDrive/peanut.weights.h5"
print("Saved weights to Drive. Download karke laptop pe le lo.")
