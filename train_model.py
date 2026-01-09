# ===============================
# NON-GUI BACKEND (ANTI ERROR TKINTER)
# ===============================
import matplotlib
matplotlib.use("Agg")

# ===============================
# IMPORT
# ===============================
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os

# ===============================
# FIX SEED (WAJIB UNTUK SKRIPSI)
# ===============================
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ===============================
# LOAD DATA
# ===============================
X_train = np.load("preprocessed/X_train.npy")
y_train = np.load("preprocessed/y_train.npy")
X_test  = np.load("preprocessed/X_test.npy")
y_test  = np.load("preprocessed/y_test.npy")

num_classes = len(np.unique(y_train))
print("Jumlah kelas:", num_classes)

# ===============================
# CLASS WEIGHT (DATA TIDAK SEIMBANG)
# ===============================
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

print("Class weight siap digunakan")

# ===============================
# MODEL (MLP – COCOK UNTUK LANDMARK STATIS)
# ===============================
model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# TRAINING
# ===============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ===============================
# SIMPAN MODEL
# ===============================
os.makedirs("model", exist_ok=True)
model.save("model/model_sibi_landmark_FINAL.keras")
print("✅ Model berhasil disimpan")

# ===============================
# SIMPAN GRAFIK (TANPA SHOW)
# ===============================
os.makedirs("hasil_evaluasi", exist_ok=True)

# ---- Grafik Akurasi ----
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Grafik Akurasi Training dan Validation")
plt.legend()
plt.grid(True)
plt.savefig("hasil_evaluasi/grafik_akurasi.png")
plt.close()

# ---- Grafik Loss ----
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Grafik Loss Training dan Validation")
plt.legend()
plt.grid(True)
plt.savefig("hasil_evaluasi/grafik_loss.png")
plt.close()

print("Grafik akurasi & loss berhasil disimpan")
print("TRAINING SELESAI TANPA ERROR")
