import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =================================================
# KONFIGURASI
# =================================================
MODEL_PATH = "model/model_sibi_landmark_FINAL.keras"
DATA_DIR = "preprocessed"
OUTPUT_DIR = "hasil_evaluasi"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label kelas (A–Z)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =================================================
# LOAD MODEL
# =================================================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model berhasil dimuat")

# =================================================
# LOAD DATA
# =================================================
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Data berhasil dimuat")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# =================================================
# ============ EVALUASI TRAINING ==================
# =================================================
y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average="weighted", zero_division=0)
train_recall = recall_score(y_train, y_train_pred, average="weighted", zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)

print("\n==============================================")
print("CLASSIFICATION REPORT (TRAINING)")
print("==============================================")
print(classification_report(
    y_train, y_train_pred, target_names=labels, digits=4
))

print("RINGKASAN METRIK TRAINING")
print(f"Akurasi   : {train_accuracy*100:.2f}%")
print(f"Precision : {train_precision*100:.2f}%")
print(f"Recall    : {train_recall*100:.2f}%")
print(f"F1-score  : {train_f1*100:.2f}%")

# =================================================
# ============ EVALUASI DATA UJI ==================
# =================================================
y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

print("\n==============================================")
print("CLASSIFICATION REPORT (DATA UJI)")
print("==============================================")
print(classification_report(
    y_test, y_test_pred, target_names=labels, digits=4
))

print("RINGKASAN METRIK DATA UJI")
print(f"Akurasi   : {test_accuracy*100:.2f}%")
print(f"Precision : {test_precision*100:.2f}%")
print(f"Recall    : {test_recall*100:.2f}%")
print(f"F1-score  : {test_f1*100:.2f}%")

# =================================================
# ============ CONFUSION MATRIX (JELAS) ===========
# =================================================
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,          # tampilkan angka
    fmt="d",             # angka bulat
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    cbar=True
)

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix Data Uji", fontsize=14)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_jelas.png"), dpi=300)
plt.close()

print("\n✅ Confusion matrix versi jelas berhasil disimpan")
print(f"📁 Lokasi: {os.path.join(OUTPUT_DIR, 'confusion_matrix_jelas.png')}")
