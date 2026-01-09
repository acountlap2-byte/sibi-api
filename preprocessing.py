import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ===============================
DATA_DIR = "landmark_data"
OUT_DIR = "preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

X = []
y = []
labels = sorted(os.listdir(DATA_DIR))

label_map = {label: idx for idx, label in enumerate(labels)}

# ===============================
def normalize_landmark(lm):
    # relatif ke wrist
    lm = lm - lm[0]

    # mirror tangan kiri
    if lm[5][0] < lm[17][0]:
        lm[:, 0] = -lm[:, 0]

    # normalisasi skala
    max_val = np.max(np.abs(lm))
    if max_val > 0:
        lm = lm / max_val

    return lm.flatten()  # (63,)

# ===============================
for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        if not file.endswith(".npy"):
            continue

        lm = np.load(os.path.join(folder, file))  # (21,3)
        if lm.shape != (21, 3):
            continue

        feat = normalize_landmark(lm)
        X.append(feat)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("Total sample:", X.shape)

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# SAVE
# ===============================
np.save(f"{OUT_DIR}/X_train.npy", X_train)
np.save(f"{OUT_DIR}/X_test.npy", X_test)
np.save(f"{OUT_DIR}/y_train.npy", y_train)
np.save(f"{OUT_DIR}/y_test.npy", y_test)

joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")

print("Preprocessing selesai & data disimpan")
