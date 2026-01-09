import os
import cv2
import numpy as np
import mediapipe as mp

# ===============================
# KONFIGURASI
# ===============================
DATA_DIR = "datasibi"          # folder A–Z berisi JPG
OUT_DIR = "landmark_data"      # hasil landmark (.npy)
IMG_SIZE = 256

# ===============================
# MEDIAPIPE HANDS
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,    # gambar statis
    max_num_hands=1,
    min_detection_confidence=0.7
)

os.makedirs(OUT_DIR, exist_ok=True)

total_images = 0
total_success = 0

# ===============================
# EKSTRAKSI
# ===============================
for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    out_label_dir = os.path.join(OUT_DIR, label)
    os.makedirs(out_label_dir, exist_ok=True)

    print(f"\n🔤 Proses huruf: {label}")

    for file in os.listdir(label_path):
        if not file.lower().endswith(".jpg"):
            continue

        total_images += 1
        img_path = os.path.join(label_path, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            continue

        # Ambil 1 tangan
        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks, dtype=np.float32)  # (21, 3)

        # Simpan
        out_name = file.replace(".jpg", ".npy")
        np.save(os.path.join(out_label_dir, out_name), landmarks)

        total_success += 1

    print(f"Selesai huruf {label}")

hands.close()

print("\n===============================")
print(f"Total gambar dibaca   : {total_images}")
print(f"Landmark berhasil     : {total_success}")
print(f"Gagal terdeteksi tangan: {total_images - total_success}")
print("python exEKSTRAKSI LANDMARK SELESAI")
