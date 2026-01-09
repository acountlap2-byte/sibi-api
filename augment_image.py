import cv2
import numpy as np
import os
import random

# =========================
# KONFIGURASI
# =========================
INPUT_DIR = "datasibi"                 # folder dataset asli
OUTPUT_DIR = "dataset_augmented"      # folder hasil augmentasi

AUG_PER_IMAGE = 3                     # berapa variasi per gambar

# =========================
# FUNGSI AUGMENTASI
# =========================

def random_rotate(img, angle_range=(-10, 10)):
    h, w = img.shape[:2]
    angle = random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def random_zoom(img, zoom_range=(0.9, 1.1)):
    h, w = img.shape[:2]
    zoom = random.uniform(*zoom_range)
    new_h, new_w = int(h * zoom), int(w * zoom)
    img_zoom = cv2.resize(img, (new_w, new_h))

    if zoom < 1:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        img_zoom = cv2.copyMakeBorder(
            img_zoom, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        img_zoom = img_zoom[start_h:start_h+h, start_w:start_w+w]

    return img_zoom

def random_brightness(img, range_val=(-40, 40)):
    value = random.randint(*range_val)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    noisy_img = img.astype(np.int16) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# =========================
# PROSES AUGMENTASI
# =========================

for label in os.listdir(INPUT_DIR):
    input_label_dir = os.path.join(INPUT_DIR, label)
    output_label_dir = os.path.join(OUTPUT_DIR, label)

    if not os.path.isdir(input_label_dir):
        continue

    os.makedirs(output_label_dir, exist_ok=True)

    for filename in os.listdir(input_label_dir):
        img_path = os.path.join(input_label_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # simpan gambar asli
        cv2.imwrite(os.path.join(output_label_dir, filename), img)

        # buat variasi
        for i in range(AUG_PER_IMAGE):
            aug = img.copy()

            if random.random() < 0.7:
                aug = random_rotate(aug)
            if random.random() < 0.5:
                aug = random_zoom(aug)
            if random.random() < 0.5:
                aug = random_brightness(aug)
            if random.random() < 0.3:
                aug = random_noise(aug)

            new_name = f"{os.path.splitext(filename)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(output_label_dir, new_name), aug)

print("✅ Augmentasi gambar selesai")
