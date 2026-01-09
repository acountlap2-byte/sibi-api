import os
from PIL import Image
import pillow_heif

# agar PIL bisa baca HEIC
pillow_heif.register_heif_opener()

SOURCE_DIR = "datasibi"     # data asli
OUTPUT_DIR = "processed"    # hasil akhir
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".heic")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in sorted(os.listdir(SOURCE_DIR)):
    label_dir = os.path.join(SOURCE_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    out_label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_label_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(label_dir)
        if f.lower().endswith(SUPPORTED_EXT)
    )

    count = 1
    for file in files:
        src_path = os.path.join(label_dir, file)
        dst_name = f"{label}_{count:04d}.jpg"
        dst_path = os.path.join(out_label_dir, dst_name)

        try:
            img = Image.open(src_path).convert("RGB")
            img.save(dst_path, "JPEG", quality=95)

            print(f"✔ {label}: {file} → {dst_name}")
            count += 1

        except Exception as e:
            print(f"❌ Gagal {file}: {e}")

    print(f"✅ {label} selesai: {count-1} file JPG")
