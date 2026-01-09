import pandas as pd
import os

# =========================
# KONFIGURASI
# =========================
DATASET_DIR = "dataset" 
OUTPUT_DIR = "dataset_final"
OUTPUT_FILE = "dataset_final.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_data = []

print("📂 Membaca file CSV & menambahkan label...")

# =========================
# BACA CSV + TAMBAH LABEL
# =========================
for file in os.listdir(DATASET_DIR):
    if file.endswith(".csv"):
        label = os.path.splitext(file)[0]  # A.csv → A
        file_path = os.path.join(DATASET_DIR, file)

        print(f"  ✔ {file} → label = {label}")

        df = pd.read_csv(file_path)

        # Tambahkan kolom label
        df["label"] = label

        all_data.append(df)

# =========================
# VALIDASI
# =========================
if len(all_data) == 0:
    raise RuntimeError("❌ Tidak ada file CSV ditemukan")

# =========================
# GABUNG & SIMPAN
# =========================
final_df = pd.concat(all_data, ignore_index=True)

# Pindahkan kolom label ke depan
cols = ["label"] + [c for c in final_df.columns if c != "label"]
final_df = final_df[cols]

final_df.to_csv(
    os.path.join(OUTPUT_DIR, OUTPUT_FILE),
    index=False
)

print("\n✅ dataset_final.csv BERHASIL DIBUAT DENGAN LABEL")
print("📊 Shape:", final_df.shape)
print("📁 Lokasi:", os.path.join(OUTPUT_DIR, OUTPUT_FILE))
