# =====================================
# REALTIME DETECTION SIBI (FINAL - FIX)
# =====================================

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque, Counter

# =========================
# KONFIGURASI
# =========================
SEQUENCE_LENGTH = 30
NUM_FEATURES = 63

MODEL_PATH = "model/model_sibi_lstm_FINAL.keras"
SCALER_PATH = "preprocessed/scaler.pkl"
LABEL_ENCODER_PATH = "preprocessed/label_encoder.pkl"

CONF_THRESHOLD = 0.60      # NAIK → cegah huruf dipaksa muncul
VOTING_WINDOW = 8          # DIPERKECIL → tidak mengunci huruf lama
STABLE_FRAMES = 8          # Pose harus stabil beberapa frame

# =========================
# LOAD MODEL & TOOLS
# =========================
print("📦 Load model & tools...")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("✅ Model, scaler, label encoder siap")

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# BUFFER
# =========================
sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=VOTING_WINDOW)
stable_count = 0

# =========================
# KAMERA
# =========================
cap = cv2.VideoCapture(0)

print("🎥 Kamera aktif. Tahan pose ±1 detik. Tekan 'q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_letter = "-"
    confidence = 0.0

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # =========================
        # AMBIL LANDMARK
        # =========================
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == NUM_FEATURES:

            # =========================
            # NORMALISASI RELATIF WRIST (PENTING!)
            # =========================
            base_x, base_y, base_z = landmarks[0], landmarks[1], landmarks[2]
            for i in range(0, len(landmarks), 3):
                landmarks[i]   -= base_x
                landmarks[i+1] -= base_y
                landmarks[i+2] -= base_z

            # =========================
            # SCALING (SAMA DENGAN TRAINING)
            # =========================
            landmarks = scaler.transform([landmarks])[0]
            sequence.append(landmarks)

    # =========================
    # PREDIKSI
    # =========================
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        preds = model.predict(input_data, verbose=0)[0]

        conf = np.max(preds)
        pred_class = np.argmax(preds)

        # =========================
        # KONTROL CONFIDENCE
        # =========================
        if conf >= CONF_THRESHOLD:
            stable_count += 1
        else:
            stable_count = 0
            predictions.clear()   # RESET agar tidak terkunci R/L

        # =========================
        # POSE HARUS STABIL
        # =========================
        if stable_count >= STABLE_FRAMES:
            letter = label_encoder.inverse_transform([pred_class])[0]
            predictions.append(letter)

            # Voting
            most_common = Counter(predictions).most_common(1)[0]
            current_letter = most_common[0]
            confidence = conf

    # =========================
    # TAMPILAN
    # =========================
    cv2.rectangle(frame, (0, 0), (360, 95), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Huruf : {current_letter}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence:.2f}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Realtime Detection SIBI (FIX)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEAN UP
# =========================
cap.release()
cv2.destroyAllWindows()
hands.close()
print(" Realtime detection dihentikan")
