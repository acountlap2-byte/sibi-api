from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import time
from collections import deque
from threading import Lock
import uuid
import logging

# ================= TF OPTIMIZATION =================
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ================= KONFIGURASI =================
MODEL_PATH = "model/model_sibi_landmark_FINAL.keras"
SCALER_PATH = "preprocessed/scaler.pkl"

BASE_CONF_THRESHOLD = 0.45
STABLE_TIME = 0.6
VOTE_SIZE = 5
SESSION_TIMEOUT = 3.0
API_MIN_INTERVAL = 0.5

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# ================= APP =================
app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

predict_lock = Lock()
sessions_lock = Lock()
sessions = {}

def get_session(sid):
    with sessions_lock:
        if sid not in sessions:
            sessions[sid] = {
                "last_label": None,
                "stable_start": None,
                "vote_buffer": deque(maxlen=VOTE_SIZE),
                "last_active": time.time(),
                "last_predict": 0.0
            }
        sessions[sid]["last_active"] = time.time()
        return sessions[sid]

def cleanup_sessions():
    now = time.time()
    with sessions_lock:
        expired = [
            sid for sid, s in sessions.items()
            if now - s["last_active"] > SESSION_TIMEOUT
        ]
        for sid in expired:
            del sessions[sid]

def preprocess_landmark(lm):
    lm = np.array(lm, dtype=np.float32)
    lm = lm - lm[0]

    if lm[5][0] < lm[17][0]:
        lm[:, 0] = -lm[:, 0]

    ref = np.linalg.norm(lm[9] - lm[0])
    if ref > 0:
        lm = lm / ref

    return lm.flatten()

@app.route("/predict", methods=["POST"])
def predict():
    cleanup_sessions()
    data = request.get_json(silent=True)

    if not data or "landmark" not in data:
        return jsonify({"huruf": "-", "status": "INVALID"})

    sid = data.get("session_id") or str(uuid.uuid4())
    session = get_session(sid)

    now = time.time()
    if now - session["last_predict"] < API_MIN_INTERVAL:
        return jsonify({
            "huruf": session["last_label"]or "-", 
            "confidence": 0.0,
            "status": "SKIPPED"})
    session["last_predict"] = now

    lm = data["landmark"]
    if len(lm) != 21:
        return jsonify({"huruf": "-", "status": "INVALID"})

    feat = preprocess_landmark(lm)
    feat = scaler.transform(feat.reshape(1, -1))

    with predict_lock:
        preds = model.predict(feat, verbose=0)[0]

    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    label = LABELS[idx]

    threshold = BASE_CONF_THRESHOLD
    if label == "U": threshold = 0.33
    elif label == "Y": threshold = 0.35

    if label == "Z" or conf < threshold:
        session["vote_buffer"].clear()
        session["stable_start"] = None
        session["last_label"] = None
        return jsonify({"huruf": "-", "status": "LOW_CONF"})

    session["vote_buffer"].append(label)

    if session["vote_buffer"].count(label) < 3:
        return jsonify({"huruf": label, "status": "SEARCHING"})

    if label == session["last_label"]:
        if session["stable_start"] is None:
            session["stable_start"] = now
        elif now - session["stable_start"] >= STABLE_TIME:
            return jsonify({"huruf": label, "status": "FINAL"})
    else:
        session["last_label"] = label
        session["stable_start"] = now

    return jsonify({"huruf": label, "status": "SEARCHING"})

# ===== WARM UP =====
_dummy = scaler.transform(np.zeros((1,63)))
model.predict(_dummy, verbose=0)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
