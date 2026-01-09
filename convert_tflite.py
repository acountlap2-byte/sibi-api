import tensorflow as tf
import os

MODEL_PATH = "model/model_cnn_lstm.keras"
TFLITE_PATH = "model/model_cnn_lstm.tflite"

print("🔄 Memuat model...")
model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# PENTING untuk LSTM
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ Model TFLite CNN+LSTM berhasil dibuat")
