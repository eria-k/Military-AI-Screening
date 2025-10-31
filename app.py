import os, warnings, logging
from typing import Any, Dict, List
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib

# Quieter sklearn warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Keep TF CPU modest on small instances
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("app")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(APP_DIR, "military_screening_cnn.h5"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(APP_DIR, "scaler.pkl"))
LE_PATH     = os.getenv("LABEL_ENCODER_PATH", os.path.join(APP_DIR, "label_encoder.pkl"))
KG_PATH     = os.getenv("KG_PATH", os.path.join(APP_DIR, "military_knowledge_graph.pkl"))
EXPECTED_FEATURES = int(os.getenv("EXPECTED_FEATURES", "561"))

app = Flask(__name__, template_folder="templates")
CORS(app)

def load_pickle(path, fallback=None):
    try:
        obj = joblib.load(path)
        log.info(f"✅ Loaded {os.path.basename(path)}")
        return obj
    except Exception as e:
        log.warning(f"⚠️ Could not load {os.path.basename(path)}: {e}")
        return fallback

def build_default_kg() -> Dict[str, Any]:
    return {
        "role_rules": {
            "LOW": ["Infantry","Special Forces","Combat Engineer"],
            "MODERATE": ["Military Police","Logistics","Signals","Administration"],
            "HIGH": ["Medical Evaluation Required"]
        }
    }

# ---------- Startup ----------
log.info("🚀 Military AI Screening System Starting...")
if not os.path.exists(MODEL_PATH):
    # If you keep a 7z in the container, auto-extract here (optional)
    archive = os.path.join(APP_DIR, "military_screening_cnn.7z")
    if os.path.exists(archive):
        try:
            import py7zr
            log.info("🔄 Extracting model from 7z...")
            with py7zr.SevenZipFile(archive, mode="r") as z:
                z.extractall(APP_DIR)
            log.info("✅ Model extracted")
        except Exception as e:
            log.error(f"❌ 7z extraction failed: {e}")

log.info("🔄 Loading TensorFlow model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
try:
    model.compile()
except Exception:
    pass
log.info("✅ TensorFlow model loaded")

scaler = load_pickle(SCALER_PATH)
label_encoder = load_pickle(LE_PATH)
knowledge_graph = load_pickle(KG_PATH, fallback=build_default_kg()) or build_default_kg()

# Warmup to avoid slow first request → proxy 502
try:
    dummy = np.zeros((1, EXPECTED_FEATURES), dtype=np.float32)
    if scaler is not None:
        dummy = scaler.transform(dummy)
    dummy = dummy.reshape((1, EXPECTED_FEATURES, 1))
    _ = model.predict(dummy, verbose=0)
    log.info("🔥 Warmup inference complete")
except Exception as e:
    log.warning(f"Warmup skipped: {e}")

SYSTEM_READY = all([model is not None, scaler is not None, label_encoder is not None])

# ---------- Helpers ----------
def compute_biomarkers(p: float) -> Dict[str,float]:
    return {
        "movement_quality": float(p),
        "fatigue_index": float(0.05 if p > 0.8 else 0.15 if p > 0.6 else 0.25),
        "movement_smoothness": float(p*0.9 + 0.1)
    }

def decide(p: float):
    if p > 0.8:  return "PASS", "Excellent movement quality", "LOW"
    if p > 0.6:  return "CONDITIONAL PASS", "Adequate, some improvement needed", "MODERATE"
    return "FAIL", "Indicators suggest limitations", "HIGH"

# ---------- Routes ----------
@app.get("/health")
def health():
    return jsonify({
        "ok": SYSTEM_READY,
        "model": os.path.basename(MODEL_PATH),
        "scaler": os.path.basename(SCALER_PATH) if scaler is not None else None,
        "label_encoder": os.path.basename(LE_PATH) if label_encoder is not None else None,
        "kg": "loaded" if knowledge_graph else "default"
    })

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if not SYSTEM_READY:
        return jsonify({"success": False, "error": "System initializing; retry shortly."}), 503

    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400

    if not payload or "sensor_data" not in payload:
        return jsonify({"success": False, "error": "Missing 'sensor_data'"}), 400

    data = payload["sensor_data"]
    if not isinstance(data, list):
        return jsonify({"success": False, "error": "sensor_data must be a list"}), 400
    if len(data) != EXPECTED_FEATURES:
        return jsonify({"success": False, "error": f"Expected {EXPECTED_FEATURES} features, got {len(data)}"}), 400

    try:
        arr = np.array(data, dtype=np.float32).reshape(1, -1)
        if np.isnan(arr).any() or np.isinf(arr).any():
            return jsonify({"success": False, "error": "sensor_data contains NaN/Inf"}), 400

        X = scaler.transform(arr) if scaler is not None else arr
        X_cnn = X.reshape((1, EXPECTED_FEATURES, 1))

        probs = model.predict(X_cnn, verbose=0)
        if probs.ndim == 2:
            proba = float(np.max(probs[0]))
            pred_idx = int(np.argmax(probs[0]))
        else:
            proba = float(probs.squeeze())
            pred_idx = int(proba >= 0.5)

        try:
            activity = str(label_encoder.inverse_transform([pred_idx])[0])
        except Exception:
            activity = str(pred_idx)

        biomarkers = compute_biomarkers(proba)
        decision, reason, risk = decide(proba)
        roles = knowledge_graph.get("role_rules", {}).get(risk, ["General Service"])

        return jsonify({
            "success": True,
            "prediction": {
                "activity": activity,
                "confidence": round(proba, 4),
                "decision": decision,
                "reason": reason,
                "risk_level": risk,
                "recommended_roles": roles,
                "biomarkers": {k: round(v,4) for k,v in biomarkers.items()},
                "performance_score": round(proba*100, 1)
            }
        })
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"success": False, "error": f"Server error during prediction: {e}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)


