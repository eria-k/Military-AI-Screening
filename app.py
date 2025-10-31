import os
import json
import warnings
from typing import Any, Dict, List

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import tensorflow as tf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(APP_DIR, "military_screening_cnn.h5"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(APP_DIR, "scaler.pkl"))
LE_PATH = os.getenv("LABEL_ENCODER_PATH", os.path.join(APP_DIR, "label_encoder.pkl"))
KG_PATH = os.getenv("KG_PATH", os.path.join(APP_DIR, "military_knowledge_graph.pkl"))
EXPECTED_FEATURES = int(os.getenv("EXPECTED_FEATURES", "561"))

app = Flask(__name__, template_folder="templates")
CORS(app)

def ensure_model_exists() -> bool:
    if os.path.exists(MODEL_PATH):
        return True
    archive_path = os.path.join(APP_DIR, "military_screening_cnn.7z")
    if os.path.exists(archive_path):
        try:
            import py7zr
            logger.info("🔄 Extracting model from 7z...")
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                z.extractall(APP_DIR)
            logger.info("✅ Model extracted")
            return os.path.exists(MODEL_PATH)
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return False
    logger.error("❌ Model missing and no .7z archive found")
    return False

def load_pickle(path: str, fallback=None):
    try:
        obj = joblib.load(path)
        logger.info(f"✅ Loaded {os.path.basename(path)}")
        return obj
    except Exception as e:
        logger.warning(f"⚠️ Could not load {os.path.basename(path)}: {e}")
        return fallback

def build_default_kg() -> Dict[str, Any]:
    return {
        "role_rules": {
            "LOW": ["Infantry", "Special Forces", "Combat Engineer"],
            "MODERATE": ["Military Police", "Logistics", "Signals", "Administration"],
            "HIGH": ["Medical Evaluation Required"]
        }
    }

logger.info("🚀 Military AI Screening System Starting...")
if not ensure_model_exists():
    logger.error("❌ Model not available at startup")

logger.info("🔄 Loading TensorFlow model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
try:
    model.compile()
except Exception:
    pass
logger.info("✅ TensorFlow model loaded")

scaler = load_pickle(SCALER_PATH)
label_encoder = load_pickle(LE_PATH)
knowledge_graph = load_pickle(KG_PATH, fallback=build_default_kg())
if knowledge_graph is None or not isinstance(knowledge_graph, dict):
    knowledge_graph = build_default_kg()

all_components_loaded = model is not None and scaler is not None and label_encoder is not None

def compute_biomarkers(p: float) -> Dict[str, float]:
    return {
        "movement_quality": float(p),
        "fatigue_index": float(0.05 if p > 0.8 else 0.15 if p > 0.6 else 0.25),
        "movement_smoothness": float(p * 0.9 + 0.1)
    }

def risk_from_biomarkers(bio: Dict[str, float]) -> str:
    mq = bio.get("movement_quality", 0.0)
    if mq > 0.8: return "LOW"
    if mq > 0.6: return "MODERATE"
    return "HIGH"

def decide(p: float) -> (str, str, str):
    if p > 0.8:  return "PASS", "Excellent movement quality and physical performance", "LOW"
    if p > 0.6:  return "CONDITIONAL PASS", "Adequate performance with some areas for improvement", "MODERATE"
    return "FAIL", "Movement analysis indicates physical limitations", "HIGH"

@app.get("/health")
def health():
    return jsonify({
        "status": "healthy" if all_components_loaded else "initializing",
        "system_ready": all_components_loaded,
        "components": {
            "model": bool(model is not None),
            "scaler": bool(scaler is not None),
            "label_encoder": bool(label_encoder is not None),
            "knowledge_graph": bool(knowledge_graph is not None),
        }
    })

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if not all_components_loaded:
        return jsonify({"success": False, "error": "System is initializing. Please retry in a moment."}), 503

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
        X_cnn = X.reshape((X.shape[0], X.shape[1], 1))

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
        decision, reason, risk_level = decide(proba)
        roles = knowledge_graph.get("role_rules", {}).get(risk_level, ["General Service"])

        return jsonify({
            "success": True,
            "prediction": {
                "activity": activity,
                "confidence": round(proba, 4),
                "decision": decision,
                "reason": reason,
                "risk_level": risk_level,
                "recommended_roles": roles,
                "detected_risks": [],
                "performance_score": round(proba * 100, 1)
            }
        })
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"success": False, "error": f"Server error during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)

