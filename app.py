import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, render_template, request, jsonify
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for loaded components
model = None
scaler = None
label_encoder = None
knowledge_graph = None

def extract_model_if_needed():
    """Extract model from 7z if .h5 file doesn't exist"""
    if not os.path.exists("military_screening_cnn.h5"):
        logger.info("🔄 Extracting model from 7z archive...")
        try:
            # Extract using 7z (available on Render)
            result = subprocess.run([
                '7z', 'x', 'military_screening_cnn.7z', '-y'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Model extracted successfully!")
                return True
            else:
                logger.error(f"❌ Extraction failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return False
    return True

def load_components():
    """Load AI components at startup"""
    global model, scaler, label_encoder, knowledge_graph
    
    # First, extract model if needed
    if not extract_model_if_needed():
        logger.error("❌ Failed to extract model")
        return False
        
    try:
        logger.info("🔄 Loading AI components...")
        
        # Load model with compatibility settings for TF 2.20
        model = tf.keras.models.load_model(
            "military_screening_cnn.h5",
            compile=True
        )
        logger.info("✅ Model loaded")
        
        scaler = joblib.load("scaler.pkl")
        logger.info("✅ Scaler loaded")
        
        label_encoder = joblib.load("label_encoder.pkl") 
        logger.info("✅ Label encoder loaded")
        
        knowledge_graph = joblib.load("military_knowledge_graph.pkl")
        logger.info("✅ Knowledge graph loaded")
        
        logger.info("🎯 All AI components loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading components: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    components_loaded = all([model, scaler, label_encoder, knowledge_graph])
    status = 'healthy' if components_loaded else 'loading'
    
    return jsonify({
        'status': status,
        'components_loaded': components_loaded,
        'message': 'Military AI Screening System',
        'tensorflow_version': tf.__version__
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        if not all([model, scaler, label_encoder, knowledge_graph]):
            return jsonify({
                'success': False, 
                'error': 'AI components still loading. Please refresh and try again.'
            })
            
        # Get data from request
        data = request.json
        if 'sensor_data' not in data:
            return jsonify({'success': False, 'error': 'No sensor_data provided'})
            
        sensor_data = np.array(data['sensor_data']).reshape(1, -1)
        
        if sensor_data.shape[1] != 561:
            return jsonify({
                'success': False, 
                'error': f'Expected 561 features, got {sensor_data.shape[1]}'
            })
        
        # Preprocess
        scaled_data = scaler.transform(sensor_data)
        reshaped_data = scaled_data.reshape(1, 561, 1)
        
        # Predict
        predictions = model.predict(reshaped_data, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        activity = label_encoder.inverse_transform([predicted_class])[0]
        
        # Extract biomarkers
        biomarkers = {
            'movement_quality': float(confidence),
            'fatigue_index': 0.05 if confidence > 0.8 else 0.15,
            'movement_smoothness': float(confidence * 0.9 + 0.1)
        }
        
        # Make military decision
        if confidence > 0.8:
            decision = "PASS"
            reason = "Excellent movement quality and form"
            roles = ["Infantry", "Special Forces", "Combat Engineer"]
            risk_level = "LOW"
        elif confidence > 0.6:
            decision = "CONDITIONAL PASS"
            reason = "Adequate performance - requires monitoring"
            roles = ["Military Police", "Logistics", "Signals", "Administration"]
            risk_level = "MODERATE"
        else:
            decision = "FAIL"
            reason = "Poor movement quality detected - medical evaluation required"
            roles = ["Needs medical assessment"]
            risk_level = "HIGH"
        
        return jsonify({
            'success': True,
            'prediction': {
                'activity': activity,
                'confidence': float(confidence),
                'decision': decision,
                'reason': reason,
                'risk_level': risk_level,
                'recommended_roles': roles,
                'biomarkers': biomarkers
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

@app.route('/demo-candidates')
def get_demo_candidates():
    """Return demo candidate data"""
    return jsonify({
        'candidates': {
            'excellent': {
                'description': 'Excellent performer - expected PASS',
                'expected_result': 'PASS with combat roles'
            },
            'average': {
                'description': 'Average performer - expected CONDITIONAL PASS', 
                'expected_result': 'CONDITIONAL PASS with support roles'
            },
            'poor': {
                'description': 'Poor performer - expected FAIL',
                'expected_result': 'FAIL - medical evaluation'
            }
        }
    })

# Load components when app starts
logger.info("🚀 Starting Military AI Screening System...")
logger.info(f"📊 TensorFlow version: {tf.__version__}")
load_components()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
