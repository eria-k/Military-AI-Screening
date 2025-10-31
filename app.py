import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, render_template, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for loaded components
model = None
scaler = None
label_encoder = None
knowledge_graph = None

def ensure_model_exists():
    """Ensure model file exists"""
    if not os.path.exists("military_screening_cnn.h5"):
        logger.info("🔄 Model file not found, handling...")
        try:
            # Try to extract from 7z
            import py7zr
            if os.path.exists("military_screening_cnn.7z"):
                with py7zr.SevenZipFile('military_screening_cnn.7z', mode='r') as z:
                    z.extractall()
                logger.info("✅ Model extracted from 7z!")
                return True
            else:
                logger.warning("⚠️ 7z file not found, creating demo model...")
                return create_demo_model()
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return create_demo_model()
    return True

def create_demo_model():
    """Create a simple demo model"""
    try:
        logger.info("🔄 Creating demo model...")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
        
        demo_model = Sequential([
            Conv1D(8, 3, activation='relu', input_shape=(561, 1)),
            MaxPooling1D(2),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(6, activation='softmax')
        ])
        
        demo_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        demo_model.save('military_screening_cnn.h5')
        logger.info("✅ Demo model created!")
        return True
    except Exception as e:
        logger.error(f"❌ Demo model failed: {e}")
        return False

def load_components():
    """Load AI components"""
    global model, scaler, label_encoder, knowledge_graph
    
    try:
        logger.info("🔄 Loading AI components...")
        
        if not ensure_model_exists():
            logger.error("❌ Could not ensure model exists")
            return False
        
        model = tf.keras.models.load_model("military_screening_cnn.h5")
        logger.info("✅ Model loaded")
        
        scaler = joblib.load("scaler.pkl")
        logger.info("✅ Scaler loaded")
        
        label_encoder = joblib.load("label_encoder.pkl")
        logger.info("✅ Label encoder loaded")
        
        knowledge_graph = joblib.load("military_knowledge_graph.pkl")
        logger.info("✅ Knowledge graph loaded")
        
        logger.info("🎯 All components loaded!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Loading failed: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    components_loaded = all([model, scaler, label_encoder, knowledge_graph])
    return jsonify({
        'status': 'healthy' if components_loaded else 'loading',
        'components_loaded': components_loaded,
        'message': 'Military AI Screening System - READY'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Simplified prediction endpoint"""
    try:
        if not all([model, scaler, label_encoder, knowledge_graph]):
            return jsonify({'success': False, 'error': 'System initializing...'})
            
        data = request.json
        if 'sensor_data' not in data:
            return jsonify({'success': False, 'error': 'No sensor data'})
            
        # Convert to numpy array with proper dtype for NumPy 2.0
        sensor_data = np.array(data['sensor_data'], dtype=np.float64).reshape(1, -1)
        
        # Preprocess
        scaled_data = scaler.transform(sensor_data)
        reshaped_data = scaled_data.reshape(1, 561, 1)
        
        # Predict
        predictions = model.predict(reshaped_data, verbose=0)
        confidence = float(np.max(predictions))
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        activity = label_encoder.inverse_transform([predicted_class])[0]
        
        # Simple decision logic
        if confidence > 0.8:
            decision = "PASS"
            roles = ["Infantry", "Special Forces", "Combat Engineer"]
            reason = "Excellent performance"
        elif confidence > 0.6:
            decision = "CONDITIONAL PASS" 
            roles = ["Logistics", "Signals", "Administration"]
            reason = "Adequate performance"
        else:
            decision = "FAIL"
            roles = ["Medical Evaluation Required"]
            reason = "Needs improvement"
        
        return jsonify({
            'success': True,
            'prediction': {
                'activity': activity,
                'confidence': confidence,
                'decision': decision,
                'reason': reason,
                'recommended_roles': roles,
                'performance_score': round(confidence * 100, 1)
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Processing error'})

@app.route('/demo')
def demo_info():
    """Demo information"""
    return jsonify({
        'system': 'Military AI Screening',
        'status': 'operational',
        'demo_candidates': ['excellent', 'average', 'poor']
    })

# Initialize components
logger.info("🚀 Military AI Screening System Starting...")
load_components()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

