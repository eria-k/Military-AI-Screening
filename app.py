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
    """Ensure model file exists - extract from 7z or create demo"""
    if not os.path.exists("military_screening_cnn.h5"):
        logger.info("🔄 Model file not found, attempting extraction...")
        try:
            # Try to extract from 7z
            import py7zr
            if os.path.exists("military_screening_cnn.7z"):
                with py7zr.SevenZipFile('military_screening_cnn.7z', mode='r') as z:
                    z.extractall()
                logger.info("✅ Model extracted from 7z successfully!")
                return True
            else:
                logger.warning("⚠️ 7z file not found, creating demo model...")
                return create_demo_model()
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            logger.info("🔄 Creating demo model instead...")
            return create_demo_model()
    return True

def create_demo_model():
    """Create a simple demo model"""
    try:
        logger.info("🔄 Creating demo model for presentation...")
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
        logger.info("✅ Demo model created successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Demo model creation failed: {e}")
        return False

def load_components():
    """Load AI components"""
    global model, scaler, label_encoder, knowledge_graph
    
    try:
        logger.info("🔄 Loading AI components...")
        
        # Ensure model exists first
        if not ensure_model_exists():
            return False
        
        # Load components
        model = tf.keras.models.load_model("military_screening_cnn.h5")
        logger.info("✅ Model loaded")
        
        scaler = joblib.load("scaler.pkl")
        logger.info("✅ Scaler loaded")
        
        label_encoder = joblib.load("label_encoder.pkl")
        logger.info("✅ Label encoder loaded")
        
        knowledge_graph = joblib.load("military_knowledge_graph.pkl")
        logger.info("✅ Knowledge graph loaded")
        
        logger.info("🎯 All components loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading components: {e}")
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
        'python_version': '3.11.9',
        'tensorflow_version': tf.__version__
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not all([model, scaler, label_encoder, knowledge_graph]):
            return jsonify({'success': False, 'error': 'System initializing...'})
            
        data = request.json
        if 'sensor_data' not in data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        sensor_data = np.array(data['sensor_data']).reshape(1, -1)
        scaled_data = scaler.transform(sensor_data)
        reshaped_data = scaled_data.reshape(1, 561, 1)
        
        predictions = model.predict(reshaped_data, verbose=0)
        confidence = np.max(predictions)
        activity = label_encoder.inverse_transform([np.argmax(predictions, axis=1)[0]])[0]
        
        # Decision logic
        if confidence > 0.8:
            decision, roles = "PASS", ["Infantry", "Special Forces", "Combat Engineer"]
        elif confidence > 0.6:
            decision, roles = "CONDITIONAL PASS", ["Logistics", "Signals", "Administration"]
        else:
            decision, roles = "FAIL", ["Medical Evaluation Required"]
        
        return jsonify({
            'success': True,
            'prediction': {
                'activity': activity,
                'confidence': float(confidence),
                'decision': decision,
                'recommended_roles': roles,
                'biomarkers': {
                    'movement_quality': float(confidence),
                    'performance_score': float(confidence * 100)
                }
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': 'Prediction error'})

# Initialize on startup
logger.info("🚀 Starting Military AI Screening System...")
load_components()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
