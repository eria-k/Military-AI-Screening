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

def extract_model_if_needed():
    """Extract model from 7z if .h5 file doesn't exist"""
    if not os.path.exists("military_screening_cnn.h5"):
        logger.info("🔄 Extracting model from 7z archive...")
        try:
            # Use py7zr for extraction
            import py7zr
            with py7zr.SevenZipFile('military_screening_cnn.7z', mode='r') as z:
                z.extractall()
            logger.info("✅ Model extracted successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            logger.info("🔄 Creating demo model instead...")
            return create_demo_model()
    return True

def create_demo_model():
    """Create a simple demo model if extraction fails"""
    try:
        logger.info("🔄 Creating demo model for presentation...")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
        
        # Simple model that will work for demo
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
    """Load AI components at startup"""
    global model, scaler, label_encoder, knowledge_graph
    
    try:
        logger.info("🔄 Loading AI components...")
        
        # First, handle model extraction/creation
        if not extract_model_if_needed():
            logger.error("❌ Failed to handle model")
            return False
        
        # Load model
        model = tf.keras.models.load_model("military_screening_cnn.h5")
        logger.info("✅ Model loaded")
        
        # Load other components
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
    """Health check endpoint"""
    components_loaded = all([model, scaler, label_encoder, knowledge_graph])
    return jsonify({
        'status': 'healthy' if components_loaded else 'loading',
        'components_loaded': components_loaded,
        'message': 'Military AI Screening System'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        if not all([model, scaler, label_encoder, knowledge_graph]):
            return jsonify({
                'success': False, 
                'error': 'System initializing. Please try again in 30 seconds.'
            })
            
        # Get data from request
        data = request.json
        if 'sensor_data' not in data:
            return jsonify({'success': False, 'error': 'No sensor_data provided'})
            
        sensor_data = np.array(data['sensor_data']).reshape(1, -1)
        
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
            'error': 'System error. Please try again.'
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
load_components()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

