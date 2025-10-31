import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, render_template, request, jsonify
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Add this line - enables CORS for all route

# Global variables for loaded components
model = None
scaler = None
label_encoder = None
knowledge_graph = None
all_components_loaded = False

def create_default_knowledge_graph():
    """Create a default knowledge graph if loading fails"""
    logger.info("🔄 Creating default knowledge graph...")
    
    class DefaultKnowledgeGraph:
        def __init__(self):
            self.rules = {
                'high_confidence': ['Infantry', 'Special Forces', 'Combat Engineer'],
                'medium_confidence': ['Military Police', 'Logistics', 'Signals'],
                'low_confidence': ['Medical Evaluation Required']
            }
        
        def get_recommendations(self, confidence):
            if confidence > 0.8:
                return self.rules['high_confidence']
            elif confidence > 0.6:
                return self.rules['medium_confidence']
            else:
                return self.rules['low_confidence']
    
    return DefaultKnowledgeGraph()

def ensure_model_exists():
    """Ensure model file exists and extract if needed"""
    if not os.path.exists("military_screening_cnn.h5"):
        logger.info("🔄 Model file not found, extracting from 7z...")
        try:
            import py7zr
            if os.path.exists("military_screening_cnn.7z"):
                with py7zr.SevenZipFile('military_screening_cnn.7z', mode='r') as z:
                    z.extractall()
                logger.info("✅ Model extracted from 7z successfully!")
                return True
            else:
                logger.error("❌ 7z file not found!")
                return False
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return False
    return True

def load_all_components():
    """Load all AI components with proper error handling"""
    global model, scaler, label_encoder, knowledge_graph, all_components_loaded
    
    try:
        logger.info("🚀 STARTING COMPONENT LOADING PROCESS...")
        
        # Step 1: Ensure model exists
        if not ensure_model_exists():
            logger.error("❌ Failed to ensure model exists")
            return False
        
        # Step 2: Load TensorFlow model
        logger.info("🔄 Loading TensorFlow model...")
        model = tf.keras.models.load_model("military_screening_cnn.h5")
        logger.info("✅ TensorFlow model loaded")
        
        # Step 3: Load scaler
        logger.info("🔄 Loading scaler...")
        scaler = joblib.load("scaler.pkl")
        logger.info("✅ Scaler loaded")
        
        # Step 4: Load label encoder
        logger.info("🔄 Loading label encoder...")
        label_encoder = joblib.load("label_encoder.pkl")
        logger.info("✅ Label encoder loaded")
        
        # Step 5: Try to load knowledge graph, create default if fails
        logger.info("🔄 Loading knowledge graph...")
        try:
            knowledge_graph = joblib.load("military_knowledge_graph.pkl")
            logger.info("✅ Knowledge graph loaded from file")
        except Exception as e:
            logger.warning(f"⚠️ Knowledge graph loading failed: {e}")
            logger.info("🔄 Creating default knowledge graph...")
            knowledge_graph = create_default_knowledge_graph()
            logger.info("✅ Default knowledge graph created")
        
        # Verify critical components are loaded
        critical_components_loaded = all([model, scaler, label_encoder])
        if critical_components_loaded:
            all_components_loaded = True
            logger.info("🎯 CRITICAL COMPONENTS LOADED - SYSTEM READY!")
            logger.info("💡 Knowledge graph: " + ("Loaded" if hasattr(knowledge_graph, 'rules') else "Using default"))
            return True
        else:
            logger.error("❌ Critical components failed to load")
            all_components_loaded = False
            return False
            
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR loading components: {e}")
        all_components_loaded = False
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Detailed health check endpoint"""
    global all_components_loaded, knowledge_graph
    
    component_status = {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'label_encoder_loaded': label_encoder is not None,
        'knowledge_graph_loaded': knowledge_graph is not None,
        'knowledge_graph_type': 'file' if hasattr(knowledge_graph, 'rules') else 'default' if knowledge_graph else 'none',
        'all_components_ready': all_components_loaded
    }
    
    status = 'healthy' if all_components_loaded else 'initializing'
    
    return jsonify({
        'status': status,
        'components': component_status,
        'message': 'Military AI Screening System',
        'system_ready': all_components_loaded
    })
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Prediction endpoint - FIXED VERSION"""
    global all_components_loaded, knowledge_graph
    
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Check if components are loaded
        if not all_components_loaded:
            return jsonify({
                'success': False, 
                'error': 'System is still initializing. Please wait a moment and try again.'
            })
        
        # Get and validate request data
        data = request.get_json()
        if not data or 'sensor_data' not in data:
            return jsonify({'success': False, 'error': 'No sensor_data provided'})
            
        sensor_data = data['sensor_data']
        if not isinstance(sensor_data, list) or len(sensor_data) != 561:
            return jsonify({
                'success': False, 
                'error': f'Expected 561 features as list, got {type(sensor_data)} with length {len(sensor_data) if isinstance(sensor_data, list) else "N/A"}'
            })
        
        # Convert to numpy array with error handling
        try:
            sensor_array = np.array(sensor_data, dtype=np.float64).reshape(1, -1)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid sensor data format: {str(e)}'})
        
        # Preprocess data
        try:
            scaled_data = scaler.transform(sensor_array)
            reshaped_data = scaled_data.reshape(1, 561, 1)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Data preprocessing failed: {str(e)}'})
        
        # Make prediction
        try:
            predictions = model.predict(reshaped_data, verbose=0)
            confidence = float(np.max(predictions))
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            activity = label_encoder.inverse_transform([predicted_class])[0]
        except Exception as e:
            return jsonify({'success': False, 'error': f'Model prediction failed: {str(e)}'})
        
        logger.info(f"✅ Prediction successful: {activity} with confidence {confidence:.3f}")
        
        # Extract biomarkers for knowledge graph
        biomarkers = {
            'movement_quality': confidence,
            'fatigue_index': 0.05 if confidence > 0.8 else 0.15,
            'movement_smoothness': confidence * 0.9 + 0.1
        }
        
        # Use knowledge graph for recommendations
        try:
            if hasattr(knowledge_graph, 'recommend_roles'):
                kg_result = knowledge_graph.recommend_roles(biomarkers)
                roles = kg_result['recommended_roles']
                detected_risks = kg_result.get('detected_risks', [])
                recommendation_source = "knowledge_graph"
            else:
                # Fallback if knowledge graph doesn't have the method
                if confidence > 0.8:
                    roles = ["Infantry", "Special Forces", "Combat Engineer"]
                elif confidence > 0.6:
                    roles = ["Military Police", "Logistics", "Signals", "Administration"]
                else:
                    roles = ["Medical Evaluation Required"]
                detected_risks = []
                recommendation_source = "fallback"
        except Exception as e:
            logger.warning(f"Knowledge graph failed, using fallback: {e}")
            # Fallback logic
            if confidence > 0.8:
                roles = ["Infantry", "Special Forces", "Combat Engineer"]
            elif confidence > 0.6:
                roles = ["Military Police", "Logistics", "Signals", "Administration"]
            else:
                roles = ["Medical Evaluation Required"]
            detected_risks = []
            recommendation_source = "error_fallback"
        
        # Make decision
        if confidence > 0.8:
            decision = "PASS"
            reason = "Excellent movement quality and physical performance"
            risk_level = "LOW"
        elif confidence > 0.6:
            decision = "CONDITIONAL PASS"
            reason = "Adequate performance with some areas for improvement"
            risk_level = "MODERATE"
        else:
            decision = "FAIL"
            reason = "Movement analysis indicates physical limitations"
            risk_level = "HIGH"
        
        return jsonify({
            'success': True,
            'prediction': {
                'activity': activity,
                'confidence': confidence,
                'decision': decision,
                'reason': reason,
                'risk_level': risk_level,
                'recommended_roles': roles,
                'detected_risks': detected_risks,
                'recommendation_source': recommendation_source,
                'performance_score': round(confidence * 100, 1)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        })    


# Initialize components when app starts
logger.info("🚀 Military AI Screening System Starting...")
load_all_components()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)

