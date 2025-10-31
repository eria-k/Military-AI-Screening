import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, render_template, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for loaded components
model = None
scaler = None
label_encoder = None
knowledge_graph = None
all_components_loaded = False

def load_knowledge_graph():
    """Load knowledge graph with detailed error handling"""
    global knowledge_graph
    
    try:
        logger.info("🔄 Loading knowledge graph...")
        
        # Check if file exists
        if not os.path.exists("military_knowledge_graph.pkl"):
            logger.error("❌ Knowledge graph file not found")
            return False
        
        # Load the file
        kg = joblib.load("military_knowledge_graph.pkl")
        logger.info("✅ Knowledge graph file loaded")
        
        # Verify it has required attributes
        required_attrs = ['graph', 'biomarkers_to_risks', 'recommend_roles']
        missing_attrs = [attr for attr in required_attrs if not hasattr(kg, attr)]
        
        if missing_attrs:
            logger.error(f"❌ Knowledge graph missing attributes: {missing_attrs}")
            return False
        
        # Verify graph structure
        if not hasattr(kg.graph, 'nodes') or not hasattr(kg.graph, 'edges'):
            logger.error("❌ Knowledge graph has invalid graph structure")
            return False
        
        knowledge_graph = kg
        logger.info(f"✅ Knowledge graph loaded successfully!")
        logger.info(f"   - Nodes: {len(knowledge_graph.graph.nodes)}")
        logger.info(f"   - Edges: {len(knowledge_graph.graph.edges)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge graph: {e}")
        return False

def load_all_components():
    """Load all AI components"""
    global model, scaler, label_encoder, knowledge_graph, all_components_loaded
    
    try:
        logger.info("🚀 STARTING COMPONENT LOADING...")
        
        # Load TensorFlow model
        logger.info("🔄 Loading TensorFlow model...")
        model = tf.keras.models.load_model("military_screening_cnn.h5")
        logger.info("✅ TensorFlow model loaded")
        
        # Load scaler
        logger.info("🔄 Loading scaler...")
        scaler = joblib.load("scaler.pkl")
        logger.info("✅ Scaler loaded")
        
        # Load label encoder
        logger.info("🔄 Loading label encoder...")
        label_encoder = joblib.load("label_encoder.pkl")
        logger.info("✅ Label encoder loaded")
        
        # Load knowledge graph
        if not load_knowledge_graph():
            logger.error("❌ Knowledge graph loading failed")
            return False
        
        # All components loaded successfully
        all_components_loaded = True
        logger.info("🎯 ALL COMPONENTS LOADED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component loading failed: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    global all_components_loaded, knowledge_graph
    
    kg_status = "not_loaded"
    if knowledge_graph:
        kg_status = f"loaded_{len(knowledge_graph.graph.nodes)}_nodes"
    
    return jsonify({
        'status': 'healthy' if all_components_loaded else 'initializing',
        'components': {
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'label_encoder_loaded': label_encoder is not None,
            'knowledge_graph_loaded': knowledge_graph is not None,
            'knowledge_graph_status': kg_status,
            'all_components_ready': all_components_loaded
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    global all_components_loaded, knowledge_graph
    
    if not all_components_loaded:
        return jsonify({'success': False, 'error': 'System initializing'})
    
    try:
        data = request.json
        sensor_data = np.array(data['sensor_data'], dtype=np.float64).reshape(1, -1)
        
        # Preprocess and predict
        scaled_data = scaler.transform(sensor_data)
        reshaped_data = scaled_data.reshape(1, 561, 1)
        predictions = model.predict(reshaped_data, verbose=0)
        
        confidence = float(np.max(predictions))
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        activity = label_encoder.inverse_transform([predicted_class])[0]
        
        # Extract biomarkers
        biomarkers = {
            'movement_quality': confidence,
            'fatigue_index': 0.05 if confidence > 0.8 else 0.15,
            'movement_smoothness': confidence * 0.9 + 0.1
        }
        
        # Use knowledge graph for recommendations
        kg_result = knowledge_graph.recommend_roles(biomarkers)
        
        # Make decision
        if confidence > 0.8:
            decision = "PASS"
            reason = "Excellent performance"
        elif confidence > 0.6:
            decision = "CONDITIONAL PASS"
            reason = "Adequate performance"
        else:
            decision = "FAIL"
            reason = "Needs improvement"
        
        return jsonify({
            'success': True,
            'prediction': {
                'activity': activity,
                'confidence': confidence,
                'decision': decision,
                'reason': reason,
                'recommended_roles': kg_result['recommended_roles'],
                'detected_risks': kg_result['detected_risks'],
                'performance_score': round(confidence * 100, 1)
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Processing error'})

# Initialize
logger.info("🚀 Starting Military AI Screening System...")
load_all_components()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

