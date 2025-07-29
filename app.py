import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for models
age_gender_model = None
emotion_model = None

# Model paths
AGE_GENDER_MODEL_PATH = os.path.join('asset', 'age_gender_model.h5')
EMOTION_MODEL_PATH = os.path.join('asset', 'emotion_model.h5')

# Age and gender labels (adjust based on your model's output)
AGE_RANGES = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
GENDER_LABELS = ['Male', 'Female']

# Emotion labels (adjust based on your model's output)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_models():
    """Load the trained models on startup"""
    global age_gender_model, emotion_model
    
    try:
        logger.info("Loading age-gender model...")
        age_gender_model = load_model(AGE_GENDER_MODEL_PATH)
        logger.info("Age-gender model loaded successfully")
        
        logger.info("Loading emotion model...")
        emotion_model = load_model(EMOTION_MODEL_PATH)
        logger.info("Emotion model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

def preprocess_image_for_age_gender(image, target_size=(224, 224)):
    """Preprocess image for age-gender prediction"""
    try:
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image for age-gender: {str(e)}")
        raise e

def preprocess_image_for_emotion(image, target_size=(48, 48)):
    """Preprocess image for emotion prediction"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image for emotion: {str(e)}")
        raise e

def decode_base64_image(base64_string):
    """Decode base64 image string to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'age_gender': age_gender_model is not None,
            'emotion': emotion_model is not None
        }
    })

@app.route('/predict/age-gender', methods=['POST'])
def predict_age_gender():
    """Predict age and gender from face image"""
    try:
        # Check if models are loaded
        if age_gender_model is None:
            return jsonify({'error': 'Age-gender model not loaded'}), 500
        
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode and preprocess image
        image = decode_base64_image(data['image'])
        processed_image = preprocess_image_for_age_gender(image)
        
        # Make prediction
        predictions = age_gender_model.predict(processed_image)
        
        # Assuming the model outputs [age_probabilities, gender_probabilities]
        if len(predictions) == 2:
            age_probs, gender_probs = predictions
            
            # Get predicted age range
            age_idx = np.argmax(age_probs[0])
            predicted_age = AGE_RANGES[age_idx]
            age_confidence = float(age_probs[0][age_idx])
            
            # Get predicted gender
            gender_idx = np.argmax(gender_probs[0])
            predicted_gender = GENDER_LABELS[gender_idx]
            gender_confidence = float(gender_probs[0][gender_idx])
            
        else:
            # Handle single output case (adjust based on your model structure)
            prediction = predictions[0]
            # You may need to adjust this based on your specific model output
            predicted_age = "Unknown"
            predicted_gender = "Unknown"
            age_confidence = 0.0
            gender_confidence = 0.0
        
        return jsonify({
            'success': True,
            'predictions': {
                'age': {
                    'range': predicted_age,
                    'confidence': age_confidence
                },
                'gender': {
                    'label': predicted_gender,
                    'confidence': gender_confidence
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in age-gender prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/emotion', methods=['POST'])
def predict_emotion():
    """Predict emotion from face image"""
    try:
        # Check if model is loaded
        if emotion_model is None:
            return jsonify({'error': 'Emotion model not loaded'}), 500
        
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode and preprocess image
        image = decode_base64_image(data['image'])
        processed_image = preprocess_image_for_emotion(image)
        
        # Make prediction
        predictions = emotion_model.predict(processed_image)
        
        # Get emotion probabilities
        emotion_probs = predictions[0]
        
        # Get predicted emotion
        emotion_idx = np.argmax(emotion_probs)
        predicted_emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(emotion_probs[emotion_idx])
        
        # Get all emotion probabilities
        all_emotions = {}
        for i, emotion in enumerate(EMOTION_LABELS):
            all_emotions[emotion] = float(emotion_probs[i])
        
        return jsonify({
            'success': True,
            'predictions': {
                'emotion': {
                    'label': predicted_emotion,
                    'confidence': confidence
                },
                'all_emotions': all_emotions
            }
        })
        
    except Exception as e:
        logger.error(f"Error in emotion prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/combined', methods=['POST'])
def predict_combined():
    """Predict both age-gender and emotion from face image"""
    try:
        # Check if models are loaded
        if age_gender_model is None or emotion_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image once
        image = decode_base64_image(data['image'])
        
        # Preprocess for both models
        age_gender_image = preprocess_image_for_age_gender(image)
        emotion_image = preprocess_image_for_emotion(image)
        
        # Make predictions
        age_gender_predictions = age_gender_model.predict(age_gender_image)
        emotion_predictions = emotion_model.predict(emotion_image)
        
        # Process age-gender predictions
        if len(age_gender_predictions) == 2:
            age_probs, gender_probs = age_gender_predictions
            
            age_idx = np.argmax(age_probs[0])
            predicted_age = AGE_RANGES[age_idx]
            age_confidence = float(age_probs[0][age_idx])
            
            gender_idx = np.argmax(gender_probs[0])
            predicted_gender = GENDER_LABELS[gender_idx]
            gender_confidence = float(gender_probs[0][gender_idx])
        else:
            predicted_age = "Unknown"
            predicted_gender = "Unknown"
            age_confidence = 0.0
            gender_confidence = 0.0
        
        # Process emotion predictions
        emotion_probs = emotion_predictions[0]
        emotion_idx = np.argmax(emotion_probs)
        predicted_emotion = EMOTION_LABELS[emotion_idx]
        emotion_confidence = float(emotion_probs[emotion_idx])
        
        # Get all emotion probabilities
        all_emotions = {}
        for i, emotion in enumerate(EMOTION_LABELS):
            all_emotions[emotion] = float(emotion_probs[i])
        
        return jsonify({
            'success': True,
            'predictions': {
                'age': {
                    'range': predicted_age,
                    'confidence': age_confidence
                },
                'gender': {
                    'label': predicted_gender,
                    'confidence': gender_confidence
                },
                'emotion': {
                    'label': predicted_emotion,
                    'confidence': emotion_confidence
                },
                'all_emotions': all_emotions
            }
        })
        
    except Exception as e:
        logger.error(f"Error in combined prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
