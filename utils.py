import numpy as np
import cv2
import base64
import io
from PIL import Image
import logging
from config import Config

logger = logging.getLogger(__name__)

def decode_base64_image(base64_string):
    """
    Decode base64 image string to OpenCV format
    
    Args:
        base64_string (str): Base64 encoded image string
        
    Returns:
        numpy.ndarray: OpenCV image array (BGR format)
    """
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

def preprocess_image_for_age_gender(image, target_size=None):
    """
    Preprocess image for age-gender prediction
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    try:
        if target_size is None:
            target_size = Config.AGE_GENDER_INPUT_SIZE
            
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image for age-gender: {str(e)}")
        raise e

def preprocess_image_for_emotion(image, target_size=None):
    """
    Preprocess image for emotion prediction
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    try:
        if target_size is None:
            target_size = Config.EMOTION_INPUT_SIZE
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image for emotion: {str(e)}")
        raise e

def process_age_gender_predictions(predictions):
    """
    Process age-gender model predictions
    
    Args:
        predictions: Model output predictions
        
    Returns:
        dict: Processed age and gender predictions with confidence scores
    """
    try:
        if len(predictions) == 2:
            age_probs, gender_probs = predictions
            
            # Get predicted age range
            age_idx = np.argmax(age_probs[0])
            predicted_age = Config.AGE_RANGES[age_idx]
            age_confidence = float(age_probs[0][age_idx])
            
            # Get predicted gender
            gender_idx = np.argmax(gender_probs[0])
            predicted_gender = Config.GENDER_LABELS[gender_idx]
            gender_confidence = float(gender_probs[0][gender_idx])
            
            return {
                'age': {
                    'range': predicted_age,
                    'confidence': age_confidence
                },
                'gender': {
                    'label': predicted_gender,
                    'confidence': gender_confidence
                }
            }
        else:
            # Handle single output case - adjust based on your model structure
            logger.warning("Unexpected age-gender model output format")
            return {
                'age': {
                    'range': "Unknown",
                    'confidence': 0.0
                },
                'gender': {
                    'label': "Unknown",
                    'confidence': 0.0
                }
            }
    except Exception as e:
        logger.error(f"Error processing age-gender predictions: {str(e)}")
        raise e

def process_emotion_predictions(predictions):
    """
    Process emotion model predictions
    
    Args:
        predictions: Model output predictions
        
    Returns:
        dict: Processed emotion predictions with confidence scores
    """
    try:
        emotion_probs = predictions[0]
        
        # Get predicted emotion
        emotion_idx = np.argmax(emotion_probs)
        predicted_emotion = Config.EMOTION_LABELS[emotion_idx]
        confidence = float(emotion_probs[emotion_idx])
        
        # Get all emotion probabilities
        all_emotions = {}
        for i, emotion in enumerate(Config.EMOTION_LABELS):
            all_emotions[emotion] = float(emotion_probs[i])
        
        return {
            'emotion': {
                'label': predicted_emotion,
                'confidence': confidence
            },
            'all_emotions': all_emotions
        }
    except Exception as e:
        logger.error(f"Error processing emotion predictions: {str(e)}")
        raise e

def validate_image_input(data):
    """
    Validate image input from request
    
    Args:
        data (dict): Request data
        
    Returns:
        bool: True if valid, raises exception if invalid
    """
    if not data or 'image' not in data:
        raise ValueError('No image provided in request')
    
    if not isinstance(data['image'], str):
        raise ValueError('Image must be a base64 encoded string')
    
    return True
