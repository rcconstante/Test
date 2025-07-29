import os

class Config:
    """Configuration settings for the Flask app"""
    
    # Model paths
    AGE_GENDER_MODEL_PATH = os.path.join('asset', 'age_gender_model.h5')
    EMOTION_MODEL_PATH = os.path.join('asset', 'emotion_model.h5')
    
    # Model labels - adjust these based on your specific model outputs
    AGE_RANGES = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
    GENDER_LABELS = ['Male', 'Female']
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Image preprocessing settings
    AGE_GENDER_INPUT_SIZE = (224, 224)  # Adjust based on your model
    EMOTION_INPUT_SIZE = (48, 48)       # Adjust based on your model
    
    # Flask settings
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # CORS settings
    CORS_ORIGINS = ['*']  # Adjust for production
    
    # Logging
    LOG_LEVEL = 'INFO'
