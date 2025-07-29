#!/usr/bin/env python3
"""
Startup script for the Face Verification Flask Backend
"""

import os
import sys
import logging
from app import app, load_models
from config import Config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )

def check_models():
    """Check if model files exist"""
    models = [
        Config.AGE_GENDER_MODEL_PATH,
        Config.EMOTION_MODEL_PATH
    ]
    
    missing_models = []
    for model_path in models:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print(f"Error: Missing model files: {missing_models}")
        print("Please ensure your .h5 model files are in the 'asset' directory")
        sys.exit(1)
    
    print("✓ All model files found")

def main():
    """Main function to start the Flask application"""
    print("Starting Face Verification Flask Backend...")
    
    # Setup logging
    setup_logging()
    
    # Check if model files exist
    check_models()
    
    # Load models
    try:
        print("Loading models...")
        load_models()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
    
    # Start Flask app
    print(f"Starting Flask server on {Config.HOST}:{Config.PORT}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )

if __name__ == '__main__':
    main()
