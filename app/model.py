import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to path
import tensorflow as tf
load_model = tf.keras.models.load_model  # Workaround for Pylance

def load_trained_model():
    # Use absolute path to ensure consistency
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet50_trained.keras'))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Sync from Google Drive: G:/My Drive/DAISY-Project/models/resnet50_trained.keras")
    return load_model(model_path)