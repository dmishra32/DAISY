import os
import sys
import streamlit as st
import tensorflow as tf
import logging
from pathlib import Path
import time

# Handle TensorFlow imports properly
try:
    from keras.models import load_model
except ImportError:
    from keras.models import load_model


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ModelManager:
    """
    Enhanced model management class with error handling and monitoring
    """
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_info = {}
        self.load_time = None
        
    def get_model_path(self):
        """
        Get the absolute path to the model file
        """
        possible_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet50_trained.keras')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet50_trained.h5')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'resnet50_trained.keras')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'resnet50_trained.h5'))
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def validate_model_file(self, model_path):
        """
        Validate model file before loading
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Check file size (should be reasonable for a trained model)
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # Less than 1MB seems too small
            raise ValueError(f"Model file seems too small ({file_size} bytes). It may be corrupted.")
        
        # Check file extension
        valid_extensions = ['.keras', '.h5']
        if not any(model_path.endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Invalid model file extension. Expected {valid_extensions}")
        
        return True
    
    def load_model_with_retry(self, model_path, max_retries=3):
        """
        Load model with retry mechanism
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading model from {model_path} (attempt {attempt + 1})")
                start_time = time.time()
                
                # Load the model
                model = load_model(model_path, compile=False)
                
                # Compile the model for inference
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                self.load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {self.load_time:.2f} seconds")
                
                return model
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Wait before retry
        
        return None
    
    def get_model_info(self, model):
        """
        Extract information about the loaded model
        """
        try:
            info = {
                'total_params': model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(p) for p in model.trainable_weights]),
                'non_trainable_params': sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights]),
                'layers': len(model.layers),
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024) if self.model_path else 0,
                'load_time': self.load_time or 0
            }
            return info
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def load_trained_model(self):
        """
        Main function to load the trained model
        """
        try:
            # Find model path
            self.model_path = self.get_model_path()
            if not self.model_path:
                raise FileNotFoundError(
                    "Model file not found. Please ensure 'resnet50_trained.keras' is in the models/ directory.\n"
                    "Expected locations:\n"
                    "- models/resnet50_trained.keras\n"
                    "- models/resnet50_trained.h5\n"
                    "\nIf the model is stored elsewhere, please update the model path in model.py"
                )
            
            # Validate model file
            self.validate_model_file(self.model_path)
            
            # Load model with retry
            self.model = self.load_model_with_retry(self.model_path)
            
            # Get model information
            self.model_info = self.get_model_info(self.model)
            
            logger.info("Model loaded and ready for inference")
            return self.model
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def predict_with_preprocessing(self, image_array):
        """
        Make prediction with additional preprocessing and validation
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_trained_model() first.")
        
        try:
            # Validate input shape
            expected_shape = (1, 224, 224, 3)
            if image_array.shape != expected_shape:
                raise ValueError(f"Input shape {image_array.shape} doesn't match expected {expected_shape}")
            
            # Make prediction
            start_time = time.time()
            predictions = self.model.predict(image_array, verbose=0)
            prediction_time = time.time() - start_time
            
            logger.info(f"Prediction completed in {prediction_time:.3f} seconds")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e
    
    def get_model_summary(self):
        """
        Get a formatted model summary
        """
        if self.model is None:
            return "Model not loaded"
        
        try:
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                self.model.summary()
            summary = f.getvalue()
            return summary
        except:
            return "Unable to generate model summary"

# Global model manager instance
model_manager = ModelManager()

def load_trained_model():
    """
    Public function to load the trained model (maintains backward compatibility)
    """
    return model_manager.load_trained_model()

def get_model_info():
    """
    Get information about the currently loaded model
    """
    return model_manager.model_info

def make_prediction(image_array):
    """
    Make a prediction using the loaded model
    """
    return model_manager.predict_with_preprocessing(image_array)

def check_model_health():
    """
    Perform a health check on the model
    """
    try:
        if model_manager.model is None:
            return False, "Model not loaded"
        
        # Create a dummy input to test the model
        dummy_input = tf.random.normal((1, 224, 224, 3))
        prediction = model_manager.model.predict(dummy_input, verbose=0)
        
        if prediction is not None and prediction.shape == (1, 5):
            return True, "Model is healthy and responsive"
        else:
            return False, "Model prediction output is unexpected"
    
    except Exception as e:
        return False, f"Model health check failed: {str(e)}"

def get_system_info():
    """
    Get system information for debugging
    """
    info = {
        'tensorflow_version': tf.__version__,
        'python_version': sys.version,
        'model_loaded': model_manager.model is not None,
        'model_path': model_manager.model_path,
        'gpu_available': len(tf.config.experimental.list_physical_devices('GPU')) > 0
    }
    return info
