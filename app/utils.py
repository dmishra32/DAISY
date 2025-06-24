import cv2
import numpy as np

def process_image(uploaded_file):
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = (img_resized / 255.0).astype(np.float32)  # Use float32 instead of float64
    return np.expand_dims(img_normalized, axis=0)