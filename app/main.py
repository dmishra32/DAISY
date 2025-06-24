import streamlit as st
import tensorflow as tf
load_model = tf.keras.models.load_model  # Workaround for Pylance false negatives
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to path
from app.model import load_trained_model
from app.utils import process_image

st.set_page_config(layout="wide")
st.title("DAISY: Dermatological AI System for You")
st.write("Industrial-grade skin disease classifier (akiec, bcc, bkl, mel, nv)")

# Load model
try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

class_names = ['akiec', 'bcc', 'bkl', 'mel', 'nv']

uploaded_file = st.file_uploader("Upload skin image", type=['jpg', 'png'], help="Supported formats: JPG, PNG")
if uploaded_file:
    try:
        img = process_image(uploaded_file)
        pred = model.predict(img)
        pred_class = class_names[np.argmax(pred)]
        pred_conf = np.max(pred) * 100
        # Convert back to uint8 for display
        img_display = (img[0] * 255).astype(np.uint8)
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR), caption="Uploaded Image", width=400)
        with col2:
            st.write(f"**Prediction**: {pred_class}")
            st.write(f"**Confidence**: {pred_conf:.2f}%")
            st.bar_chart(dict(zip(class_names, pred[0])))
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.sidebar.header("System Status")
st.sidebar.write("Model: resnet50_trained.keras")
st.sidebar.write("Sync required if model missing.")
st.sidebar.write("Note: Optimization in progress for akiec/bcc/bkl.")