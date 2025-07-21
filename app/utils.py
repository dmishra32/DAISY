import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ExifTags
import io

def validate_image(uploaded_file):
    """
    Validate uploaded image file
    """
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        raise ValueError("File size too large. Please upload an image smaller than 10MB.")
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    if uploaded_file.type not in allowed_types:
        raise ValueError("Unsupported file type. Please upload JPG or PNG images only.")
    
    return True

def correct_image_orientation(image):
    """
    Correct image orientation based on EXIF data
    """
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        # If no EXIF data or error reading it, continue without rotation
        pass
    
    return image

def enhance_image_quality(image):
    """
    Enhance image quality for better analysis
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    # Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.05)
    
    return image

def preprocess_for_model(image_array, target_size=(224, 224)):
    """
    Preprocess image array for model input
    """
    # Resize image
    if image_array.shape[:2] != target_size:
        image_resized = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LANCZOS4)
    else:
        image_resized = image_array.copy()
    
    # Normalize pixel values to [0, 1]
    image_normalized = (image_resized / 255.0).astype(np.float32)
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

def process_image(uploaded_file):
    """
    Main image processing function with comprehensive preprocessing
    """
    try:
        # Validate the uploaded file
        validate_image(uploaded_file)
        
        # Read image data
        image_data = uploaded_file.read()
        
        # Convert to PIL Image for initial processing
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Correct orientation if needed
        pil_image = correct_image_orientation(pil_image)
        
        # Convert to RGB if needed (handle RGBA, grayscale, etc.)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Enhance image quality
        pil_image = enhance_image_quality(pil_image)
        
        # Convert PIL to numpy array
        image_array = np.array(pil_image)
        
        # Preprocess for model
        processed_image = preprocess_for_model(image_array)
        
        return processed_image
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        raise e

def get_image_info(uploaded_file):
    """
    Extract basic information about the uploaded image
    """
    try:
        image_data = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        info = {
            'filename': uploaded_file.name,
            'format': pil_image.format,
            'mode': pil_image.mode,
            'size': pil_image.size,
            'width': pil_image.width,
            'height': pil_image.height,
            'file_size': len(image_data),
            'file_size_mb': round(len(image_data) / (1024 * 1024), 2)
        }
        
        return info
        
    except Exception as e:
        return {'error': str(e)}

def create_image_grid(images, titles=None, max_cols=3):
    """
    Create a grid layout for displaying multiple images
    """
    if titles is None:
        titles = [f"Image {i+1}" for i in range(len(images))]
    
    cols = min(len(images), max_cols)
    rows = (len(images) + cols - 1) // cols
    
    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(images):
                with columns[col]:
                    st.image(images[idx], caption=titles[idx], use_column_width=True)

def calculate_image_statistics(image_array):
    """
    Calculate basic statistics about the image
    """
    try:
        stats = {
            'mean_brightness': np.mean(image_array),
            'std_brightness': np.std(image_array),
            'min_value': np.min(image_array),
            'max_value': np.max(image_array),
            'dynamic_range': np.max(image_array) - np.min(image_array)
        }
        
        # Calculate per-channel statistics for RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            stats['red_mean'] = np.mean(image_array[:, :, 0])
            stats['green_mean'] = np.mean(image_array[:, :, 1])
            stats['blue_mean'] = np.mean(image_array[:, :, 2])
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}

def detect_image_quality_issues(image_array):
    """
    Detect potential quality issues in the uploaded image
    """
    issues = []
    
    try:
        # Check brightness
        mean_brightness = np.mean(image_array)
        if mean_brightness < 50:
            issues.append("Image appears too dark - consider better lighting")
        elif mean_brightness > 200:
            issues.append("Image appears overexposed - reduce lighting or camera exposure")
        
        # Check contrast
        std_brightness = np.std(image_array)
        if std_brightness < 30:
            issues.append("Low contrast detected - image may appear flat")
        
        # Check if image is too blurry (using Laplacian variance)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            issues.append("Image may be blurry - ensure camera is focused")
        
        # Check image size
        height, width = image_array.shape[:2]
        if width < 224 or height < 224:
            issues.append("Image resolution is low - use higher quality camera if possible")
        
        return issues
        
    except Exception as e:
        return [f"Error analyzing image quality: {str(e)}"]