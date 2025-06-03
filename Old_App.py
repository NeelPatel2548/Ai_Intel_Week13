import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="InspectorsAlly", layout="wide")

import keras
import numpy as np
from PIL import Image
import os
import io
import time

# Load your model (cached for performance)
@st.cache_resource
def load_model():
    try:
        model_path = "keras_model.h5"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, target_size=(250, 250)):
    # Resize image to exact dimensions
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to bytes with compression
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='JPEG', quality=70)
    img_byte_arr.seek(0)
    
    return resized_image

def predict_image(model, image):
    # Preprocessing for model
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    
    return class_index, confidence

model = load_model()

# Define class names
class_names = ["Defective", "Non_Defective"]

# UI settings
st.title("🔍 InspectorsAlly: Anomaly Detector")
st.markdown("Upload a product image or use live camera to check for **defects** using a Teachable Machine model.")

# Create two main columns for the vertical split
left_col, right_col = st.columns([1, 1])

# Initialize session state for live prediction
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_confidence' not in st.session_state:
    st.session_state.last_confidence = None
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
if 'camera_frame' not in st.session_state:
    st.session_state.camera_frame = None
if 'cropped_image' not in st.session_state:
    st.session_state.cropped_image = None

# Left column - Input options
with left_col:
    st.subheader("📥 Input Options")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Upload Image", "📸 Live Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    with tab2:
        camera_input = st.camera_input("Live camera feed")
        if camera_input is not None:
            # Store the current frame
            st.session_state.camera_frame = camera_input
            
            # Add cropping options
            st.subheader("✂️ Crop Options")
            crop_enabled = st.checkbox("Enable Cropping", value=False)
            
            if crop_enabled and st.session_state.camera_frame is not None:
                # Get image dimensions
                img = Image.open(st.session_state.camera_frame)
                width, height = img.size
                
                # Create columns for crop coordinates
                col1, col2 = st.columns(2)
                with col1:
                    left = st.slider("Left", 0, width, 0)
                    top = st.slider("Top", 0, height, 0)
                with col2:
                    right = st.slider("Right", 0, width, width)
                    bottom = st.slider("Bottom", 0, height, height)
                
                # Crop the image
                if right > left and bottom > top:
                    cropped = img.crop((left, top, right, bottom))
                    st.session_state.cropped_image = cropped
                    st.image(cropped, caption="Cropped Preview", width=250)
            
            analyze_button = st.button("Analyze Current Frame")

# Right column - Preview and Results
with right_col:
    st.subheader("🔍 Preview & Results")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            processed_image = process_image(input_image)
            st.image(processed_image, caption="Input Image", width=250)

            if model is not None:
                class_index, confidence = predict_image(model, processed_image)
                st.success(f"🧠 **{class_names[class_index]}** with **{confidence * 100:.2f}%** confidence")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Process live camera feed
    elif st.session_state.camera_frame is not None and 'analyze_button' in locals() and analyze_button:
        try:
            # Get the appropriate image (cropped or original)
            if st.session_state.cropped_image is not None:
                current_image = st.session_state.cropped_image
            else:
                current_image = Image.open(st.session_state.camera_frame).convert("RGB")
            
            processed_image = process_image(current_image)
            
            # Display the processed image
            st.image(processed_image, caption="Live Camera Feed", width=250)
            
            if model is not None:
                # Make prediction
                class_index, confidence = predict_image(model, processed_image)
                
                # Update session state
                st.session_state.last_prediction = class_index
                st.session_state.last_confidence = confidence
                st.session_state.last_image = processed_image
                
                # Display result
                st.success(f"🧠 **{class_names[class_index]}** with **{confidence * 100:.2f}%** confidence")
        
        except Exception as e:
            st.error(f"Error processing camera feed: {str(e)}")
    
    # Display last prediction if available
    elif st.session_state.last_prediction is not None:
        st.image(st.session_state.last_image, caption="Last Analyzed Frame", width=250)
        st.success(f"🧠 **{class_names[st.session_state.last_prediction]}** with **{st.session_state.last_confidence * 100:.2f}%** confidence")
