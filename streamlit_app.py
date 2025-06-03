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
st.markdown("Detect defects using either live camera or uploaded images.")

# Create two main columns for the vertical split
left_col, right_col = st.columns([1, 1])

# Initialize session state
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_confidence' not in st.session_state:
    st.session_state.last_confidence = None
if 'last_image' not in st.session_state:
    st.session_state.last_image = None

# Left column - Input Options
with left_col:
    st.subheader("📥 Input Options")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Upload Image", "📸 Live Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.session_state.last_image = image
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
    
    with tab2:
        st.subheader("📸 Live Camera Feed")
        camera_input = st.camera_input("Take a photo")
        
        if camera_input is not None:
            try:
                image = Image.open(camera_input).convert("RGB")
                st.session_state.last_image = image
            except Exception as e:
                st.error(f"Error processing camera image: {str(e)}")

# Right column - Analysis Results
with right_col:
    st.subheader("🔍 Analysis Results")
    
    if st.session_state.last_image is not None:
        try:
            if model is not None:
                # Make prediction
                class_index, confidence = predict_image(model, st.session_state.last_image)
                
                # Update session state
                st.session_state.last_prediction = class_index
                st.session_state.last_confidence = confidence
                
                # Display result with color-coded message
                if class_index == 0:  # Defective
                    st.error(f"⚠️ **{class_names[class_index]}** detected with **{confidence * 100:.2f}%** confidence")
                else:  # Non-Defective
                    st.success(f"✅ **{class_names[class_index]}** with **{confidence * 100:.2f}%** confidence")
                
                # Display the analyzed frame
                st.image(st.session_state.last_image, caption="Analyzed Image", use_column_width=True)
        
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
    else:
        st.info("👆 Upload an image or take a photo to start analysis")
