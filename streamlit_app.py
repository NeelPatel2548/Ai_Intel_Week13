import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="InspectorsAlly", layout="centered")

import keras
import numpy as np
from PIL import Image
import os

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

model = load_model()

# Define class labels
class_names = ["Defective", "Non_Defective"]

# UI settings
st.title("🔍 InspectorsAlly: Anomaly Detector")
st.markdown("Upload a product image to check for **defects** using a Teachable Machine model.")

# Image uploader
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocessing
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model is not None:
            # Predict
            prediction = model.predict(img_array, verbose=0)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            # Result
            st.subheader("Prediction")
            st.success(f"🧠 **{class_names[class_index]}** with **{confidence * 100:.2f}%** confidence")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
