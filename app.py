import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your model (cached for performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("F:\AI Intel + GTU\My Work'\Week13_Ours_toothbrush\keras_model.h5")

model = load_model()

# Define class labels
class_names = ["Defective", "Non_Defective"]

# UI settings
st.set_page_config(page_title="InspectorsAlly", layout="centered")
st.title("🔍 InspectorsAlly: Anomaly Detector")
st.markdown("Upload a product image to check for **defects** using a Teachable Machine model.")

# Image uploader
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Result
    st.subheader("Prediction")
    st.success(f"🧠 **{class_names[class_index]}** with **{confidence * 100:.2f}%** confidence")
