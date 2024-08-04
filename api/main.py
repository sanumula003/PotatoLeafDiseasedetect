import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(file) -> np.ndarray:
    image = np.array(Image.open(file))
    return image

def predict(image):
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

# Streamlit UI
st.title("Potato Disease Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class, confidence = predict(image)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence}")
