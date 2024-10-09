import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Class names
class_names = ['With Mask', 'Without Mask']

# Streamlit UI
st.title("Face Mask Detection")
st.write("Upload an image to check if the person is wearing a mask.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Convert the image to RGB and resize it to 128x128
    image = image.convert("RGB")
    image = image.resize((128, 128))
    
    # Preprocess the image
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    
    # Predict using the model
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_name = class_names[class_idx]
    
    # Display the prediction
    st.image(image, caption=f"Prediction: {class_name}", use_column_width=True)
    st.write(f"Prediction: {class_name}")