import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title("Face Mask Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the mask status
    prediction = model.predict(processed_image)

    # Display the prediction
    if prediction[0] > 0.5:
        st.write("The person is NOT wearing a mask.")
    else:
        st.write("The person is wearing a mask.")
