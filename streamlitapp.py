import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.h5')


# Streamlit app
st.title("Face Mask Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

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
    
    
    # Display the prediction
    if prediction[0] > 0.5:
        st.write("The person is NOT wearing a mask.")
    else:
        st.write("The person is wearing a mask.")
else:
    st.write("Please upload an image to make a prediction.")
