import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to the input size expected by the model
    image = np.array(image)  # Convert to a NumPy array
    if image.shape[-1] == 4:  # If the image has an alpha channel, remove it (RGBA -> RGB)
        image = image[..., :3]
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Face Mask Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict the mask status
        prediction = model.predict(processed_image)

        # Extract the prediction value (assuming binary classification)
        prediction_value = float(prediction[0][0])

        # Display the prediction
        if prediction_value > 0.5:
            st.write("The person is NOT wearing a mask.")
        else:
            st.write("The person is wearing a mask.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("Please upload an image to make a prediction.")
