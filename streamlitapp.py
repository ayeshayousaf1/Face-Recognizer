import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to the input size expected by the model (150x150 in this case)
    image = image.resize((150, 150))  # Resize to 150x150
    image = np.array(image)  # Convert to NumPy array
    if image.shape[-1] == 4:  # Remove alpha channel if necessary (RGBA to RGB)
        image = image[..., :3]
    image = image / 255.0  # Normalize pixel values to [0, 1]
    
    # Add batch dimension (make it shape: (1, 150, 150, 3))
    img_array = np.expand_dims(image, axis=0)
    
    return img_array

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

        # Extract the prediction value
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

