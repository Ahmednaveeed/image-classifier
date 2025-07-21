import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('catsVSdpgs.keras')

# Define function to preprocess image
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape((1, 256, 256, 3))
    return image_array

# Title
st.title("Cat vs Dog Classifier 🐱🐶")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)[0][0]

    st.write("### Prediction Result:")
    if prediction >= 0.5:
        st.success(f"This looks like a **Dog** 🐶 (Confidence: {prediction:.2f})")
    else:
        st.success(f"This looks like a **Cat** 🐱 (Confidence: {1 - prediction:.2f})")
