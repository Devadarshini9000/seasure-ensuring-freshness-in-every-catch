import sys
import io
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Suppress TensorFlow progress bars (fix for Streamlit)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define image dimensions
img_width, img_height = 128, 128

# Load the trained model
model = load_model('fish_freshness_model.h5')

# Class labels
class_labels = ['eye-fresh', 'eye-non-fresh', 'gill-fresh', 'gill-non-fresh']

# Function to preprocess and predict an image
def predict_image(uploaded_file):
    # Load the image
    img = image.load_img(uploaded_file, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    img_array = img_array / 255.0  # Normalize pixel values

    # Make prediction without showing a progress bar
    with tf.device('/CPU:0'):  # Specify device to avoid GPU progress bar interference
        predictions = model.predict(img_array, verbose=0)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return predicted_class, confidence

# Streamlit UI
st.title("SeaSure: Ensuring Freshness in Every Catch")
st.write("Upload an image of the fish's eye or gill to detect its freshness.")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)


    # Predict the class
    with st.spinner("Analyzing..."):
        predicted_class, confidence = predict_image(uploaded_file)

    # Display the result
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
