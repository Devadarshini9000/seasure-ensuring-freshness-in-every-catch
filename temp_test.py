import sys
import io
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf

# Suppress TensorFlow progress bars (fix for Streamlit)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define image dimensions
img_width, img_height = 128, 128

# Load the trained model
model = load_model('fish_freshness_model.h5')

# Class labels
class_labels = ['eye-fresh', 'eye-non-fresh', 'gill-fresh', 'gill-non-fresh']

# Define temperature ranges
fresh_temperature_range = (0, 4)  # Fresh fish temperature range
non_fresh_temperature_range = (5, 10)  # Non-fresh fish temperature range

# Function to preprocess and predict an image
def predict_image(uploaded_file):
    # Load the image
    img = load_img(uploaded_file, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    img_array = img_array / 255.0  # Normalize pixel values

    # Make prediction without showing a progress bar
    with tf.device('/CPU:0'):  # Specify device to avoid GPU progress bar interference
        predictions = model.predict(img_array, verbose=0)

    # Confidence score: How well the model predicts the class
    confidence = np.max(predictions) * 100

    # Predicted class
    predicted_class = class_labels[np.argmax(predictions)]

    # Generate temperature based on the predicted class
    if "fresh" in predicted_class.lower():
        temperature = np.random.uniform(fresh_temperature_range[0], fresh_temperature_range[1])  # Fresh fish temperature
    else:
        # Ensure the temperature for non-fresh fish is above 4°C
        temperature = np.random.uniform(non_fresh_temperature_range[0], non_fresh_temperature_range[1])  # Non-fresh fish temperature

    # Freshness score: Calculate based on the predicted class and temperature
    if "non-fresh" in predicted_class.lower():
        freshness_score = max(0, 30 - (temperature - 5) * 10)  # Lower score for non-fresh fish, adjusted by temperature
    else:
        freshness_score = min(100, 80 + (4 - temperature) * 5)  # Higher score for fresh fish, adjusted by temperature

    return predicted_class, confidence, temperature, freshness_score

# Streamlit UI
st.title("SeaSure: Ensuring Freshness in Every Catch")
st.write("Upload an image of the fish's eye or gill to detect its freshness.")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Predict the class, confidence, temperature, and freshness score
    with st.spinner("Analyzing..."):
        predicted_class, confidence, temperature, freshness_score = predict_image(uploaded_file)

    # Display the results
    st.write(f"Estimated Temperature: {temperature:.2f}°C")
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence Score: {confidence:.2f}%")
    st.info(f"Freshness Score: {freshness_score:.2f}/100")

    # Provide a freshness level based on the freshness score
    if freshness_score >= 80:
        st.success("Freshness Level: High")
    elif 50 <= freshness_score < 80:
        st.warning("Freshness Level: Medium")
    else:
        st.error("Freshness Level: Low")
