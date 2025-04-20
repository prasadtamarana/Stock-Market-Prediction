import pickle
import streamlit as st
import numpy as np
import base64

# Convert image to Base64 and set background
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded}") center/cover no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(r"C:\Users\prasa\OneDrive\Pictures\th (9).jpeg")  # Use the uploaded image


# Title and Description
st.markdown('<h1 style="color:#FF5733; text-align:center;">Stock Market Prediction App</h1>', unsafe_allow_html=True)

st.markdown('<p style="font-weight:bold;">This app uses a pre-trained model to predict stock market trends. Enter the necessary details to get predictions.</p>', unsafe_allow_html=True)

# Define function to load the model
@st.cache_resource
def load_model():
    try:
        with open(r"final_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None

# Load the model
model = load_model()

if model:  # Check if the model loaded successfully
    st.success("Model loaded successfully!")

    # Input features for prediction (example inputs; customize as needed)
    st.markdown("### Enter Feature Values")
    feature_1 = st.number_input("Feature 1 (e.g., Open Price):", min_value=0.0, step=0.01)
    feature_2 = st.number_input("Feature 2 (e.g., Volume):", min_value=0.0, step=0.01)
    feature_3 = st.number_input("Feature 3 (e.g., Market Cap):", min_value=0.0, step=0.01)

    # Predict and display results
    if st.button("Predict"):
        try:
            # Convert input features to NumPy array
            input_features = np.array([[feature_1, feature_2, feature_3]])
            prediction = model.predict(input_features)
            st.write(f"**Prediction:** {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("No model loaded. Predictions cannot be made.")