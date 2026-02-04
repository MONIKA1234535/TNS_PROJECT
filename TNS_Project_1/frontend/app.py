import streamlit as st
import requests

st.set_page_config(page_title="ML Model Predictor")

st.title("ML Model Prediction App")

# Example: change number of inputs based on your model
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")

if st.button("Predict"):
    payload = {
        "features": [feature1, feature2]
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        if response.status_code == 200:
            st.success(f"Prediction: {response.json()['prediction']}")
        else:
            st.error("Prediction failed")

    except Exception as e:
        st.error(f"Backend not reachable: {e}")
