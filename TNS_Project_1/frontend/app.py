import streamlit as st
import requests

st.set_page_config(page_title="Production AI", layout="wide")

st.title("🏭 Manufacturing Production Predictor")

# Define the 17 features for display labels
feature_names = [
    "Machine_ID", "Temperature", "Pressure", "Vibration_Level", 
    "Operational_Hours", "Maintenance_Cycle", "Material_Quality", "Shift",
    "Machine_Type", "Material_Grade", "Day_of_Week", "Speed_Setting",
    "Energy_Consumption", "Coolant_Temperature", "Material_Viscosity",
    "Ambient_Temperature", "Operator_Experience"
]

st.sidebar.header("Input Parameters")
inputs = []

# Create a clean input form
with st.form("input_form"):
    cols = st.columns(3)
    for i, name in enumerate(feature_names):
        val = cols[i % 3].number_input(f"{name}", value=0.0)
        inputs.append(val)
    
    submit = st.form_submit_button("Predict Performance")

if submit:
    try:
        # Post to the FastAPI backend
        response = requests.post(
            "https://tns-project.onrender.com/predict", 
            json={"features": inputs}
        )
        
        if response.status_code == 200:
            prediction = response.json()["predicted_output"]
            st.balloons()
            st.success(f"### 🚀 Predicted Production: {prediction} Parts Per Hour")
        else:
            st.error(f"Backend Error: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to Backend. Is it running? {e}")