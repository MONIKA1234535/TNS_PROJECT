from fastapi import FastAPI, HTTPException
from app.model import ManufacturingModel
from app.schemas import InputData
import torch
import joblib
import os

app = FastAPI(title="Manufacturing API")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "final_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")

# Initialize model and load weights
model = ManufacturingModel()

try:
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Load model weights
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading assets: {e}")

@app.post("/predict")
async def predict(data: InputData):
    try:
        # 1. Ensure we have exactly 17 features
        if len(data.features) != 17:
            raise HTTPException(status_code=400, detail=f"Expected 17 features, got {len(data.features)}")

        # 2. Scale the input
        scaled_data = scaler.transform([data.features])
        
        # 3. Convert to tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        
        # 4. Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
        
        return {"predicted_output": round(float(prediction[0][0]), 2)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "Backend is running", "features_required": 17}