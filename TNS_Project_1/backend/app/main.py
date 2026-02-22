import os
import torch
import joblib
from fastapi import FastAPI, HTTPException
from app.schemas import InputData
from app.model import ManufacturingModel

# Create app instance before defining routes
app = FastAPI(title="Manufacturing Predictor API")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "final_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")

# Initialize global objects
model = ManufacturingModel()
scaler = None

# Load assets on startup
try:
    scaler = joblib.load(SCALER_PATH)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ System Ready: Model (20) and Scaler (17) loaded.")
except Exception as e:
    print(f"❌ Load Error: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "input_required": 17}

@app.post("/predict")
async def predict(data: InputData):
    # Check frontend input
    if len(data.features) != 17:
        raise HTTPException(status_code=400, detail=f"Expected 17 features, got {len(data.features)}")

    try:
        # 1. Scale the original 17 features
        scaled_data = scaler.transform([data.features])
        
        # 2. Pad to 20 features for the model (add 3 zeros)
        # This prevents the "mat1 and mat2 shapes" error
        full_features = scaled_data.tolist()[0] + [0.0, 0.0, 0.0]
        
        # 3. Convert to Tensor
        input_tensor = torch.tensor([full_features], dtype=torch.float32)

        # 4. Predict
        with torch.no_grad():
            prediction = model(input_tensor)

        return {"prediction": round(float(prediction[0][0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))