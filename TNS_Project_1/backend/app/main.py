from fastapi import FastAPI, HTTPException
from app.schemas import InputData
from app.model import ManufacturingModel
import torch
import joblib
import os

app = FastAPI(title="Manufacturing Production Predictor API")

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "final_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")

# -------- Load model & scaler --------
model = ManufacturingModel()
scaler = None

try:
    scaler = joblib.load(SCALER_PATH)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")

# -------- Routes --------
@app.get("/")
def home():
    return {
        "status": "Backend is running",
        "features_required": 17
    }

@app.post("/predict")
async def predict(data: InputData):
    # ✅ Validation
    if len(data.features) != 17:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 17 features, got {len(data.features)}"
        )

    try:
        # Scale features
        scaled_data = scaler.transform([data.features])
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # Model prediction
        with torch.no_grad():
            prediction = model(input_tensor)

        return {
            "prediction": round(float(prediction[0][0]), 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
