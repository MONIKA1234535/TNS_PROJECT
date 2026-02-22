import os
import torch
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Added for connection
from app.schemas import InputData
from app.model import ManufacturingModel

# 1. Create app instance
app = FastAPI(title="Manufacturing Predictor API")

# 2. ADD CORS MIDDLEWARE (Crucial for Render connection)
# This prevents the "Could not connect to Backend" error in the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    if scaler is None:
         raise HTTPException(status_code=500, detail="Scaler not loaded on server.")

    try:
        # 1. Scale the original 17 features
        scaled_data = scaler.transform([data.features])
        
        # 2. Pad to 20 features for the model (add 3 zeros)
        # This matches your model weight file (final_model.pth)
        full_features = scaled_data.tolist()[0] + [0.0, 0.0, 0.0]
        
        # 3. Convert to Tensor
        input_tensor = torch.tensor([full_features], dtype=torch.float32)

        # 4. Predict
        with torch.no_grad():
            prediction = model(input_tensor)

        # 5. Return with the key your Frontend expects
        # Changed 'prediction' to 'predicted_output' based on your UI error
        return {"predicted_output": round(float(prediction[0][0]), 2)}

    except Exception as e:
        # This will show the error in the Frontend red box if something fails
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")