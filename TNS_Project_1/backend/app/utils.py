import os
import torch
from .model import SimpleNet

# Get absolute path to backend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to model file
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "final_model.pth")

device = torch.device("cpu")

# Load model
model = SimpleNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict(features):
    input_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.tolist()
