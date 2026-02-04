import torch
from .model import ManufacturingModel

MODEL_PATH = "saved_models/final_model.pth"
device = torch.device("cpu")

model = ManufacturingModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def predict(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    # Return the raw continuous value (Parts_Per_Hour)
    return output.flatten().tolist()