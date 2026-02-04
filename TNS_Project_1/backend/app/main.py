from fastapi import FastAPI
from .schemas import InputData
from .utils import predict

app = FastAPI(title="TNS ML Backend API")

@app.get("/")
def root():
    return {"status": "Backend running successfully"}

@app.post("/predict")
def predict_api(data: InputData):
    preds = predict(data.features)
    return {"predictions": preds}
