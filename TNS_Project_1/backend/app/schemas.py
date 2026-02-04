from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    # This receives the list of 17 numbers from Streamlit
    features: List[float]

class PredictionResponse(BaseModel):
    predicted_output: float