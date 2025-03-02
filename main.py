from fastapi import FastAPI
import coremltools as ct
import numpy as np

app = FastAPI()

# Load model
model = ct.models.MLModel("PadamML.mlmodel")

@app.get("/")
def home():
    return {"message": "CoreML Model API Running"}

@app.post("/predict/")
def predict(data: list):
    input_data = np.array(data)
    prediction = model.predict({"input": input_data})
    return {"prediction": prediction}
