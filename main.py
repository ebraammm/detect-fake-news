from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.layers import TFSMLayer
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the model directory
model_path = "E:/detect_diabetes_fake_news/fake-news/lstm_model"

# Load the TensorFlow SavedModel using TFSMLayer
try:
    model = TFSMLayer(model_path, call_endpoint="serving_default")  # Adjust `call_endpoint` as needed
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model. Ensure the model path and format are correct.")

# Input schema for prediction requests
class TextData(BaseModel):
    text: str

# Dummy preprocessing function (replace with actual logic)
def preprocess_text(text: str):
    sequence_length = 100  # Ensure this matches your model's expected input
    dummy_input = np.zeros((1, sequence_length))  # Replace with actual preprocessing
    return dummy_input

@app.post("/predict")
async def predict_fake_news(data: TextData):
    try:
        # Preprocess the input text
        processed_input = preprocess_text(data.text)

        # Use the model to make predictions
        prediction = model(processed_input)
        fake_news_probability = float(prediction.numpy()[0][0])  # Assuming single-output neuron

        return {"text": data.text, "fake_news_probability": fake_news_probability}
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
