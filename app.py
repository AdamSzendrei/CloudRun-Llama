
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
import torch
import asyncio
import threading
import os

logging.set_verbosity_debug()

# Initialize FastAPI app
app = FastAPI(title="Huggingface Model API")

# Placeholder for model and tokenizer
model = None
tokenizer = None
model_loading = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/app/model" 

# Asynchronous function to load the model
def load_model_in_background():
    global model, tokenizer, model_loading
    try:
        print("Loading model in the background...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        model_loading = False  # Model is ready
        print("Model loaded successfully!")
    except Exception as e:
        model_loading = False
        print(f"Error loading model: {e}")

# Start model loading in a background thread
threading.Thread(target=load_model_in_background, daemon=True).start()

# Request schema
class ModelRequest(BaseModel):
    prompt: str
    max_length: int = 150

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "API is running"}

# Define an endpoint to check model readiness
@app.get("/status")
async def get_status():
    if model_loading:
        return {"status": "Model is loading. Please wait."}
    return {"status": "Model is ready."}

# Example inference endpoint
@app.post("/generate-text")
async def generate_text(request: ModelRequest):
    if model_loading:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again later.")
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    outputs = model.generate(
            inputs["input_ids"],
            max_length=request.max_length,
            num_return_sequences=1,
            do_sample=True,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}