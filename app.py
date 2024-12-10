from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
import torch
import asyncio

logging.set_verbosity_debug()

# Initialize FastAPI app
app = FastAPI(title="Hugging Face Model API")

import os
#local mount point for local test
#MODEL_PATH = os.getenv("MODEL_PATH", "./model") 

#gcp mount from gcsfuse
#print("Files in mounted directory:", os.listdir("/app/model"))
#MODEL_PATH = "/app/model"
tokenizer, model = None, None
is_model_ready = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def load_model():
    global tokenizer, model, is_model_ready
    try:
        print("Starting model load...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("Tokenizer loaded successfully.")
        model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device))
        is_model_ready = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Request schema
class ModelRequest(BaseModel):
    prompt: str
    max_length: int = 50

@app.on_event("startup")
async def startup_event():
    print("Starting application...")
    asyncio.create_task(load_model())
    print("Application is ready to serve requests.")

# Endpoint for inference
@app.post("/generate")
async def generate_text(request: ModelRequest):
    if not is_model_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=request.max_length,
            num_return_sequences=1,
            do_sample=True,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "API is running"}

# Status endpoint
@app.get("/status")
async def status():
    return {"model_ready": is_model_ready}