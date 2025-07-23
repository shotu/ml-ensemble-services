# main.py
import os
import time
import logging
import threading
import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from enum import Enum

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("longformer_service")

app = FastAPI(
    title="Longformer Service to detect coherence",
    version="1.0.0",
    description="Computes coherence score for text"
)

# Enums & Response Schema
class ActualValueDtype(str, Enum):
    FLOAT = "float"
    INT = "int"
    STRING = "string"

class EvaluationType(str, Enum):
    COHERENCE_EVALUATION = "coherence_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

# Model Loading - Optimized for performance
model = None
tokenizer = None
model_ready = False

class CoherenceRequest(BaseModel):
    text: str

# ------------------- Background Services -------------------
def background_ping():
    """
    Background service to ping the API endpoint periodically.
    Uses configurable environment variables for flexibility.
    """
    # Get configuration from environment variables
    ping_url = os.getenv("PING_URL")
    ping_interval = int(os.getenv("PING_INTERVAL_SECONDS", "300"))  # Default: 5 minutes
    api_key = os.getenv("PING_API_KEY")  # Optional API key for gateway
    
    if not ping_url:
        logger.warning("PING_URL not configured, skipping ping service")
        return
    
    # Default payload for coherence detection - configurable via environment
    payload = {
        "text": os.getenv("PING_TEXT", "This is a coherent sentence that makes sense.")
    }
    
    # Set up headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    logger.info(f"Starting ping service: URL={ping_url}, interval={ping_interval}s")
    
    while True:
        try:
            if model_ready:  # Only ping when model is ready
                logger.info(f"Pinging endpoint: {ping_url}")
                response = requests.post(ping_url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    logger.info(f"Ping successful: {response.status_code}")
                else:
                    logger.warning(f"Ping returned non-200 status: {response.status_code}")
            else:
                logger.info("Model not ready, skipping ping")
                
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            
        time.sleep(ping_interval)

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, model_ready
    try:
        logger.info("Loading Longformer model...")
        model = AutoModelForSequenceClassification.from_pretrained("/app/model_cache")
        tokenizer = AutoTokenizer.from_pretrained("/app/model_cache")
        
        # Set model to evaluation mode for inference (CPU optimization)
        model.eval()
        model_ready = True
        logger.info("Model loaded successfully on CPU")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        model_ready = False
        raise
    
    # Start background ping service if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

@torch.no_grad()  # Optimization: Disable gradient computation (works on CPU)
def predict_coherence(text: str):
    """Optimized coherence prediction for CPU inference"""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096, padding=True)
    
    # Get model predictions (CPU)
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = torch.max(probabilities).item()
    
    return predicted_class, confidence

@app.get("/health")
def health_check():
    if model is None or tokenizer is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok"}

@app.post("/detect/coherence", response_model=MetricReturnModel)
async def detect_coherence(request: CoherenceRequest):
    """
    Runs coherence detection and returns standardized evaluation format.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        t0 = time.time()
        
        # Use optimized prediction function
        predicted_class, confidence = predict_coherence(request.text)
        
        # Convert prediction to coherence score
        # Assuming class 0 = incoherent, class 1 = coherent
        if predicted_class == 0:  # NEGATIVE/incoherent
            coherence_score = 1.0 - confidence  # Invert for incoherent text
            label = "NEGATIVE"
        else:  # POSITIVE/coherent
            coherence_score = confidence  # Direct score for coherent text
            label = "POSITIVE"

        inference_time_ms = round((time.time() - t0) * 1000, 2)

        return MetricReturnModel(
            metric_name=EvaluationType.COHERENCE_EVALUATION,
            actual_value=round(coherence_score, 4),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "raw_label": label, 
                "raw_score": round(confidence, 4), 
                "inference_time_ms": inference_time_ms,
                "predicted_class": predicted_class,
                "text_length": len(request.text)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 