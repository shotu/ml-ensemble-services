import os
import time
import logging
import threading
import requests
from enum import Enum

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------- Logging Configuration -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bertweet_base_service")

# ------------------- FastAPI App Definition -------------------
app = FastAPI(
    title="BERTweet-Base Service",
    version="1.0.0",
    description="Evaluates tone of text using BERTweet sentiment model."
)

# ------------------- Enums & Response Schema -------------------
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    RESPONSE_TONE_EVALUATION = "response_tone_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

# ------------------- Request Schema -------------------
class ToneRequest(BaseModel):
    text: str

# ------------------- Model & Pipeline Load -------------------
tokenizer = None
model = None

# Label mappings for BERTweet sentiment model
LABEL_MAPPING = {
    'LABEL_0': 'NEG',
    'LABEL_1': 'NEU', 
    'LABEL_2': 'POS'
}

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        model_dir = "/app/model_cache"
        logger.info("Loading BERTweet sentiment model...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Optimize model for inference (CPU)
        model.eval()
        logger.info("Model loaded successfully on CPU")
        
        # Start background ping thread if enabled
        if os.getenv("ENABLE_PING", "false").lower() == "true":
            threading.Thread(target=background_ping, daemon=True).start()
            logger.info("Background ping service started")
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError("Model failed to load")

# ------------------- Health Check -------------------
@app.get("/health")
def health_check():
    if model is None or tokenizer is None:
        return {"status": "warning", "message": "Model not loaded"}
    return {"status": "ok"}

# ------------------- Background Pinger -------------------
def background_ping():
    """
    Background service to ping the response tone endpoint periodically.
    Uses configurable environment variables for flexibility.
    """
    # Get configuration from environment variables
    ping_url = os.getenv("PING_URL")
    ping_interval = int(os.getenv("PING_INTERVAL_SECONDS", "300"))  # Default: 5 minutes
    api_key = os.getenv("PING_API_KEY")  # Optional API key for gateway
    
    if not ping_url:
        logger.warning("PING_URL not configured, skipping ping service")
        return
    
    # Default payload for response tone detection
    payload = {
        "text": os.getenv("PING_TEXT", "I am very happy with this excellent service!")
    }
    
    # Set up headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    logger.info(f"Starting ping service: URL={ping_url}, interval={ping_interval}s")
    
    while True:
        try:
            if model is not None and tokenizer is not None:  # Only ping when model is ready
                logger.info(f"Pinging endpoint: {ping_url}")
                response = requests.post(
                    ping_url, 
                    json=payload, 
                    headers=headers, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    logger.info(f"Ping successful: {response.status_code}")
                else:
                    logger.warning(f"Ping returned non-200 status: {response.status_code}")
            else:
                logger.info("Model not ready, skipping ping")
                
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            
        time.sleep(ping_interval)

# ------------------- Optimized Prediction Functions -------------------
@torch.no_grad()  # Disable gradient computation for inference (works on CPU)
def predict_sentiment_optimized(text: str):
    """Optimized sentiment prediction for CPU inference"""
    # Tokenize input with proper truncation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get model predictions (CPU)
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    confidence = torch.max(probabilities).item()
    
    # Convert to label
    label_key = f'LABEL_{predicted_class_idx}'
    predicted_label = LABEL_MAPPING.get(label_key, label_key)
    
    return predicted_label, confidence

def convert_to_sentiment_polarity_score_optimized(label: str, confidence: float):
    """
    Optimized sentiment polarity score conversion (0.0 = negative, 1.0 = positive)
    """
    # Convert to continuous polarity score (0=negative, 1=positive)
    if label == 'POS':
        # Positive sentiment: map confidence to upper half (0.5-1.0)
        polarity_score = 0.5 + (confidence * 0.5)
        
    elif label == 'NEG':
        # Negative sentiment: map confidence to lower half (0.0-0.5)
        polarity_score = 0.5 - (confidence * 0.5)
        
    else:  # NEU
        # Neutral sentiment: vary slightly based on confidence for better continuity
        if confidence > 0.8:
            polarity_score = 0.5  # Very confident neutral
        else:
            # Less confident neutral - add small variation
            polarity_score = 0.45 + (confidence * 0.1)  # Range: 0.45-0.55
    
    # Ensure score is within bounds
    polarity_score = max(0.0, min(1.0, polarity_score))
    
    return polarity_score

def predict_tone_optimized(text: str) -> tuple[float, str, float]:
    """
    Optimized sentiment tone prediction with direct model calls
    Returns: (polarity_score, predicted_label, confidence)
    """
    # Get prediction using optimized function
    predicted_label, confidence = predict_sentiment_optimized(text)
    
    # Convert to sentiment polarity score
    polarity_score = convert_to_sentiment_polarity_score_optimized(predicted_label, confidence)
    
    return polarity_score, predicted_label, confidence

# ------------------- Inference Endpoint -------------------
@app.post("/detect/response_tone", response_model=MetricReturnModel)
async def evaluate_tone(req: ToneRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    t0 = time.time()
    try:
        polarity_score, predicted_label, confidence = predict_tone_optimized(req.text)
        
        inference_time_ms = round((time.time() - t0) * 1000, 2)

        result = MetricReturnModel(
            metric_name=EvaluationType.RESPONSE_TONE_EVALUATION,
            actual_value=polarity_score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "inference_time_ms": inference_time_ms,
                "text_length": len(req.text),
                "raw_label": predicted_label,
                "raw_score": polarity_score,
                "sentiment_confidence": confidence,
                "score_interpretation": "0.0=negative, 0.5=neutral, 1.0=positive"
            }
        )

        logger.info(f"Tone evaluation: {predicted_label} → polarity={polarity_score:.3f} (conf={confidence:.3f}) in {inference_time_ms}ms")
        return result

    except Exception as e:
        logger.error(f"Error in tone evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 