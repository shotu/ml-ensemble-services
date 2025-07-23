import os
import time
import logging
import threading
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from enum import Enum

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nsfw_service")

# Initialize FastAPI app
app = FastAPI(
    title="NSFW Detection Service",
    version="1.0.0",
    description="Detects whether input text is NSFW or safe."
)

# Define enums and response models
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    NSFW_EVALUATION = "nsfw_evaluation"

class NSFWRequest(BaseModel):
    text: str

class NSFWResponse(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

# Global variables for model state
classifier = None
model_ready = False

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
    
    # Default payload for NSFW detection - configurable via environment
    payload = {
        "text": os.getenv("PING_TEXT", "Hello, how are you?")
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

# Load model and tokenizer at startup
@app.on_event("startup")
async def load_model():
    global classifier, model_ready
    try:
        model_dir = "/app/model_cache"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        model_ready = True
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        classifier = None
        model_ready = False
    
    # Start background ping service if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

# Health check endpoint
@app.get("/health")
def health_check():
    if classifier:
        return {"status": "ok"}
    else:
        return {"status": "error", "message": "Model not loaded."}

# NSFW detection endpoint
@app.post("/detect/nsfw", response_model=NSFWResponse)
def detect_nsfw(request: NSFWRequest):
    if not classifier:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    start_time = time.time()
    
    try:
        result = classifier(request.text)[0]
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Convert label to score (NSFW probability)
        # If label is 'NSFW', use the score directly
        # If label is 'SAFE', use 1 - score to get NSFW probability
        if result["label"].upper() == "NSFW":
            nsfw_score = result["score"]
        else:
            nsfw_score = 1.0 - result["score"]
        
        response = NSFWResponse(
            metric_name=EvaluationType.NSFW_EVALUATION,
            actual_value=float(nsfw_score),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": processing_time_ms,
                "text_length": len(request.text),
                "model_label": result["label"],
                "model_raw_score": float(result["score"]),
                "model_name": "eliasalbouzidi/distilbert-nsfw-text-classifier"
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference error.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 