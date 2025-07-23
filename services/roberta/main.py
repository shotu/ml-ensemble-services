import os
import time
import logging
import threading
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ------------------- Logging Configuration -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("roberta_service")

# ------------------- FastAPI App Definition -------------------
app = FastAPI(
    title="RoBERTa Bias Detection Service",
    version="1.0.0",
    description="Detects political bias in text using RoBERTa model"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Request Schema -------------------
class TextInput(BaseModel):
    text: str

# ------------------- Model Loading -------------------
model = None
tokenizer = None
classifier = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, classifier
    try:
        logger.info("Loading RoBERTa model...")
        model = AutoModelForSequenceClassification.from_pretrained("/app/model_cache")
        tokenizer = AutoTokenizer.from_pretrained("/app/model_cache")
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        logger.info("RoBERTa model loaded successfully")

        # Start background ping thread if enabled
        if os.getenv("ENABLE_PING", "false").lower() == "true":
            threading.Thread(target=background_ping, daemon=True).start()
            logger.info("Background ping service started")

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# ------------------- Background Pinger -------------------
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
    
    # Default payload for bias detection
    payload = {
        "text": os.getenv("PING_TEXT", "The liberal policies have failed to address economic inequality.")
    }
    
    # Set up headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    logger.info(f"Starting ping service: URL={ping_url}, interval={ping_interval}s")
    
    while True:
        try:
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
                
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            
        time.sleep(ping_interval)

# ------------------- Health Check -------------------
@app.get("/health")
def health_check():
    if classifier is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok"}

# ------------------- Utility Functions -------------------
def predict_bias(text: str) -> float:
    """
    Predict bias score for the given text.
    Returns the confidence score for the bias prediction.
    """
    results = classifier(text)[0]
    
    label = results["label"]  # Get the predicted label
    score = results["score"]  # Get the confidence score
    
    # Fixed scoring logic with case-sensitive comparison
    # LABEL_1 corresponds to "Biased", LABEL_0 corresponds to "Non-biased"
    if label.upper() == "LABEL_1":
        bias_score = score  # High score for biased content
    else:
        bias_score = 1.0 - score  # Convert non-biased confidence to bias score
    
    return bias_score

# ------------------- Inference Endpoint -------------------
@app.post("/detect/bias")
async def detect_bias(req: TextInput):
    """
    Detects bias in text and returns simplified response.
    """
    start_time = time.time()
    try:
        # Run the bias detection classifier
        results = classifier(req.text)[0]
        label = results["label"]
        raw_score = results["score"]
        
        # Apply corrected scoring logic
        if label.upper() == "LABEL_1":
            bias_score = raw_score  # High score for biased content
        else:
            bias_score = 1.0 - raw_score  # Convert non-biased confidence to bias score
        
        processing_time = time.time() - start_time
        
        result = {
            "metric_name": "bias_evaluation",
            "actual_value": bias_score,
            "actual_value_type": "float",
            "others": {
                "processing_time_ms": processing_time * 1000,  # Convert to milliseconds
                "text_length": len(req.text),
                "model_label": label,
                "model_raw_score": float(raw_score),
                "model_name": "mediabiasgroup/da-roberta-babe-ft"
            }
        }

        logger.info(f"Bias evaluation completed in {processing_time:.4f}s")
        return result

    except Exception as e:
        logger.error(f"Error in bias evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- App Entry Point -------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 