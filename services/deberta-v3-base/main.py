import os
import time
import logging
import sys
import threading
import requests
import numpy as np
from enum import Enum
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ------------------- Constants -------------------
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
CACHE_DIR = "/app/model_cache"

# Temperature scaling parameter for better score distribution
# Based on research, values between 1.5-3.0 often work well
TEMPERATURE = 2.0

# ------------------- Configure Model -------------------
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variables for model state
model = None
model_loading = False
model_ready = False
startup_time = time.time()

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("deberta_v3_base_service")

# ------------------- FastAPI App -------------------
app = FastAPI(
    title="DeBERTa v3 Base Prompt Injection Service",
    version="2.0.0",
    description="Optimized prompt injection detection with improved continuous scoring using DeBERTa v3."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Enums & Schema -------------------
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    DEBERTA_V3_BASE_EVALUATION = "deberta_v3_base_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

class DebertaRequest(BaseModel):
    text: str

# ------------------- Model Loading -------------------
def load_model_in_background():
    global model, model_loading, model_ready
    model_loading = True
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        
        model_obj = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        
        # Use simple pipeline without return_all_scores for better performance
        model = pipeline("text-classification", model=model_obj, tokenizer=tokenizer)
        
        logger.info("Model loaded successfully")
        
        # Test the model
        test_result = model("Hello, how are you?")
        logger.info(f"Model test result: {test_result}")
        
        model_ready = True
        logger.info("Model initialization complete")
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
    finally:
        model_loading = False

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
    
    # Default payload for prompt injection detection - configurable via environment
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

@app.on_event("startup")
async def startup_event():
    logger.info("Starting DeBERTa v3 Prompt Injection Service")
    
    thread = threading.Thread(target=load_model_in_background)
    thread.daemon = True
    thread.start()
    logger.info("Model loading started in background thread")
    
    # Start background ping service if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

# ------------------- Helper Functions -------------------
def apply_temperature_scaling(confidence, temperature=TEMPERATURE):
    """
    Apply temperature scaling to improve score distribution.
    
    Args:
        confidence: Raw confidence score from model
        temperature: Temperature parameter (>1 makes scores less extreme)
    
    Returns:
        Calibrated confidence score
    """
    if temperature == 1.0:
        return confidence
    
    # Convert confidence to logit, apply temperature, convert back
    # Avoid numerical issues with extreme values
    confidence = max(0.001, min(0.999, confidence))
    logit = np.log(confidence / (1 - confidence))
    scaled_logit = logit / temperature
    scaled_confidence = 1 / (1 + np.exp(-scaled_logit))
    
    return float(scaled_confidence)

def predict_injection_risk_single(text):
    """
    Predict injection risk for a single text using RoBERTa-style simple scoring.
    
    Based on testing the deployed model, this model returns:
    - "INJECTION" = injection detected
    - "SAFE" = safe/benign content
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (injection_risk_score, label, raw_confidence)
    """
    result = model(text)[0]
    
    label = result["label"]
    raw_confidence = float(result["score"])
    
    # Apply RoBERTa-style scoring logic with actual model labels
    if label == "INJECTION":
        # Model predicts injection with given confidence
        injection_risk_score = raw_confidence
    elif label == "SAFE":
        # Model predicts safe with given confidence
        # Convert to injection risk: high confidence safe = low injection risk
        injection_risk_score = 1.0 - raw_confidence
    else:
        # Unexpected label format - log error and return neutral score
        logger.error(f"Unexpected label format: '{label}'. Expected 'INJECTION' or 'SAFE'.")
        injection_risk_score = 0.5
    
    # Apply temperature scaling for better score distribution
    calibrated_score = apply_temperature_scaling(injection_risk_score)
    
    return calibrated_score, label, raw_confidence

# ------------------- Middleware -------------------
@app.middleware("http")
async def check_model_readiness(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
        
    if not model_ready:
        if model_loading:
            return Response(
                content="Model is still loading, please try again later",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "10"}
            )
        else:
            return Response(
                content="Model failed to load, service is unavailable",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    return await call_next(request)

# ------------------- Endpoints -------------------
@app.post("/detect/prompt_injection", response_model=MetricReturnModel)
async def evaluate_text(req: DebertaRequest):
    """
    Detect prompt injection using optimized DeBERTa v3 model with improved continuous scoring.
    
    Returns injection risk score where:
    - 0.0 = definitely safe
    - 1.0 = definitely injection
    - Values in between represent continuous risk levels
    """
    start_time = time.time()
    text = req.text

    try:
        # Get injection risk prediction using simple approach
        injection_risk_score, model_label, model_raw_score = predict_injection_risk_single(text)
        
        processing_time = (time.time() - start_time) * 1000
        
        return MetricReturnModel(
            metric_name=EvaluationType.DEBERTA_V3_BASE_EVALUATION,
            actual_value=injection_risk_score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": round(processing_time, 2),
                "text_length": len(text),
                "model_label": model_label,
                "model_raw_score": model_raw_score,
                "model_name": MODEL_NAME,
                "calibrated_score": injection_risk_score,
                "temperature_used": TEMPERATURE,
                "model_decision": "INJECTION" if injection_risk_score > 0.5 else "SAFE",
                "confidence_level": "HIGH" if abs(injection_risk_score - 0.5) > 0.3 else "MEDIUM" if abs(injection_risk_score - 0.5) > 0.1 else "LOW"
            }
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    uptime = time.time() - startup_time
    
    if model_loading:
        return {"status": "initializing", "message": "Model is still loading", "uptime_seconds": uptime}
    
    if not model_ready:
        return {"status": "error", "message": "Model failed to load", "uptime_seconds": uptime}
    
    return {
        "status": "ok", 
        "uptime_seconds": uptime,
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 