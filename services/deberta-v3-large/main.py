import os
import time
import logging
import sys
import threading
import requests
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers.cross_encoder import CrossEncoder

import numpy as np

# ------------------- Constants -------------------
MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
CACHE_DIR = "/app/model_cache"

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
logger = logging.getLogger("deberta_v3_large_service")

# ------------------- FastAPI App -------------------
app = FastAPI(
    title="DeBERTa v3 Large Service",
    version="1.0.0",
    description="Text classification using DeBERTa v3 large cross-encoder model."
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
    DEBERTA_V3_LARGE_EVALUATION = "deberta_v3_large_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

class DebertaRequest(BaseModel):
    input_text: str
    output_text: str

# ------------------- Enhanced Score Normalization Function -------------------
def normalize_score(raw_score, input_text="", output_text=""):
    """
    Enhanced normalization with calibration for better score distribution.
    
    Args:
        raw_score: Raw logit score from the model
        input_text: Input text for context-aware adjustments
        output_text: Output text for context-aware adjustments
        
    Returns:
        float: Calibrated score between 0 and 1
    """
    try:
        # Step 1: Apply sigmoid normalization
        sigmoid_score = 1.0 / (1.0 + np.exp(-raw_score))
        
        # Step 2: Apply calibration transformation to better use the 0-1 range
        # This stretches the distribution to use more of the lower range
        
        # Parameters for calibration (empirically determined)
        # These values help map the compressed 0.7-1.0 range to a better 0.0-1.0 distribution
        baseline_threshold = 0.65  # Scores below this map to lower range
        scale_factor = 1.5  # Amplifies differences
        
        # Apply piecewise transformation
        if sigmoid_score < baseline_threshold:
            # For very low scores, map to 0.0-0.3 range
            calibrated_score = (sigmoid_score / baseline_threshold) * 0.3
        else:
            # For higher scores, apply stretching transformation
            normalized_high = (sigmoid_score - baseline_threshold) / (1.0 - baseline_threshold)
            # Apply power transformation to stretch the distribution
            stretched = np.power(normalized_high, 1/scale_factor)
            calibrated_score = 0.3 + (stretched * 0.7)
        
        # Step 3: Context-aware adjustments
        input_len = len(input_text.strip()) if input_text else 0
        output_len = len(output_text.strip()) if output_text else 0
        
        # Penalty for empty or very short outputs
        if output_len == 0:
            calibrated_score *= 0.1  # Severe penalty for empty output
        elif output_len < 10:
            calibrated_score *= 0.3  # Moderate penalty for very short output
        elif output_len < 20:
            calibrated_score *= 0.6  # Light penalty for short output
        
        # Bonus for reasonable length matching
        if input_len > 0 and output_len > 0:
            length_ratio = min(output_len / input_len, input_len / output_len)
            if length_ratio > 0.1:  # Reasonable length relationship
                length_bonus = min(0.1, length_ratio * 0.05)
                calibrated_score = min(1.0, calibrated_score + length_bonus)
        
        # Step 4: Final bounds checking and smoothing
        calibrated_score = np.clip(calibrated_score, 0.0, 1.0)
        
        # Apply final smoothing to avoid extreme values
        if calibrated_score > 0.95:
            calibrated_score = 0.95 + (calibrated_score - 0.95) * 0.5
        
        return float(calibrated_score)
        
    except (OverflowError, RuntimeWarning, ZeroDivisionError):
        # Handle extreme values and edge cases
        if raw_score > 20:  # Very large positive values
            return 0.9  # High but not maximum
        elif raw_score < -20:  # Very large negative values
            return 0.1  # Low but not minimum
        else:
            return 0.5  # Fallback for any other edge cases

# ------------------- Model Loading -------------------
def load_model_in_background():
    global model, model_loading, model_ready
    model_loading = True
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        
        model = CrossEncoder(MODEL_NAME)
        logger.info("Model loaded successfully")
        
        # Test the model
        model.predict([("test input", "test output")])
        
        model_ready = True
        logger.info("Model initialization complete")
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
    finally:
        model_loading = False

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application")
    
    thread = threading.Thread(target=load_model_in_background)
    thread.daemon = True
    thread.start()
    logger.info("Model loading started in background thread")
    
    # Start background ping thread if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

# ------------------- Background Pinger -------------------
def background_ping():
    """
    Background service to ping the answer relevance endpoint periodically.
    Uses configurable environment variables for flexibility.
    """
    # Get configuration from environment variables
    ping_url = os.getenv("PING_URL")
    ping_interval = int(os.getenv("PING_INTERVAL_SECONDS", "300"))  # Default: 5 minutes
    api_key = os.getenv("PING_API_KEY")  # Optional API key for gateway
    
    if not ping_url:
        logger.warning("PING_URL not configured, skipping ping service")
        return
    
    # Default payload for answer relevance detection
    payload = {
        "input_text": os.getenv("PING_INPUT_TEXT", "What is the capital of France?"),
        "output_text": os.getenv("PING_OUTPUT_TEXT", "The capital of France is Paris, a beautiful city known for its culture and history.")
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
@app.post("/detect/answer_relevance", response_model=MetricReturnModel)
async def evaluate_text(req: DebertaRequest):
    t0 = time.time()
    input_text = req.input_text
    output_text = req.output_text

    try:
        # Get raw scores from the model
        raw_scores = model.predict([(input_text, output_text)], convert_to_numpy=True)
        
        # Extract raw score from model output
        if isinstance(raw_scores, np.ndarray) and len(raw_scores) > 0:
            if len(raw_scores.shape) == 1 and raw_scores.shape[0] == 1:
                raw_score = float(raw_scores[0])
            elif len(raw_scores.shape) > 1 and raw_scores.shape[1] > 1:
                raw_score = float(np.max(raw_scores[0]))
            else:
                raw_score = float(raw_scores[0])
        else:
            raw_score = 0.0  # Fallback for empty results
        
        # Normalize the raw score to 0-1 range
        normalized_score = normalize_score(raw_score, input_text, output_text)
        
        logger.info(f"Raw score: {raw_score:.3f}, Calibrated score: {normalized_score:.3f}")

        return MetricReturnModel(
            metric_name=EvaluationType.DEBERTA_V3_LARGE_EVALUATION,
            actual_value=normalized_score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
                "input_text_length": len(input_text),
                "output_text_length": len(output_text),
                "model_name": MODEL_NAME,
                "raw_score": round(raw_score, 3),
                "calibrated_score": round(normalized_score, 3)
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
    
    return {"status": "ok", "uptime_seconds": uptime}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 