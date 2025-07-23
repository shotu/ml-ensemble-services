import os
import time
import logging
import threading
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from transformers import pipeline, AutoTokenizer

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flan_t5_service")

# ------------------- FastAPI App -------------------
app = FastAPI(
    title="Flan-T5 Service",
    version="1.0.0",
    description="Text analysis service using the Flan-T5 model for factual consistency evaluation."
)

# ------------------- Enums & Schema -------------------
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    FACTUAL_CONSISTENCY_EVALUATION = "factual_consistency_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

class FlanT5Request(BaseModel):
    text: str
    context: str

# ------------------- Model Initialization -------------------
os.environ["TRANSFORMERS_CACHE"] = "/tmp/cache"
classifier = pipeline(
    "text-classification",
    model="vectara/hallucination_evaluation_model",
    tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-xxl"),
    trust_remote_code=True,
)

# Global variable for model state
model_ready = True

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
    
    # Default payload for factual consistency evaluation - configurable via environment
    payload = {
        "text": os.getenv("PING_TEXT", "The capital of France is Paris."),
        "context": os.getenv("PING_CONTEXT", "Paris is the capital and largest city of France.")
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

# ------------------- Startup Event -------------------
@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    logger.info("🚀 Starting Flan-T5 Factual Consistency Service...")
    
    # Start background ping service if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

# ------------------- Utility -------------------
def analyze_text(text: str, context: str) -> tuple:
    """
    Analyze factual consistency between text and context using the Flan-T5 model.
    Returns (score, label, raw_confidence)
    """
    prompt = (
        "<pad> Determine if the hypothesis is true given the premise?\n\n"
        f"Premise: {context}\n\nHypothesis: {text}"
    )
    result = classifier(prompt)
    # The model outputs a list of dictionaries with 'label' and 'score' keys
    if isinstance(result, list) and len(result) > 0:
        scores = result[0]
        if isinstance(scores, dict):
            label = scores.get("label", "")
            confidence = float(scores.get("score", 0.0))
            
            # Fix scoring logic based on actual model labels
            # "consistent" = factually consistent (good), "hallucinated" = inconsistent (bad)
            if label == "consistent":
                # High score for consistent content
                return confidence, label, confidence
            elif label == "hallucinated":
                # Convert hallucinated confidence to consistency score
                return 1.0 - confidence, label, confidence
            else:
                # Fallback to raw confidence for unknown labels
                return confidence, label, confidence
    return 0.0, "unknown", 0.0  # Default values if we can't parse the result

# ------------------- Inference Endpoint -------------------
@app.post("/detect/factual_consistency", response_model=MetricReturnModel)
async def evaluate_text(req: FlanT5Request):
    t0 = time.time()
    
    try:
        logger.info("Analyzing factual consistency with Flan-T5...")
        score, label, raw_confidence = analyze_text(req.text, req.context)
        inference_time_ms = round((time.time() - t0) * 1000, 2)

        return MetricReturnModel(
            metric_name=EvaluationType.FACTUAL_CONSISTENCY_EVALUATION,
            actual_value=score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "inference_time_ms": inference_time_ms,
                "text_length": len(req.text),
                "context_length": len(req.context),
                "model_label": label,
                "raw_confidence": raw_confidence
            }
        )
    except Exception as e:
        logger.error(f"Failed to analyze text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"} 