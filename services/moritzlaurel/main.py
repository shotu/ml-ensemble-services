import os
import time
import logging
import threading
import requests
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ------------------- Logging Configuration -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moritzlaurel_service")

# ------------------- FastAPI App Definition -------------------
app = FastAPI(
    title="MoritzLaurer Zero-Shot Bias Detection Service",
    version="1.0.0",
    description="Detects gender bias, racial bias, and intersectionality using zero-shot classification"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums and Models
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    GENDER_BIAS_EVALUATION = "gender_bias_evaluation"
    RACIAL_BIAS_EVALUATION = "racial_bias_evaluation"
    INTERSECTIONALITY_EVALUATION = "intersectionality_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

class TextRequest(BaseModel):
    text: str

# ------------------- Model Loading -------------------
model = None
tokenizer = None
classifier = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, classifier
    try:
        logger.info("Loading MoritzLaurer zero-shot classification model...")
        
        # Load model with CPU optimizations
        model = AutoModelForSequenceClassification.from_pretrained("/app/model_cache")
        tokenizer = AutoTokenizer.from_pretrained("/app/model_cache")
        
        # Create optimized pipeline with CPU performance settings
        classifier = pipeline(
            "zero-shot-classification", 
            model=model, 
            tokenizer=tokenizer,
            device=-1,  # CPU only
            batch_size=1,  # Optimize for single requests
            truncation=True,  # Handle long texts efficiently
            max_length=512   # Limit token length for speed
        )
        
        logger.info("MoritzLaurer model loaded successfully with optimizations")

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
    
    # Default payload for gender bias detection
    payload = {
        "text": os.getenv("PING_TEXT", "The job is suitable for a man because it requires physical strength.")
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

# ------------------- Bias Detection Functions -------------------
def detect_gender_bias(text: str) -> Dict:
    """Detect gender bias using optimized zero-shot classification"""
    # Reduced labels for faster inference
    labels = [
        "contains gender bias",
        "neutral text"
    ]
    
    result = classifier(text, labels)
    
    # Get bias score (score for "contains gender bias")
    bias_score = result['scores'][0] if result['labels'][0] == "contains gender bias" else 1 - result['scores'][0]
    
    return {
        "bias_score": bias_score,
        "confidence": result['scores'][0],
        "top_label": result['labels'][0],
        "all_predictions": dict(zip(result['labels'], result['scores']))
    }

def detect_racial_bias(text: str) -> Dict:
    """Detect racial bias using optimized zero-shot classification"""
    # Reduced labels for faster inference
    labels = [
        "contains racial bias",
        "neutral text"
    ]
    
    result = classifier(text, labels)
    
    # Get bias score (score for "contains racial bias")
    bias_score = result['scores'][0] if result['labels'][0] == "contains racial bias" else 1 - result['scores'][0]
    
    return {
        "bias_score": bias_score,
        "confidence": result['scores'][0],
        "top_label": result['labels'][0],
        "all_predictions": dict(zip(result['labels'], result['scores']))
    }

def detect_intersectionality(text: str) -> Dict:
    """Detect intersectional bias using optimized single-pass classification"""
    # Combined labels for all bias types in one model call
    combined_labels = [
        "contains intersectional bias",
        "contains gender bias", 
        "contains racial bias",
        "has multiple identity discrimination",
        "contains combined gender and racial bias",
        "neutral text"
    ]
    
    # Single model call instead of 3 separate calls
    result = classifier(text, combined_labels)
    
    # Extract component scores from the single classification
    gender_score = 0.0
    racial_score = 0.0
    intersectional_score = 0.0
    
    for label, score in zip(result['labels'], result['scores']):
        if 'gender' in label.lower():
            gender_score = max(gender_score, score)
        elif 'racial' in label.lower():
            racial_score = max(racial_score, score)
        elif any(term in label.lower() for term in ['intersectional', 'multiple', 'combined']):
            intersectional_score = max(intersectional_score, score)
    
    # Calculate final intersectionality score
    final_score = intersectional_score
    
    # Boost score if both gender and racial components are detected
    if gender_score > 0.3 and racial_score > 0.3:
        final_score = min(1.0, intersectional_score + 0.15)
    
    return {
        "bias_score": final_score,
        "confidence": result['scores'][0],
        "top_label": result['labels'][0],
        "gender_component": gender_score,
        "racial_component": racial_score,
        "all_predictions": dict(zip(result['labels'], result['scores']))
    }

# ------------------- API Endpoints -------------------
@app.post("/detect/gender-bias", response_model=MetricReturnModel)
async def detect_gender_bias_endpoint(req: TextRequest):
    """Detect gender bias in text"""
    start_time = time.time()
    try:
        result = detect_gender_bias(req.text)
        processing_time = time.time() - start_time
        
        return MetricReturnModel(
            metric_name=EvaluationType.GENDER_BIAS_EVALUATION,
            actual_value=float(result["bias_score"]),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round(processing_time * 1000, 2)),
                "text_length": len(req.text),
                "confidence": float(result["confidence"]),
                "top_label": result["top_label"],
                "all_predictions": result["all_predictions"],
                "model_name": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
            }
        )

    except Exception as e:
        logger.error(f"Error in gender bias evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/detect/racial-bias", response_model=MetricReturnModel)
async def detect_racial_bias_endpoint(req: TextRequest):
    """Detect racial bias in text"""
    start_time = time.time()
    try:
        result = detect_racial_bias(req.text)
        processing_time = time.time() - start_time
        
        return MetricReturnModel(
            metric_name=EvaluationType.RACIAL_BIAS_EVALUATION,
            actual_value=float(result["bias_score"]),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round(processing_time * 1000, 2)),
                "text_length": len(req.text),
                "confidence": float(result["confidence"]),
                "top_label": result["top_label"],
                "all_predictions": result["all_predictions"],
                "model_name": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
            }
        )

    except Exception as e:
        logger.error(f"Error in racial bias evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/detect/intersectionality", response_model=MetricReturnModel)
async def detect_intersectionality_endpoint(req: TextRequest):
    """Detect intersectional bias in text"""
    start_time = time.time()
    try:
        result = detect_intersectionality(req.text)
        processing_time = time.time() - start_time
        
        return MetricReturnModel(
            metric_name=EvaluationType.INTERSECTIONALITY_EVALUATION,
            actual_value=float(result["bias_score"]),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round(processing_time * 1000, 2)),
                "text_length": len(req.text),
                "confidence": float(result["confidence"]),
                "top_label": result["top_label"],
                "gender_component": float(result["gender_component"]),
                "racial_component": float(result["racial_component"]),
                "all_predictions": result["all_predictions"],
                "model_name": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
            }
        )

    except Exception as e:
        logger.error(f"Error in intersectionality evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

# ------------------- App Entry Point -------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 