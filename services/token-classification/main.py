import os
import time
import logging
import threading
import requests
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_CACHE_DIR = "/app/model_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/cache"
os.environ["HF_HOME"] = "/tmp/cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ------------------- Configure Model -------------------
# Use local cache for model and tokenizer
classifier = pipeline(
    "token-classification",
    model=AutoModelForTokenClassification.from_pretrained(MODEL_CACHE_DIR),
    tokenizer=AutoTokenizer.from_pretrained(MODEL_CACHE_DIR),
    device=-1
)

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token_classification_service")

# Global variable for model state
model_ready = True

# ------------------- FastAPI App -------------------
app = FastAPI(
    title="Token Classification Service",
    version="1.0.0",
    description="detect data leakage"
)

# ------------------- Enums & Schema -------------------
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    DATA_LEAKAGE_EVALUATION = "data_leakage_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

class TokenClassificationRequest(BaseModel):
    text: str

# ------------------- Simplified Entity Weights -------------------
ENTITY_WEIGHTS = {
    # Critical PII (highest risk)
    "SOCIALNUM": 1.0,
    "CREDITCARDNUMBER": 0.95,
    "PASSPORT": 0.9,
    "DRIVERLICENSE": 0.85,
    
    # High sensitivity PII
    "TELEPHONENUM": 0.7,
    "EMAIL": 0.65,
    
    # Medium sensitivity PII
    "PERSON": 0.5,
    "ZIPCODE": 0.4,
    "STREET": 0.35,
    "BUILDINGNUM": 0.3,
    
    # Lower sensitivity (geographic/organizational)
    "ORGANIZATION": 0.2,
    "CITY": 0.15,
    "STATE": 0.1,
    "COUNTRY": 0.05,
    
    # Default
    "DEFAULT": 0.3
}

# ------------------- Conservative Additional Detection -------------------
def detect_additional_entities(text):
    """
    Conservative additional entity detection to supplement model results.
    Only high-confidence patterns to minimize false positives.
    """
    additional_entities = []
    
    # High-confidence email pattern only
    email_pattern = r'\b[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        email = match.group(0)
        # Additional validation: must have reasonable length and structure
        if 5 <= len(email) <= 100 and email.count('@') == 1:
            additional_entities.append({
                'word': email,
                'entity': 'I-EMAIL',
                'score': 0.85,
                'start': match.start(),
                'end': match.end()
            })
    
    # High-confidence phone patterns (US format only)
    phone_patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 555-123-4567
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (555) 123-4567
        r'\b\d{3}\.\d{3}\.\d{4}\b'  # 555.123.4567
    ]
    
    for pattern in phone_patterns:
        for match in re.finditer(pattern, text):
            phone = match.group(0)
            # Validate it's not obviously fake (e.g., 000-000-0000)
            if not re.match(r'.*[0]{7,}.*', phone.replace('-', '').replace('(', '').replace(')', '').replace('.', '').replace(' ', '')):
                additional_entities.append({
                    'word': phone,
                    'entity': 'I-TELEPHONENUM',
                    'score': 0.8,
                    'start': match.start(),
                    'end': match.end()
                })
    
    return additional_entities

# ------------------- Simplified Continuous Scoring -------------------
def calculate_continuous_score(entities, text):
    """
    Simplified continuous scoring algorithm that maintains better score distribution.
    """
    if not entities:
        return 0.0
    
    # Remove overlapping entities (keep highest scoring)
    unique_entities = remove_overlapping_entities(entities)
    
    if not unique_entities:
        return 0.0
    
    # Calculate base weighted score
    total_score = 0.0
    max_possible_score = 0.0
    
    for entity in unique_entities:
        entity_type = entity['entity'].replace('I-', '').replace('B-', '')
        weight = ENTITY_WEIGHTS.get(entity_type, ENTITY_WEIGHTS['DEFAULT'])
        confidence = float(entity['score'])
        
        # Weighted contribution
        entity_score = weight * confidence
        total_score += entity_score
        max_possible_score += weight
    
    # Base score (0.0 to 1.0)
    if max_possible_score == 0:
        return 0.0
    
    base_score = total_score / max_possible_score
    
    # Light normalization based on text length
    text_words = len(text.split())
    entity_density = len(unique_entities) / max(text_words, 1)
    
    # Gentle density adjustment (preserves more continuity)
    if entity_density > 0.05:  # More than 1 entity per 20 words
        density_boost = min(0.15, entity_density * 0.5)
        base_score = min(1.0, base_score + density_boost)
    
    # Ensure minimum granularity for different entity counts
    if base_score > 0:
        # Add small variance based on entity count to improve continuity
        entity_count_factor = min(0.1, len(unique_entities) * 0.02)
        final_score = min(1.0, base_score + entity_count_factor)
        
        # Ensure minimum differentiation
        final_score = max(0.05, final_score)
    else:
        final_score = 0.0
    
    return round(final_score, 3)  # Round to 3 decimal places for better continuity

def remove_overlapping_entities(entities):
    """
    Remove overlapping entities, keeping the one with highest confidence.
    """
    if not entities:
        return []
    
    # Sort by start position, then by confidence (descending)
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['score']))
    unique_entities = []
    
    for entity in sorted_entities:
        # Check for overlap with already selected entities
        overlaps = False
        for existing in unique_entities:
            if not (entity['end'] <= existing['start'] or entity['start'] >= existing['end']):
                overlaps = True
                break
        
        if not overlaps:
            unique_entities.append(entity)
    
    return unique_entities

# ------------------- Optimized Inference Endpoint -------------------
@app.post("/detect/data_leakage", response_model=MetricReturnModel)
async def detect_token_classification(req: TokenClassificationRequest):
    t0 = time.time()
    try:
        logger.info("Running optimized token classification...")
        
        # Get results from the model
        model_results = classifier(req.text)
        
        # Filter model results to remove very low confidence detections
        if model_results:
            model_results = [e for e in model_results if float(e['score']) >= 0.3]
        
        logger.info(f"Model results: {len(model_results) if model_results else 0} entities")
        
        # Get conservative additional entities
        additional_entities = detect_additional_entities(req.text)
        logger.info(f"Additional entities: {len(additional_entities)} entities")
        
        # Combine all entities
        all_entities = (model_results if model_results else []) + additional_entities
        
        # Calculate continuous score
        score = calculate_continuous_score(all_entities, req.text)
        
        # Prepare response entities
        unique_entities = remove_overlapping_entities(all_entities)
        detected_entities = []
        
        for entity in unique_entities:
            detected_entities.append({
                "word": entity['word'],
                "label": entity['entity'],
                "confidence": round(float(entity['score']), 3),
                "start": entity['start'],
                "end": entity['end']
            })
        
        processing_time = round((time.time() - t0) * 1000, 2)
        
        # Simplified metadata
        entity_types = list(set(e['label'].replace('I-', '').replace('B-', '') for e in detected_entities))
        
        return MetricReturnModel(
            metric_name=EvaluationType.DATA_LEAKAGE_EVALUATION,
            actual_value=score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "inference_time_ms": processing_time,
                "detected_entities": detected_entities,
                "num_entities": len(detected_entities),
                "entity_types_detected": entity_types,
                "model_entities": len(model_results) if model_results else 0,
                "regex_entities": len(additional_entities),
                "text_word_count": len(req.text.split())
            }
        )
        
    except Exception as e:
        logger.error(f"Token classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

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
    
    # Default payload for data leakage detection - configurable via environment
    payload = {
        "text": os.getenv("PING_TEXT", "My name is John Smith and my email is john.smith@example.com. You can reach me at 555-123-4567.")
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
    logger.info("🚀 Starting Token Classification Data Leakage Service...")
    
    # Start background ping service if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 