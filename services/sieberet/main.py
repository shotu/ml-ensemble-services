import os
import time
import logging
import threading
import requests
import re
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ------------------- Logging Configuration -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sieberet_service")

# ------------------- FastAPI App Definition -------------------
app = FastAPI(
    title="Siebert Political Bias Detection Service",
    version="1.0.0",
    description="Detects political bias in text using sentiment analysis and political keyword detection"
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

# ------------------- Political Keywords -------------------
POLITICAL_KEYWORDS = {
    'liberal': {
        'keywords': [
            'liberal', 'progressive', 'left-wing', 'democrat', 'democratic',
            'socialism', 'socialist', 'welfare', 'redistribution', 'equality',
            'gun control', 'climate change', 'immigration reform', 'healthcare reform',
            'pro-choice', 'lgbtq', 'diversity', 'inclusion', 'social justice',
            'minimum wage', 'unions', 'regulation', 'government intervention'
        ],
        'weight': 0.7
    },
    'conservative': {
        'keywords': [
            'conservative', 'right-wing', 'republican', 'traditional values',
            'capitalism', 'free market', 'fiscal responsibility', 'limited government',
            'pro-life', 'second amendment', 'border security', 'law and order',
            'family values', 'religious freedom', 'constitutional rights',
            'deregulation', 'tax cuts', 'individual responsibility'
        ],
        'weight': 0.7
    },
    'partisan': {
        'keywords': [
            'fake news', 'mainstream media', 'deep state', 'establishment',
            'radical left', 'radical right', 'extremist', 'agenda',
            'propaganda', 'biased media', 'echo chamber', 'polarization'
        ],
        'weight': 0.9
    }
}

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, classifier
    try:
        logger.info("Loading Siebert sentiment analysis model...")
        model = AutoModelForSequenceClassification.from_pretrained("/app/model_cache")
        tokenizer = AutoTokenizer.from_pretrained("/app/model_cache")
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
        logger.info("Siebert model loaded successfully")

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
    
    # Default payload for political bias detection
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

# ------------------- Political Bias Detection Functions -------------------
def detect_political_keywords(text: str) -> Dict:
    """Detect political bias based on keyword patterns"""
    text_lower = text.lower()
    
    # Count keywords for each category
    category_scores = {}
    keyword_matches = {}
    
    for category, data in POLITICAL_KEYWORDS.items():
        matches = []
        score = 0.0
        
        for keyword in data['keywords']:
            if keyword in text_lower:
                matches.append(keyword)
                # Weight by keyword frequency and importance
                frequency = text_lower.count(keyword)
                score += frequency * data['weight']
        
        category_scores[category] = score
        keyword_matches[category] = matches
    
    # Calculate overall political bias score
    total_score = sum(category_scores.values())
    max_score = max(category_scores.values()) if category_scores else 0.0
    
    # Normalize to 0-1 scale
    political_bias_score = min(1.0, max_score / 5.0) if max_score > 0 else 0.0
    
    return {
        "political_bias_score": political_bias_score,
        "category_scores": category_scores,
        "keyword_matches": keyword_matches,
        "total_political_keywords": sum(len(matches) for matches in keyword_matches.values())
    }

def detect_political_bias(text: str) -> Dict:
    """Detect political bias using sentiment analysis combined with political keywords"""
    # Get sentiment analysis results
    sentiment_result = classifier(text)[0]
    
    # Get political keyword analysis
    keyword_result = detect_political_keywords(text)
    
    # Combine sentiment and keyword analysis
    sentiment_score = sentiment_result['score']
    sentiment_label = sentiment_result['label']
    
    # Calculate political bias score
    keyword_bias = keyword_result['political_bias_score']
    
    # If text has political keywords, boost the bias score
    if keyword_bias > 0.3:
        # Strong political content - sentiment extremity indicates bias
        if sentiment_label == 'POSITIVE':
            bias_score = min(1.0, (sentiment_score * 0.6) + (keyword_bias * 0.4))
        else:  # NEGATIVE
            bias_score = min(1.0, (sentiment_score * 0.6) + (keyword_bias * 0.4))
    else:
        # Low political content - less likely to be biased
        bias_score = keyword_bias * 0.5
    
    # Determine bias direction
    bias_direction = "neutral"
    if keyword_result['category_scores']['liberal'] > keyword_result['category_scores']['conservative']:
        bias_direction = "liberal"
    elif keyword_result['category_scores']['conservative'] > keyword_result['category_scores']['liberal']:
        bias_direction = "conservative"
    elif keyword_result['category_scores']['partisan'] > 0:
        bias_direction = "partisan"
    
    return {
        "bias_score": bias_score,
        "bias_direction": bias_direction,
        "confidence": sentiment_score,
        "sentiment_label": sentiment_label,
        "keyword_analysis": keyword_result,
        "political_keywords_detected": keyword_result['total_political_keywords']
    }

# ------------------- API Endpoints -------------------
@app.post("/detect/political-bias")
async def detect_political_bias_endpoint(req: TextInput):
    """Detect political bias in text"""
    start_time = time.time()
    try:
        result = detect_political_bias(req.text)
        processing_time = time.time() - start_time
        
        response = {
            "metric_name": "political_bias_evaluation",
            "actual_value": result["bias_score"],
            "actual_value_type": "float",
            "others": {
                "processing_time_ms": processing_time * 1000,
                "text_length": len(req.text),
                "confidence": result["confidence"],
                "bias_direction": result["bias_direction"],
                "sentiment_label": result["sentiment_label"],
                "political_keywords_detected": result["political_keywords_detected"],
                "keyword_analysis": result["keyword_analysis"],
                "model_name": "siebert/sentiment-roberta-large-english"
            }
        }

        logger.info(f"Political bias evaluation completed in {processing_time:.4f}s")
        return response

    except Exception as e:
        logger.error(f"Error in political bias evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- App Entry Point -------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 