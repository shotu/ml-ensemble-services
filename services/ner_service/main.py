import logging
import threading
import requests
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ner_service")

# Global variable for model state
model_ready = False

# Load spaCy model on startup
try:
    nlp = spacy.load("en_core_web_sm")
    model_ready = True
    logger.info("Loaded spaCy model en_core_web_sm")
except OSError:
    logger.error("spaCy model not installed; run:")
    logger.error("    python -m spacy download en_core_web_sm")
    model_ready = False
    raise

app = FastAPI(
    title="NER Extraction Service",
    version="1.0.0",
    description="Extract named entities via spaCy"
)

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
    
    # Default payload for NER extraction - configurable via environment
    payload = {
        "text": os.getenv("PING_TEXT", "John Smith works at Google in New York.")
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
    logger.info("🚀 Starting NER Extraction Service...")
    
    # Start background ping service if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: list[str]

@app.post("/extract/entities", response_model=NERResponse)
async def extract_entities(req: NERRequest):
    """
    Given some text, return the list of named entities found.
    """
    if not req.text:
        raise HTTPException(status_code=400, detail="`text` cannot be empty")
    doc = nlp(req.text)
    ents = [ent.text for ent in doc.ents]
    logger.info("Extracted %d entities", len(ents))
    return NERResponse(entities=ents)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting NER service on 0.0.0.0:8001")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")
