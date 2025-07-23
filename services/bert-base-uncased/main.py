import os
import time
import logging
import threading
import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from enum import Enum

# ------------------- Logging Configuration -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bert_base_uncased_service")

# ------------------- FastAPI App Definition -------------------
app = FastAPI(
    title="BERT Base Uncased Service",
    version="1.0.0",
    description="Evaluates text similarity and relevance using BERT Base Uncased model."
)

# ------------------- Enums & Response Schema -------------------
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    CREATIVITY_EVALUATION = "creativity_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    relevance_score: float
    novelty_score: float
    others: dict = {}

# ------------------- Request Schema -------------------
class CreativityRequest(BaseModel):
    context: str
    response: str

# ------------------- Model & Pipeline Load -------------------
tokenizer = None
model = None
classifier = None
bertscore = None

@app.on_event("startup")
async def startup_event():
    global tokenizer, model, classifier, bertscore
    try:
        logger.info("Loading BERT model and tokenizer...")
        model_dir = "/app/model_cache"
        
        # Load BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        
        # Create a text classification pipeline
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        logger.info("Model loaded successfully!")
        
        # Try to load BERTScore for semantic evaluation
        logger.info("Loading BERTScore...")
        try:
            import evaluate
            bertscore = evaluate.load("bertscore")
            logger.info("BERTScore loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load BERTScore: {str(e)}. Will use alternative scoring method.")
            bertscore = None
            
        # Start background ping thread if enabled
        if os.getenv("ENABLE_PING", "false").lower() == "true":
            threading.Thread(target=background_ping, daemon=True).start()
            logger.info("Background ping service started")
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        # Continue anyway to let the service start

# ------------------- Health Check -------------------
@app.get("/health")
def health_check():
    if classifier is None:
        return {"status": "warning", "message": "Model only partially loaded"}
    return {"status": "ok"}

# ------------------- Background Pinger -------------------
def background_ping():
    """
    Background service to ping the creativity endpoint periodically.
    Uses configurable environment variables for flexibility.
    """
    # Get configuration from environment variables
    ping_url = os.getenv("PING_URL")
    ping_interval = int(os.getenv("PING_INTERVAL_SECONDS", "300"))  # Default: 5 minutes
    api_key = os.getenv("PING_API_KEY")  # Optional API key for gateway
    
    if not ping_url:
        logger.warning("PING_URL not configured, skipping ping service")
        return
    
    # Default payload for creativity detection
    payload = {
        "context": os.getenv("PING_CONTEXT", "Write a story about artificial intelligence"),
        "response": os.getenv("PING_RESPONSE", "AI robots became sentient and decided to create art instead of destroying humanity")
    }
    
    # Set up headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    logger.info(f"Starting ping service: URL={ping_url}, interval={ping_interval}s")
    
    while True:
        try:
            if classifier is not None:  # Only ping when model is ready
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

# ------------------- Utility Functions -------------------
def compute_similarity(text1, text2):
    """Compute similarity between two texts using BERT embeddings"""
    global bertscore, tokenizer, model
    
    if bertscore:
        try:
            # Use BERTScore if available
            results = bertscore.compute(
                predictions=[text1], 
                references=[text2], 
                lang="en"
            )
            return results['f1'][0]
        except Exception as e:
            logger.warning(f"BERTScore failed: {str(e)}, falling back to alternative method")
    
    # Fallback to direct cosine similarity with BERT
    try:
        # Encode both texts
        with torch.no_grad():
            inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
            inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
            
            embeddings1 = model.bert(**inputs1).pooler_output
            embeddings2 = model.bert(**inputs2).pooler_output
            
            # Normalize
            embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
            embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.matmul(embeddings1, embeddings2.T).item()
            return max(0.0, min(1.0, similarity))  # Clip to 0-1 range
    except Exception as e:
        logger.error(f"Similarity computation failed: {str(e)}")
        return 0.5  # Return a default middle value

# ------------------- Inference Endpoint -------------------
@app.post("/detect/creativity", response_model=MetricReturnModel)
async def compute_creativity(req: CreativityRequest):
    t0 = time.time()
    
    # Extract input data
    context = req.context
    response = req.response
    
    try:
        # 1. Semantic Relevance (similarity between response and context)
        logger.info("Calculating semantic relevance...")
        similarity_score = compute_similarity(response, context)
        relevance_score = similarity_score
        
        # 2. Novelty/Innovation (using semantic difference from context)
        logger.info("Calculating novelty...")
        # High novelty means low similarity to the context (within reason)
        # Scale from 0-1 where novelty is the inverse of similarity (with adjustment)
        novelty_score = max(min(1.0 - similarity_score * 0.6, 1.0), 0.0)
        
        # Combined creativity score (using only relevance and novelty)
        w1, w2 = 0.5, 0.5  # Equal weights for relevance and novelty
        creativity_score = w1 * relevance_score + w2 * novelty_score
        
        # Prepare response
        result = {
            "metric_name": EvaluationType.CREATIVITY_EVALUATION,
            "actual_value": float(creativity_score),
            "actual_value_type": ActualValueDtype.FLOAT,
            "relevance_score": float(relevance_score),
            "novelty_score": float(novelty_score),
            "others": {
                "processing_time": time.time() - t0
            }
        }
        
        logger.info(f"Creativity evaluation completed in {time.time() - t0:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error in creativity evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- App Entry Point -------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
