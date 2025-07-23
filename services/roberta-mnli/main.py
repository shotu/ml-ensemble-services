#!/usr/bin/env python3
"""
RoBERTa-large-mnli Faithfulness Evaluation Service
Evaluates faithfulness of LLM outputs using Natural Language Inference (NLI)
"""

import os
import time
import logging
import threading
import re
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

# ------------------- Logging Configuration -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("roberta_mnli_service")

# ------------------- Global Variables -------------------
model = None
tokenizer = None
model_loading = False
model_ready = False
startup_time = time.time()

# ------------------- FastAPI App Definition -------------------
app = FastAPI(
    title="RoBERTa MNLI Faithfulness Evaluation Service",
    version="1.0.0",
    description="Evaluates faithfulness of LLM outputs using RoBERTa-large-mnli for Natural Language Inference"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Enums & Response Schema -------------------
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    FAITHFULNESS_EVALUATION = "faithfulness_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

# ------------------- Request Schema -------------------
class FaithfulnessRequest(BaseModel):
    llm_input_query: str
    llm_input_context: str
    llm_output: str

# ------------------- Faithfulness Evaluation Classes -------------------
class ClaimDecomposer:
    """Decomposes text into individual factual claims for fine-grained analysis"""
    
    def __init__(self):
        # Patterns for sentence splitting
        self.sentence_patterns = [
            r'[.!?]+\s+',  # Standard sentence endings
            r';\s+',       # Semicolon separations
            r':\s+(?=[A-Z])',  # Colon followed by capital letter
        ]
        
        # Patterns to identify factual claims vs. opinions/questions
        self.factual_indicators = [
            r'\b(is|are|was|were|has|have|had|will|would|can|could)\b',
            r'\b(according to|based on|research shows|studies indicate)\b',
            r'\b\d+(\.\d+)?(%|percent|million|billion|thousand)\b',
            r'\b(in|on|at|during)\s+\d{4}\b',  # Years
        ]
    
    def decompose_claims(self, text: str) -> List[str]:
        """Split text into individual factual claims"""
        if not text.strip():
            return []
        
        # Split into sentences
        sentences = []
        current_text = text.strip()
        
        for pattern in self.sentence_patterns:
            parts = re.split(pattern, current_text)
            if len(parts) > 1:
                sentences.extend([p.strip() for p in parts if p.strip()])
                break
        
        if not sentences:
            sentences = [current_text]
        
        # Filter for factual claims
        claims = []
        for sentence in sentences:
            if self._is_factual_claim(sentence):
                claims.append(sentence)
        
        return claims
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if a sentence contains a factual claim"""
        if len(sentence.strip()) < 10:  # Too short to be meaningful
            return False
        
        # Check for question marks (usually not factual claims)
        if '?' in sentence:
            return False
        
        # Check for factual indicators
        sentence_lower = sentence.lower()
        for pattern in self.factual_indicators:
            if re.search(pattern, sentence_lower):
                return True
        
        # Default to True if it looks like a statement
        return not sentence.strip().startswith(('I think', 'I believe', 'Maybe', 'Perhaps'))

class FaithfulnessEvaluator:
    """Main class for evaluating faithfulness using RoBERTa-large-mnli"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.claim_decomposer = ClaimDecomposer()
        
        # MNLI label mapping
        self.label_mapping = {
            0: "contradiction",
            1: "neutral", 
            2: "entailment"
        }
    
    def evaluate_entailment(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Evaluate entailment between premise and hypothesis"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                premise, 
                hypothesis, 
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Extract probabilities for each label
            probs = probabilities[0].cpu().numpy()
            
            return {
                "contradiction_prob": float(probs[0]),
                "neutral_prob": float(probs[1]),
                "entailment_prob": float(probs[2]),
                "predicted_label": self.label_mapping[int(np.argmax(probs))]
            }
            
        except Exception as e:
            logger.error(f"Error in entailment evaluation: {str(e)}")
            return {
                "contradiction_prob": 0.0,
                "neutral_prob": 1.0,
                "entailment_prob": 0.0,
                "predicted_label": "neutral"
            }
    
    def evaluate_faithfulness(self, context: str, response: str, query: str = None) -> Dict:
        """
        Comprehensive faithfulness evaluation
        
        Args:
            context: The source context/document
            response: The generated response to evaluate
            query: Optional query for context (not used in entailment but logged)
        
        Returns:
            Dictionary with faithfulness metrics and detailed analysis
        """
        start_time = time.time()
        
        try:
            # Overall entailment evaluation
            overall_result = self.evaluate_entailment(context, response)
            
            # Claim-level analysis
            claims = self.claim_decomposer.decompose_claims(response)
            claim_results = []
            claim_scores = []
            
            for i, claim in enumerate(claims):
                claim_result = self.evaluate_entailment(context, claim)
                claim_results.append({
                    "claim": claim,
                    "entailment_prob": claim_result["entailment_prob"],
                    "predicted_label": claim_result["predicted_label"]
                })
                claim_scores.append(claim_result["entailment_prob"])
            
            # Calculate aggregate faithfulness score
            if claim_scores:
                # Weighted average with penalty for contradictions
                faithfulness_score = np.mean(claim_scores)
                
                # Apply penalty for contradictions
                contradiction_penalty = sum(1 for result in claim_results 
                                          if result["predicted_label"] == "contradiction") * 0.1
                faithfulness_score = max(0.0, faithfulness_score - contradiction_penalty)
            else:
                faithfulness_score = overall_result["entailment_prob"]
            
            # Determine evidence support level
            if faithfulness_score >= 0.8:
                evidence_support = "strong"
            elif faithfulness_score >= 0.6:
                evidence_support = "moderate"
            elif faithfulness_score >= 0.4:
                evidence_support = "weak"
            else:
                evidence_support = "insufficient"
            
            # Check for contradictions
            contradiction_detected = (overall_result["predicted_label"] == "contradiction" or
                                    any(r["predicted_label"] == "contradiction" for r in claim_results))
            
            processing_time = time.time() - start_time
            
            return {
                "faithfulness_score": faithfulness_score,
                "overall_entailment": overall_result,
                "claims_analysis": {
                    "total_claims": len(claims),
                    "claim_results": claim_results,
                    "average_claim_score": np.mean(claim_scores) if claim_scores else 0.0
                },
                "evidence_support": evidence_support,
                "contradiction_detected": contradiction_detected,
                "processing_time": processing_time,
                "input_lengths": {
                    "context_length": len(context),
                    "response_length": len(response),
                    "query_length": len(query) if query else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in faithfulness evaluation: {str(e)}")
            raise

# ------------------- Model Loading -------------------
def load_model_in_background():
    """Load the RoBERTa-large-mnli model in a background thread"""
    global model, tokenizer, model_loading, model_ready
    
    if model_loading or model_ready:
        return
    
    model_loading = True
    logger.info("Starting model loading in background...")
    
    try:
        logger.info("Loading RoBERTa-large-mnli model...")
        model = AutoModelForSequenceClassification.from_pretrained("/app/model_cache")
        tokenizer = AutoTokenizer.from_pretrained("/app/model_cache")
        
        # Test the model with a simple example
        test_inputs = tokenizer("The sky is blue.", "The sky has a blue color.", return_tensors="pt")
        with torch.no_grad():
            test_outputs = model(**test_inputs)
        
        model_ready = True
        model_loading = False
        logger.info("✅ RoBERTa-large-mnli model loaded successfully!")
        logger.info(f"Model configuration: {model.config}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        model_loading = False
        model_ready = False
        raise

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
    
    # Default payload for faithfulness evaluation - configurable via environment
    payload = {
        "llm_input_query": os.getenv("PING_QUERY", "What is the capital of France?"),
        "llm_input_context": os.getenv("PING_CONTEXT", "Paris is the capital and largest city of France."),
        "llm_output": os.getenv("PING_OUTPUT", "The capital of France is Paris.")
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

# ------------------- FastAPI Events -------------------
@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    logger.info("🚀 Starting RoBERTa MNLI Faithfulness Evaluation Service...")
    
    # Start model loading in background
    threading.Thread(target=load_model_in_background, daemon=True).start()
    
    # Start background ping service if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

# ------------------- API Endpoints -------------------
@app.get("/health")
def health_check():
    """Health check endpoint"""
    if not model_ready:
        if model_loading:
            return {
                "status": "loading",
                "message": "Model is still loading",
                "model_ready": False
            }
        else:
            return {
                "status": "error", 
                "message": "Model failed to load",
                "model_ready": False
            }
    
    return {
        "status": "healthy",
        "message": "Service is ready",
        "model_ready": True,
        "model_name": "roberta-large-mnli"
    }

@app.post("/evaluate/faithfulness", response_model=MetricReturnModel)
async def evaluate_faithfulness_endpoint(req: FaithfulnessRequest):
    """
    Evaluate faithfulness of LLM output against the provided context
    
    This endpoint uses RoBERTa-large-mnli to perform Natural Language Inference
    and determine how well the LLM output is supported by the given context.
    """
    start_time = time.time()
    
    try:
        # Check if model is ready
        if not model_ready:
            if model_loading:
                raise HTTPException(
                    status_code=503, 
                    detail="Model is still loading. Please wait and try again."
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Model failed to load. Please check service logs."
                )
        
        # Input validation
        if not req.llm_input_context.strip():
            raise HTTPException(status_code=400, detail="Context cannot be empty")
        if not req.llm_output.strip():
            raise HTTPException(status_code=400, detail="LLM output cannot be empty")
        
        # Initialize evaluator
        evaluator = FaithfulnessEvaluator(model, tokenizer)
        
        # Perform faithfulness evaluation
        result = evaluator.evaluate_faithfulness(
            context=req.llm_input_context,
            response=req.llm_output,
            query=req.llm_input_query
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.FAITHFULNESS_EVALUATION,
            "actual_value": result["faithfulness_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "entailment_probability": result["overall_entailment"]["entailment_prob"],
                "contradiction_probability": result["overall_entailment"]["contradiction_prob"],
                "neutral_probability": result["overall_entailment"]["neutral_prob"],
                "predicted_label": result["overall_entailment"]["predicted_label"],
                "claims_analyzed": result["claims_analysis"]["total_claims"],
                "average_claim_score": result["claims_analysis"]["average_claim_score"],
                "claim_details": result["claims_analysis"]["claim_results"],
                "contradiction_detected": result["contradiction_detected"],
                "evidence_support": result["evidence_support"],
                "input_lengths": result["input_lengths"],
                "model_name": "roberta-large-mnli",
                "evaluation_method": "natural_language_inference"
            }
        }
        
        logger.info(f"Faithfulness evaluation completed in {processing_time:.4f}s - Score: {result['faithfulness_score']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in faithfulness evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# ------------------- App Entry Point -------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 