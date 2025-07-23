import time, logging
import threading
import requests
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
from enum import Enum

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("similarity_service")

app = FastAPI(
    title="Text Similarity Service",
    version="1.0.0",
    description="Compute pairwise cosine similarities via SBERT and agentic metrics evaluation"
)

# Global model variable
model = None
model_ready = False

# ------------------- Enums & Response Schema -------------------
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    TEXT_SIMILARITY_EVALUATION = "text_similarity_evaluation"
    # Agentic Metrics
    AGENT_GOAL_ACCURACY_EVALUATION = "agent_goal_accuracy_evaluation"
    INTENT_RESOLUTION_EVALUATION = "intent_resolution_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

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
    
    # Default payload for text similarity - configurable via environment
    payload = {
        "sources": ["Machine learning models require careful training"],
        "targets": ["ML models need proper training"]
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
    global model, model_ready
    try:
        logger.info("Loading model...")
        model = SentenceTransformer("/app/model_cache")
        model_ready = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_ready = False
        raise
    
    # Start background ping service if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok"}

class SimilarityRequest(BaseModel):
    sources: list[str]
    targets: list[str]
    model_name: str = "all-MiniLM-L6-v2"
    params: dict[str, float] = {}

class PairSimilarity(BaseModel):
    source: str
    target: str
    similarity: float

class SimilarityResponse(BaseModel):
    metric_name: str
    similarities: list[list[float]]
    pairs: list[PairSimilarity]
    diagnostics: dict
    model_info: dict

# Agentic Metrics Request Schema
class AgenticMetricsRequest(BaseModel):
    conversation_history: List[str]
    tool_calls: List[Dict[str, Any]]
    agent_responses: List[str]
    reference_data: Dict[str, Any]

@app.post(
    "/compute/text-similarity",
    response_model=SimilarityResponse,
    summary="Compute text similarity",
    tags=["similarity"]
)
async def compute_text_similarity(req: SimilarityRequest):
    """
    Computes cosine similarities between each source/target pair.
    """
    if model is None:
        raise RuntimeError("Model not loaded")
        
    t0 = time.time()
    emb_s = model.encode(req.sources, convert_to_tensor=True)
    emb_t = model.encode(req.targets, convert_to_tensor=True)
    sims = util.cos_sim(emb_s, emb_t).tolist()

    pairs = [
        PairSimilarity(source=src, target=tgt, similarity=sims[i][j])
        for i, src in enumerate(req.sources)
        for j, tgt in enumerate(req.targets)
    ]

    model_info = {
        "model_name":  req.model_name,
        "model_version":"1.0.0",
        "compute_time_ms": (time.time() - t0) * 1000
    }

    return SimilarityResponse(
        metric_name="text_similarity",
        similarities=sims,
        pairs=pairs,
        diagnostics={"num_sources": len(req.sources), "num_targets": len(req.targets)},
        model_info=model_info
    )

# ------------------- Agentic Metrics Implementation -------------------
class AgenticMetricsEvaluator:
    """Agentic metrics evaluator using paraphrase-MiniLM-L6-v2"""
    
    def __init__(self, model):
        self.model = model
    
    def _normalize_and_clamp_score(self, raw_similarity: float) -> float:
        """
        Normalize and clamp cosine similarity to [0, 1] range with better utilization.
        
        Args:
            raw_similarity: Raw cosine similarity score (can be negative)
            
        Returns:
            Normalized score between 0 and 1
        """
        # Clamp to [0, 1] range first
        clamped = max(0.0, min(1.0, raw_similarity))
        
        # Apply sigmoid-like normalization to better utilize the range
        # This transforms the typical 0.4-1.0 range to better use 0.0-1.0
        if clamped < 0.4:
            # Very low similarity - map to 0.0-0.3 range
            normalized = (clamped / 0.4) * 0.3
        else:
            # Above 0.4 - map to 0.3-1.0 range with sigmoid-like curve
            normalized = 0.3 + (0.7 * (clamped - 0.4) / 0.6)
            # Apply slight curve for better differentiation
            normalized = min(1.0, normalized * 1.1)
        
        return float(normalized)
    
    def _get_score_interpretation(self, score: float) -> str:
        """
        Get human-readable interpretation of the score.
        
        Args:
            score: Normalized score between 0 and 1
            
        Returns:
            Interpretation string
        """
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        elif score >= 0.2:
            return "poor"
        else:
            return "very_poor"
    
    def evaluate_agent_goal_accuracy(self, final_agent_response: str, target_goal: str) -> Dict[str, Any]:
        """
        Evaluate if the agent accomplishes the user's stated or implied goal.
        
        Args:
            final_agent_response: Agent's final output/summary
            target_goal: Expected goal/outcome description
            
        Returns:
            Dict containing similarity score and goal achievement assessment
        """
        start_time = time.time()
        
        try:
            if not final_agent_response.strip() or not target_goal.strip():
                return {
                    "goal_accuracy_score": 0.0,
                    "raw_similarity": 0.0,
                    "similarity": 0.0,
                    "score_interpretation": "very_poor",
                    "processing_time": time.time() - start_time
                }
            
            # Compute semantic similarity using paraphrase-MiniLM-L6-v2
            final_embedding = self.model.encode([final_agent_response], convert_to_tensor=True)
            goal_embedding = self.model.encode([target_goal], convert_to_tensor=True)
            
            raw_similarity = float(util.cos_sim(final_embedding, goal_embedding)[0][0])
            
            # Normalize and clamp the score
            normalized_score = self._normalize_and_clamp_score(raw_similarity)
            score_interpretation = self._get_score_interpretation(normalized_score)
            
            processing_time = time.time() - start_time
            
            return {
                "goal_accuracy_score": normalized_score,
                "raw_similarity": raw_similarity,
                "similarity": normalized_score,
                "score_interpretation": score_interpretation,
                "final_response_length": len(final_agent_response),
                "target_goal_length": len(target_goal),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in agent goal accuracy evaluation: {str(e)}")
            return {
                "goal_accuracy_score": 0.0,
                "raw_similarity": 0.0,
                "similarity": 0.0,
                "score_interpretation": "very_poor",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def evaluate_intent_resolution(self, first_agent_response: str, reference_intent: str) -> Dict[str, Any]:
        """
        Evaluate if the agent correctly identified and acted on user intent.
        
        Args:
            first_agent_response: Agent's initial response showing intent understanding
            reference_intent: Expected intent understanding
            
        Returns:
            Dict containing intent alignment score
        """
        start_time = time.time()
        
        try:
            if not first_agent_response.strip() or not reference_intent.strip():
                return {
                    "intent_resolution_score": 0.0,
                    "raw_similarity": 0.0,
                    "similarity": 0.0,
                    "score_interpretation": "very_poor",
                    "processing_time": time.time() - start_time
                }
            
            # Compute semantic similarity using paraphrase-MiniLM-L6-v2
            response_embedding = self.model.encode([first_agent_response], convert_to_tensor=True)
            intent_embedding = self.model.encode([reference_intent], convert_to_tensor=True)
            
            raw_similarity = float(util.cos_sim(response_embedding, intent_embedding)[0][0])
            
            # Normalize and clamp the score
            normalized_score = self._normalize_and_clamp_score(raw_similarity)
            score_interpretation = self._get_score_interpretation(normalized_score)
            
            processing_time = time.time() - start_time
            
            return {
                "intent_resolution_score": normalized_score,
                "raw_similarity": raw_similarity,
                "similarity": normalized_score,
                "score_interpretation": score_interpretation,
                "first_response_length": len(first_agent_response),
                "reference_intent_length": len(reference_intent),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in intent resolution evaluation: {str(e)}")
            return {
                "intent_resolution_score": 0.0,
                "raw_similarity": 0.0,
                "similarity": 0.0,
                "score_interpretation": "very_poor",
                "error": str(e),
                "processing_time": time.time() - start_time
            }

# ------------------- Agentic Metrics Endpoints -------------------

@app.post("/evaluate/agent-goal-accuracy", response_model=MetricReturnModel)
async def evaluate_agent_goal_accuracy_endpoint(req: AgenticMetricsRequest):
    """
    Evaluate if the agent accomplishes the user's stated or implied goal.
    
    This metric measures how well the agent's final response aligns with the target goal
    using semantic similarity analysis with paraphrase-MiniLM-L6-v2.
    
    Score Interpretation:
    - 0.8-1.0: Excellent goal alignment
    - 0.6-0.8: Good goal alignment  
    - 0.4-0.6: Fair goal alignment
    - 0.2-0.4: Poor goal alignment
    - 0.0-0.2: Very poor goal alignment
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.agent_responses:
            raise HTTPException(status_code=400, detail="Agent responses cannot be empty")
        if not req.reference_data.get("target_goal"):
            raise HTTPException(status_code=400, detail="Target goal must be provided in reference_data")
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Initialize evaluator
        evaluator = AgenticMetricsEvaluator(model)
        
        # Get final agent response
        final_response = req.agent_responses[-1] if req.agent_responses else ""
        
        # Perform evaluation
        result = evaluator.evaluate_agent_goal_accuracy(
            final_agent_response=final_response,
            target_goal=req.reference_data["target_goal"]
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.AGENT_GOAL_ACCURACY_EVALUATION,
            "actual_value": result["goal_accuracy_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "raw_similarity": result.get("raw_similarity", 0.0),
                "similarity": result.get("similarity", 0.0),
                "score_interpretation": result.get("score_interpretation", "very_poor"),
                "final_response_length": result.get("final_response_length", 0),
                "target_goal_length": result.get("target_goal_length", 0),
                "input_lengths": {
                    "agent_responses_count": len(req.agent_responses),
                    "final_response_length": len(final_response),
                    "target_goal_length": len(req.reference_data.get("target_goal", ""))
                },
                "model_name": "paraphrase-MiniLM-L6-v2",
                "evaluation_method": "semantic_goal_alignment_with_normalization"
            }
        }
        
        logger.info(f"Agent goal accuracy evaluation completed in {processing_time:.4f}s - Score: {result['goal_accuracy_score']:.3f} ({result['score_interpretation']})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agent goal accuracy evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/evaluate/intent-resolution", response_model=MetricReturnModel)
async def evaluate_intent_resolution_endpoint(req: AgenticMetricsRequest):
    """
    Evaluate if the agent correctly identified and acted on user intent.
    
    This metric measures how well the agent's initial response demonstrates understanding
    of the user's intent using semantic similarity analysis with paraphrase-MiniLM-L6-v2.
    
    Score Interpretation:
    - 0.8-1.0: Excellent intent understanding
    - 0.6-0.8: Good intent understanding
    - 0.4-0.6: Fair intent understanding
    - 0.2-0.4: Poor intent understanding
    - 0.0-0.2: Very poor intent understanding
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.agent_responses:
            raise HTTPException(status_code=400, detail="Agent responses cannot be empty")
        if not req.reference_data.get("reference_intent"):
            raise HTTPException(status_code=400, detail="Reference intent must be provided in reference_data")
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Initialize evaluator
        evaluator = AgenticMetricsEvaluator(model)
        
        # Get first agent response
        first_response = req.agent_responses[0] if req.agent_responses else ""
        
        # Perform evaluation
        result = evaluator.evaluate_intent_resolution(
            first_agent_response=first_response,
            reference_intent=req.reference_data["reference_intent"]
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.INTENT_RESOLUTION_EVALUATION,
            "actual_value": result["intent_resolution_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "raw_similarity": result.get("raw_similarity", 0.0),
                "similarity": result.get("similarity", 0.0),
                "score_interpretation": result.get("score_interpretation", "very_poor"),
                "first_response_length": result.get("first_response_length", 0),
                "reference_intent_length": result.get("reference_intent_length", 0),
                "input_lengths": {
                    "agent_responses_count": len(req.agent_responses),
                    "first_response_length": len(first_response),
                    "reference_intent_length": len(req.reference_data.get("reference_intent", ""))
                },
                "model_name": "paraphrase-MiniLM-L6-v2",
                "evaluation_method": "semantic_intent_alignment_with_normalization"
            }
        }
        
        logger.info(f"Intent resolution evaluation completed in {processing_time:.4f}s - Score: {result['intent_resolution_score']:.3f} ({result['score_interpretation']})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in intent resolution evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable, default to 8080
    port = int(os.getenv("PORT", "8080"))
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port) 