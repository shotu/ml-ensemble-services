import os
import time
import logging
import sys
import threading
import requests
import nltk
import textstat
import numpy as np
from enum import Enum
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, cmudict
from nltk.probability import FreqDist
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from rapidfuzz.fuzz import ratio as fuzzy_ratio
from rouge_score import rouge_scorer

# ------------------- Configure NLTK -------------------
nltk_data_path = "/app/nltk_data"
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Download required NLTK data
required_packages = ['punkt', 'stopwords', 'cmudict', 'wordnet']
for package in required_packages:
    try:
        nltk.download(package, download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        logging.error(f"Failed to download {package}: {str(e)}")

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
cmu_dict = cmudict.dict()

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("nltk_service")

# ------------------- FastAPI App -------------------
app = FastAPI(
    title="NLTK Analysis Service",
    version="1.0.0",
    description="Combined NLTK-based service for text analysis including clarity, diversity, readability, creativity, BLEU score, compression score, cosine similarity, fuzzy score, ROUGE score, and METEOR score metrics."
)

# ------------------- Startup Event -------------------
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("NLTK service starting up...")
        
        # Start background ping thread if enabled
        if os.getenv("ENABLE_PING", "false").lower() == "true":
            threading.Thread(target=background_ping, daemon=True).start()
            logger.info("Background ping service started")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

# ------------------- Background Pinger -------------------
def background_ping():
    """
    Background service to ping the readability endpoint periodically.
    Uses configurable environment variables for flexibility.
    """
    # Get configuration from environment variables
    ping_url = os.getenv("PING_URL")
    ping_interval = int(os.getenv("PING_INTERVAL_SECONDS", "300"))  # Default: 5 minutes
    api_key = os.getenv("PING_API_KEY")  # Optional API key for gateway
    
    if not ping_url:
        logger.warning("PING_URL not configured, skipping ping service")
        return
    
    # Default payload for readability detection
    payload = {
        "text": os.getenv("PING_TEXT", "The quick brown fox jumps over the lazy dog. The fox is quick and brown. The dog is lazy.")
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

# ------------------- Schema -------------------
class TextRequest(BaseModel):
    text: str

class BLEURequest(BaseModel):
    references: List[str]
    predictions: List[str]

class BLEUResponse(BaseModel):
    metric_name: str
    actual_score: float
    actual_value_type: str
    others: dict

class CompressionRequest(BaseModel):
    references: List[str]
    predictions: List[str]

class CompressionResponse(BaseModel):
    metric_name: str
    actual_score: float
    actual_value_type: str
    others: dict

class CosineSimilarityRequest(BaseModel):
    references: List[str]
    predictions: List[str]

class CosineSimilarityResponse(BaseModel):
    metric_name: str
    actual_score: float
    actual_value_type: str
    others: dict

class FuzzyScoreRequest(BaseModel):
    references: List[str]
    predictions: List[str]

class FuzzyScoreResponse(BaseModel):
    metric_name: str
    actual_score: float
    actual_value_type: str
    others: dict

class RougeScoreRequest(BaseModel):
    references: List[str]
    predictions: List[str]

class RougeScoreResponse(BaseModel):
    metric_name: str
    actual_score: float
    actual_value_type: str
    others: dict

class MeteorScoreRequest(BaseModel):
    references: List[str]
    predictions: List[str]

class MeteorScoreResponse(BaseModel):
    metric_name: str
    actual_score: float
    actual_value_type: str
    others: dict

class SentenceTokenizeRequest(BaseModel):
    text: str

class SentenceTokenizeResponse(BaseModel):
    sentences: List[str]
    sentence_count: int
    processing_time_ms: float

# ------------------- Utility Functions -------------------
def type_token_ratio(tokens):
    """Calculate Type-Token Ratio (TTR) for diversity analysis."""
    return len(set(tokens)) / len(tokens) if tokens else 0

def count_syllables(word):
    """Count syllables in a word using CMU dictionary."""
    word = word.lower()
    if word in cmu_dict:
        return max([len([y for y in x if y[-1].isdigit()]) for x in cmu_dict[word]])
    return 1  # fallback

def flesch_reading_ease(text):
    """Calculate Flesch Reading Ease score."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(word) for word in words if word.isalpha())

    if num_sentences == 0 or num_words == 0:
        return 0.0

    ASL = num_words / num_sentences
    ASW = num_syllables / num_words
    return 206.835 - 1.015 * ASL - 84.6 * ASW

def normalize_values(input_val, min_val, max_val):
    """Normalize values to 0-1 range."""
    normalized = (input_val - min_val) / (max_val - min_val)
    return normalized

def evaluate_conciseness_score(text: str) -> float:
    """Calculate conciseness score based on sentence and word length."""
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
            
        avg_sentence_length = sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)
        words = word_tokenize(text)
        if not words:
            return 0.0
            
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Defining conciseness thresholds
        sentence_length_threshold = 15
        word_length_threshold = 6

        # Normalize scores to be in the range of 0 to 1
        sentence_length_score = min(avg_sentence_length / sentence_length_threshold, 1.0)
        word_length_score = min(avg_word_length / word_length_threshold, 1.0)

        # Calculate conciseness score between 0 and 1
        conciseness_score = (sentence_length_score + word_length_score) / 2
        return conciseness_score
    except Exception as e:
        logger.error(f"Error computing conciseness score: {e}")
        return 0.0

def compute_readability_score(text: str) -> float:
    """Calculate readability score using textstat."""
    try:
        readability = textstat.flesch_reading_ease(text)
        return min(max(readability / 100.0, 0), 1.0)
    except Exception as e:
        logger.error(f"Error computing readability score: {e}")
        return 0.0

def redundancy(text: str) -> float:
    """Calculate diversity score using type-token ratio."""
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]
        ttr = type_token_ratio(filtered_tokens)
        return ttr
    except Exception as e:
        logger.error(f"Error computing redundancy score: {e}")
        return 0.0

def get_sentence_embedding(sentence: List[str], model: Word2Vec) -> np.ndarray:
    """Get sentence embedding by averaging word embeddings."""
    # Get the embeddings for words in the sentence that are in the model's vocabulary
    word_embeddings = [model.wv[word] for word in sentence if word in model.wv]
    
    # If the sentence is empty or none of its words are in the model, return a zero vector
    if not word_embeddings:
        return np.zeros(model.vector_size)
    
    # Average the word embeddings to get the sentence embedding
    return np.mean(word_embeddings, axis=0)

# ------------------- Endpoints -------------------
@app.post("/detect/diversity")
async def detect_diversity(req: TextRequest):
    t0 = time.time()
    text = req.text
    
    try:
        logger.info("Tokenizing response...")
        try:
            tokens = word_tokenize(text.lower())
            nltk_loaded = True
        except Exception as e:
            logger.warning(f"NLTK failed: {e}; falling back to split()")
            tokens = text.lower().split()
            nltk_loaded = False

        try:
            if nltk_loaded:
                stop_words = set(stopwords.words("english"))
                filtered_tokens = [t for t in tokens if t not in stop_words and t.isalpha()]
            else:
                basic_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
                filtered_tokens = [t for t in tokens if t not in basic_stopwords]
        except Exception as e:
            logger.warning(f"Stopword filtering failed: {e}, skipping filtering")
            filtered_tokens = tokens

        ttr = type_token_ratio(filtered_tokens)
        
        return {
            "metric_name": "diversity_evaluation",
            "actual_score": float(ttr),
            "token_count": len(tokens),
            "unique_token_count": len(set(tokens)),
            "processing_time": time.time() - t0
        }
    except Exception as e:
        logger.error(f"Diversity analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/readability")
async def detect_readability(req: TextRequest):
    t0 = time.time()
    text = req.text
    
    try:
        logger.info("Evaluating readability...")
        readability_score = compute_readability_score(text)
        
        return {
            "metric_name": "readability_evaluation",
            "actual_score": float(readability_score),
            "processing_time": time.time() - t0
        }
    except Exception as e:
        logger.error(f"Readability analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/clarity")
async def detect_clarity(req: TextRequest):
    t0 = time.time()
    text = req.text
    
    try:
        logger.info("Evaluating clarity score...")
        
        # Calculate individual scores using the same methods as original service
        readability_score = compute_readability_score(text)
        conciseness_score = evaluate_conciseness_score(text)
        diversity_score = redundancy(text)
        
        # Calculate total score
        total_score = (readability_score + conciseness_score + diversity_score) / 3
        
        return {
            "metric_name": "clarity_evaluation",
            "actual_score": float(total_score),
            "token_count": len(word_tokenize(text)),
            "readability_score": readability_score,
            "conciseness_score": conciseness_score,
            "diversity_score": diversity_score,
            "processing_time": time.time() - t0
        }
    except Exception as e:
        logger.error(f"Clarity analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        

@app.post("/detect/creativity-similarity")
async def detect_creativity(req: TextRequest):
    t0 = time.time()
    text = req.text
    
    try:
        logger.info("Calculating linguistic diversity...")
        try:
            tokens = word_tokenize(text.lower())
            nltk_loaded = True
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}, using basic tokenization")
            tokens = text.lower().split()
            nltk_loaded = False
            
        try:
            if nltk_loaded:
                stop_words = set(stopwords.words("english"))
                filtered_tokens = [t for t in tokens if t not in stop_words and t.isalpha()]
            else:
                # Basic stopword filtering as fallback
                basic_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
                filtered_tokens = [t for t in tokens if t not in basic_stopwords]
        except Exception as e:
            logger.warning(f"Stopword filtering failed: {str(e)}, skipping filtering")
            filtered_tokens = tokens
            
        ttr_score = type_token_ratio(filtered_tokens) if filtered_tokens else 0
        
        return {
            "metric_name": "creativity_evaluation",
            "actual_value": float(ttr_score),
            "actual_value_type": "float",
            "others": {
                "token_count": len(tokens),
                "unique_token_count": len(set(tokens)),
                "processing_time": time.time() - t0
            }
        }
    except Exception as e:
        logger.error(f"Creativity analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compute/bleu-score", response_model=BLEUResponse)
async def compute_bleu_score(req: BLEURequest):
    """
    Compute BLEU score between reference and predicted sentences.
    """
    if len(req.references) != len(req.predictions):
        raise HTTPException(status_code=400, detail="Reference and prediction lists must be the same length")

    t0 = time.time()
    
    try:
        logger.info("Computing BLEU scores...")
        scores = [
            sentence_bleu([nltk.word_tokenize(ref)], nltk.word_tokenize(pred))
            for ref, pred in zip(req.references, req.predictions)
        ]
        avg_score = round(sum(scores) / len(scores), 4)

        return BLEUResponse(
            metric_name="bleu_score_evaluation",
            actual_score=avg_score,
            actual_value_type="float",
            others={"processing_time": time.time() - t0}
        )
    except Exception as e:
        logger.error(f"BLEU score computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compute/compression-score", response_model=CompressionResponse)
async def compute_compression_score(req: CompressionRequest):
    """
    Compute compression score between reference and predicted sentences.
    Normalized to [0.0, 1.0) range using ratio normalization.
    0.5 = no compression, <0.5 = compression, >0.5 = expansion.
    """
    if len(req.references) != len(req.predictions):
        raise HTTPException(status_code=400, detail="Reference and prediction lists must be the same length")

    t0 = time.time()
    
    try:
        logger.info("Computing compression scores...")
        scores = [
            len(pred) / len(ref) if len(ref) > 0 else 0.0
            for ref, pred in zip(req.references, req.predictions)
        ]
        
        # Apply ratio normalization: score / (score + 1)
        normalized_scores = [
            score / (score + 1) for score in scores
        ]
        
        avg_score = round(sum(normalized_scores) / len(normalized_scores), 4)

        return CompressionResponse(
            metric_name="compression_score_evaluation",
            actual_score=avg_score,
            actual_value_type="float",
            others={"processing_time": time.time() - t0}
        )
    except Exception as e:
        logger.error(f"Compression score computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compute/cosine-similarity", response_model=CosineSimilarityResponse)
async def compute_cosine_similarity(req: CosineSimilarityRequest):
    """
    Compute cosine similarity score between reference and predicted sentences.
    """
    if len(req.references) != len(req.predictions):
        raise HTTPException(status_code=400, detail="Reference and prediction lists must be the same length")

    t0 = time.time()
    
    try:
        logger.info("Computing cosine similarity scores...")
        
        # Prepare sentences for Word2Vec training
        all_sentences = []
        for ref, pred in zip(req.references, req.predictions):
            # Tokenize sentences
            ref_tokens = word_tokenize(ref.lower())
            pred_tokens = word_tokenize(pred.lower())
            all_sentences.extend([ref_tokens, pred_tokens])
        
        # Train Word2Vec model
        model = Word2Vec(all_sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        # Calculate cosine similarity for each pair
        scores = []
        for ref, pred in zip(req.references, req.predictions):
            ref_tokens = word_tokenize(ref.lower())
            pred_tokens = word_tokenize(pred.lower())
            
            ref_embedding = get_sentence_embedding(ref_tokens, model)
            pred_embedding = get_sentence_embedding(pred_tokens, model)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(ref_embedding.reshape(1, -1), pred_embedding.reshape(1, -1))[0][0]
            scores.append(similarity)
        
        avg_score = round(np.mean(scores), 4)

        return CosineSimilarityResponse(
            metric_name="cosine_similarity_evaluation",
            actual_score=avg_score,
            actual_value_type="float",
            others={"processing_time": time.time() - t0}
        )
    except Exception as e:
        logger.error(f"Cosine similarity computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compute/fuzzy-score", response_model=FuzzyScoreResponse)
async def compute_fuzzy_score(request: FuzzyScoreRequest):
    """Compute fuzzy score between references and predictions."""
    t0 = time.time()
    
    scores = []
    for ref, pred in zip(request.references, request.predictions):
        score = fuzzy_ratio(ref, pred) / 100.0  # Convert to 0-1 range
        scores.append(score)
    
    avg_score = sum(scores) / len(scores)
    
    return FuzzyScoreResponse(
        metric_name="fuzzy_score_evaluation",
        actual_score=avg_score,
        actual_value_type="float",
        others={"processing_time": time.time() - t0}
    )

@app.post("/compute/rouge-score", response_model=RougeScoreResponse)
async def compute_rouge_score(request: RougeScoreRequest):
    """Compute ROUGE score between references and predictions."""
    t0 = time.time()
    
    scorer = rouge_scorer.RougeScorer(rouge_types=['rougeL'], use_stemmer=True)
    rouge_scores = [
        scorer.score(ref, pred) for ref, pred in zip(request.references, request.predictions)
    ]
    
    # Calculate average F-measure for ROUGE-L
    fmeasures = [score['rougeL'].fmeasure for score in rouge_scores]
    avg_fmeasure = sum(fmeasures) / len(fmeasures)
    
    # Calculate average precision and recall for additional info
    precisions = [score['rougeL'].precision for score in rouge_scores]
    recalls = [score['rougeL'].recall for score in rouge_scores]
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    
    return RougeScoreResponse(
        metric_name="rouge_score_evaluation",
        actual_score=avg_fmeasure,
        actual_value_type="float",
        others={
            "precision": avg_precision,
            "recall": avg_recall,
            "fmeasure": avg_fmeasure,
            "processing_time": time.time() - t0
        }
    )

@app.post("/compute/meteor-score", response_model=MeteorScoreResponse)
async def compute_meteor_score(request: MeteorScoreRequest):
    """Compute METEOR score between references and predictions."""
    if len(request.references) != len(request.predictions):
        raise HTTPException(status_code=400, detail="Reference and prediction lists must be the same length")
        
    t0 = time.time()
    
    try:
        logger.info("Computing METEOR scores...")
        scores = []
        for ref, pred in zip(request.references, request.predictions):
            try:
                # METEOR score expects tokenized inputs
                ref_tokens = word_tokenize(ref)
                pred_tokens = word_tokenize(pred)
                score = meteor_score([ref_tokens], pred_tokens)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error computing METEOR for: '{ref}' vs '{pred}' -> {e}")
                scores.append(0.0)
        
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        
        return MeteorScoreResponse(
            metric_name="meteor_score_evaluation",
            actual_score=avg_score,
            actual_value_type="float",
            others={"processing_time": time.time() - t0}
        )
    except Exception as e:
        logger.error(f"METEOR score computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenize/sentences", response_model=SentenceTokenizeResponse)
async def tokenize_sentences(request: SentenceTokenizeRequest):
    """Tokenize text into sentences using NLTK's sent_tokenize."""
    t0 = time.time()
    
    try:
        logger.info("Tokenizing text into sentences...")
        sentences = sent_tokenize(request.text)
        processing_time = (time.time() - t0) * 1000  # Convert to milliseconds
        
        return SentenceTokenizeResponse(
            sentences=sentences,
            sentence_count=len(sentences),
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Sentence tokenization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 