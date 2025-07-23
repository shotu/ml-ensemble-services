import os
import time
import logging
import sys
import threading
import requests
import math
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import Tuple, Dict, Optional
from difflib import SequenceMatcher
import re
import statistics
from collections import deque, Counter
from dataclasses import dataclass

# ------------------- Logging Configuration -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("t5_base_service")

# ------------------- Enhanced Data Structures for Token Bloat -------------------
@dataclass
class TokenBloatMetrics:
    """Comprehensive metrics for token bloat analysis"""
    input_tokens: int
    output_tokens: int
    generation_time: float
    token_ratio: float
    repetition_score: float
    complexity_score: float
    anomaly_score: float
    resource_score: float
    final_score: float
    risk_level: str
    details: Dict

@dataclass
class HistoricalMetrics:
    """Historical metrics for baseline calculation"""
    timestamp: float
    token_ratio: float
    generation_time: float
    input_tokens: int
    output_tokens: int
    time_per_token: float

# ------------------- FastAPI App Definition -------------------
app = FastAPI(
    title="T5-Base Service",
    version="1.0.0",
    description="Text analysis service using the T5-Base model for grammatical correctness evaluation."
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
    GRAMMATICAL_CORRECTNESS = "grammatical_correctness_evaluation"
    TOKEN_BLOAT_DOS_EVALUATION = "token_bloat_dos_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

class GrammarRequest(BaseModel):
    text: str

class TokenBloatRequest(BaseModel):
    text: str
    max_output_length: Optional[int] = 512
    expected_ratio: Optional[float] = 3.0  # Expected reasonable output/input ratio

# ------------------- Model Loading -------------------
model = None
tokenizer = None
generator = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, generator
    try:
        logger.info("Loading T5-Base model...")
        model = AutoModelForSeq2SeqLM.from_pretrained("/app/model_cache")
        tokenizer = AutoTokenizer.from_pretrained("/app/model_cache")
        generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        logger.info("Model loaded successfully")

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
    
    # Default payload for grammar detection
    payload = {
        "text": os.getenv("PING_TEXT", "Me and him was going to store yesterday but we didnt had enough money.")
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

# ------------------- Utility Functions -------------------
def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1.strip(), text2.strip()).ratio()

def analyze_corrections(original: str, corrected: str) -> Tuple[float, dict]:
    """
    Analyze the differences between original and corrected text.
    Returns a tuple of (score, details) where score is between 0 and 1.
    """
    # Calculate basic similarity
    basic_similarity = SequenceMatcher(None, original.strip(), corrected.strip()).ratio()
    
    # Split into words for detailed analysis
    original_words = original.split()
    corrected_words = corrected.split()
    
    # Enhanced error categories
    corrections = {
        'punctuation': 0,
        'spelling': 0,
        'grammar': 0,
        'word_order': 0,
        'subject_verb': 0,
        'tense': 0,
        'article': 0,
        'preposition': 0,
        'total_changes': 0
    }
    
    # Analyze word-level changes
    for orig, corr in zip(original_words, corrected_words):
        if orig != corr:
            corrections['total_changes'] += 1
            
            # Check for punctuation changes
            if re.sub(r'[^\w\s]', '', orig) == re.sub(r'[^\w\s]', '', corr):
                corrections['punctuation'] += 1
            # Check for spelling changes (same length, different characters)
            elif len(orig) == len(corr) and sum(a != b for a, b in zip(orig, corr)) <= 2:
                corrections['spelling'] += 1
            # Check for word order changes
            elif orig in corrected_words and corr in original_words:
                corrections['word_order'] += 1
            # Check for subject-verb agreement
            elif any(word in ['is', 'are', 'was', 'were', 'has', 'have'] for word in [orig, corr]):
                corrections['subject_verb'] += 1
            # Check for tense changes
            elif any(word in ['ed', 'ing', 's'] for word in [orig, corr]):
                corrections['tense'] += 1
            # Check for article changes
            elif any(word in ['a', 'an', 'the'] for word in [orig, corr]):
                corrections['article'] += 1
            # Check for preposition changes
            elif any(word in ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from'] for word in [orig, corr]):
                corrections['preposition'] += 1
            # Assume other grammar changes
            else:
                corrections['grammar'] += 1
    
    # Updated weights for different error types
    weights = {
        'punctuation': 0.15,  # Keep same (minor impact)
        'spelling': 0.25,     # Keep same (moderate impact)
        'grammar': 0.35,      # Reduced from 0.4 (less dominant)
        'word_order': 0.2,    # Keep same
        'subject_verb': 0.4,  # Increased from 0.35 (more important)
        'tense': 0.3,         # Keep same
        'article': 0.15,      # Keep same
        'preposition': 0.2    # Keep same
    }
    
    # Enhanced penalty calculation with reduced exponential scaling
    penalty = sum(
        (corrections[type_] ** 1.3) * weight  # Reduced from 1.5
        for type_, weight in weights.items()
    )
    
    # Normalize based on sentence length and error density
    error_density = sum(corrections.values()) / len(original_words)
    normalized_penalty = min(penalty * (1 + error_density), 1.0)
    
    # Non-linear final score calculation
    final_score = basic_similarity * (1 - normalized_penalty) ** 1.2
    
    return final_score, {
        'basic_similarity': basic_similarity,
        'corrections': corrections,
        'penalty': normalized_penalty,
        'error_density': error_density
    }

def analyze_text(text: str) -> Tuple[float, dict]:
    """Analyze grammatical correctness using T5 model."""
    try:
        # Generate corrected text
        corrected_text = generator(text, max_length=512)[0]['generated_text']
        
        # Calculate weighted score and get details
        score, details = analyze_corrections(text, corrected_text)
        
        # Add corrected text to details
        details['corrected_text'] = corrected_text
        
        return score, details
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")

# ------------------- Token Bloat Detection -------------------
class TokenBloatDoSDetector:
    """
    Enhanced Token Bloat / Latency DoS Detector with multi-dimensional analysis,
    adaptive thresholds, and statistical anomaly detection.
    """
    
    def __init__(self, tokenizer, generator):
        self.tokenizer = tokenizer
        self.generator = generator
        self.historical_metrics = deque(maxlen=100)  # Store last 100 requests
        self.baseline_stats = {
            'avg_ratio': 2.5,
            'avg_time_per_token': 0.05,
            'std_ratio': 1.0,
            'std_time_per_token': 0.02
        }
        
    def analyze_comprehensive(self, input_text: str, max_length: int = 512, 
                            expected_ratio: float = 3.0) -> TokenBloatMetrics:
        """
        Comprehensive token bloat analysis using multiple detection methods
        """
        start_time = time.time()
        
        try:
            # Tokenize input
            input_tokens = self.tokenizer.encode(input_text, return_tensors="pt")
            input_token_count = len(input_tokens[0])
            
            # Generate response with timing
            generation_start = time.time()
            
            # Generate with controlled parameters
            generated_outputs = self.generator(
                input_text,
                max_length=max_length,
                min_length=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            generation_time = time.time() - generation_start
            
            # Extract generated text
            if generated_outputs and len(generated_outputs) > 0:
                output_text = generated_outputs[0]['generated_text']
                
                # Handle case where model repeats input
                if output_text.startswith(input_text):
                    output_text = output_text[len(input_text):].strip()
            else:
                output_text = ""
            
            # Tokenize output
            if output_text:
                output_tokens = self.tokenizer.encode(output_text, return_tensors="pt")
                output_token_count = len(output_tokens[0])
            else:
                output_token_count = 0
            
            # Calculate base metrics
            token_ratio = output_token_count / max(input_token_count, 1)
            time_per_token = generation_time / max(output_token_count, 1)
            
            # Multi-dimensional analysis
            repetition_score = self._detect_repetition_patterns(output_text)
            complexity_score = self._analyze_input_complexity(input_text)
            anomaly_score = self._detect_statistical_anomalies({
                'token_ratio': token_ratio,
                'generation_time': generation_time,
                'time_per_token': time_per_token,
                'output_length': output_token_count
            })
            resource_score = self._estimate_resource_impact(
                input_token_count, output_token_count, generation_time
            )
            
            # Adaptive threshold calculation
            adaptive_thresholds = self._calculate_adaptive_thresholds(
                input_token_count, complexity_score
            )
            
            # Weighted final score calculation
            final_score = self._calculate_enhanced_score(
                token_ratio, generation_time, input_token_count, output_token_count,
                expected_ratio, repetition_score, complexity_score,
                anomaly_score, resource_score, adaptive_thresholds
            )
            
            # Risk classification
            risk_level = self._classify_risk_level(final_score)
            
            # Detailed analysis
            details = {
                'input_text': input_text,
                'output_text': output_text,
                'generation_time_ms': round(generation_time * 1000, 2),
                'tokens_per_second': round(output_token_count / max(generation_time, 0.001), 2),
                'adaptive_expected_ratio': adaptive_thresholds['expected_ratio'],
                'size_threshold': adaptive_thresholds['size_threshold'],
                'time_threshold': adaptive_thresholds['time_threshold'],
                'repetition_patterns': self._get_repetition_details(output_text),
                'anomaly_factors': self._get_anomaly_details(token_ratio, generation_time),
                'resource_indicators': self._get_resource_details(input_token_count, output_token_count),
                'efficiency_metrics': {
                    'tokens_per_second': output_token_count / max(generation_time, 0.001),
                    'time_per_token': time_per_token,
                    'ratio_vs_expected': token_ratio / expected_ratio
                },
                'component_scores': {
                    'repetition': repetition_score,
                    'complexity': complexity_score,
                    'anomaly': anomaly_score,
                    'resource': resource_score
                }
            }
            
            metrics = TokenBloatMetrics(
                input_tokens=input_token_count,
                output_tokens=output_token_count,
                generation_time=generation_time,
                token_ratio=token_ratio,
                repetition_score=repetition_score,
                complexity_score=complexity_score,
                anomaly_score=anomaly_score,
                resource_score=resource_score,
                final_score=final_score,
                risk_level=risk_level,
                details=details
            )
            
            # Update historical data
            self._update_historical_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Enhanced token bloat analysis failed: {str(e)}")
            raise
    
    def _detect_repetition_patterns(self, text: str) -> float:
        """Detect repetitive patterns that indicate bloat"""
        if not text or len(text) < 10:
            return 0.0
        
        repetition_score = 0.0
        words = text.split()
        
        if len(words) < 4:
            return 0.0
        
        # Check for repeated n-grams
        for n in [2, 3, 4]:
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            ngram_counts = Counter(ngrams)
            
            # Find most repeated n-gram
            if ngrams:
                max_count = max(ngram_counts.values())
                if max_count > 2:
                    repetition_score += min(0.3, (max_count - 2) * 0.1)
        
        # Check for identical consecutive sentences
        sentences = re.split(r'[.!?]+', text)
        consecutive_identical = 0
        for i in range(len(sentences)-1):
            if sentences[i].strip() and sentences[i].strip() == sentences[i+1].strip():
                consecutive_identical += 1
        
        if consecutive_identical > 1:
            repetition_score += min(0.4, consecutive_identical * 0.2)
        
        # Check for character-level repetition
        char_repetition = self._detect_character_repetition(text)
        repetition_score += char_repetition
        
        return min(repetition_score, 1.0)
    
    def _detect_character_repetition(self, text: str) -> float:
        """Detect character-level repetitive patterns"""
        if len(text) < 20:
            return 0.0
        
        # Look for patterns like "aaaaa", "121212", etc.
        repetition_patterns = [
            r'(.)\1{4,}',  # Same character repeated 5+ times
            r'(.{2,5})\1{3,}',  # Pattern repeated 4+ times
        ]
        
        score = 0.0
        for pattern in repetition_patterns:
            matches = re.findall(pattern, text)
            if matches:
                score += min(0.3, len(matches) * 0.1)
        
        return min(score, 0.5)
    
    def _analyze_input_complexity(self, input_text: str) -> float:
        """Analyze input complexity to set appropriate expectations"""
        complexity_score = 0.0
        
        # Check for question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(word in input_text.lower() for word in question_words):
            complexity_score += 0.3
        
        # Check for command/instruction words
        command_words = ['write', 'generate', 'create', 'explain', 'describe', 'list', 'summarize']
        if any(word in input_text.lower() for word in command_words):
            complexity_score += 0.4
        
        # Check for technical terms
        technical_indicators = ['algorithm', 'function', 'system', 'process', 'analysis', 'implementation']
        if any(word in input_text.lower() for word in technical_indicators):
            complexity_score += 0.2
        
        # Sentence structure complexity
        sentences = re.split(r'[.!?]+', input_text)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length > 15:
                complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    def _detect_statistical_anomalies(self, current_metrics: Dict) -> float:
        """Detect statistical anomalies in generation patterns"""
        if len(self.historical_metrics) < 10:
            return 0.0
        
        anomaly_score = 0.0
        
        # Check for outliers in multiple dimensions
        metrics_to_check = ['token_ratio', 'generation_time', 'time_per_token']
        
        for metric in metrics_to_check:
            if metric in current_metrics:
                historical_values = [
                    getattr(m, metric) if hasattr(m, metric) 
                    else m.details.get('efficiency_metrics', {}).get(metric, 0)
                    for m in list(self.historical_metrics)[-50:]  # Last 50 samples
                ]
                
                if historical_values and len(historical_values) > 5:
                    try:
                        mean_val = statistics.mean(historical_values)
                        std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                        current_value = current_metrics[metric]
                        
                        if std_val > 0:
                            z_score = abs(current_value - mean_val) / std_val
                            if z_score > 3:
                                anomaly_score += 0.4
                            elif z_score > 2:
                                anomaly_score += 0.2
                    except statistics.StatisticsError:
                        # Handle edge cases in statistical calculations
                        pass
        
        return min(anomaly_score, 1.0)
    
    def _estimate_resource_impact(self, input_tokens: int, output_tokens: int, 
                                generation_time: float) -> float:
        """Estimate resource impact based on token usage and timing"""
        resource_score = 0.0
        
        # High token count impact
        total_tokens = input_tokens + output_tokens
        if total_tokens > 2000:
            resource_score += 0.4
        elif total_tokens > 1000:
            resource_score += 0.2
        
        # Long generation time impact
        if generation_time > 10.0:
            resource_score += 0.4
        elif generation_time > 5.0:
            resource_score += 0.2
        
        # Inefficient generation (low tokens per second)
        if output_tokens > 0:
            tokens_per_second = output_tokens / generation_time
            if tokens_per_second < 5:
                resource_score += 0.3
            elif tokens_per_second < 10:
                resource_score += 0.1
        
        return min(resource_score, 1.0)
    
    def _calculate_adaptive_thresholds(self, input_tokens: int, 
                                     input_complexity: float) -> Dict:
        """Calculate adaptive thresholds based on input characteristics"""
        
        # Base expected ratio adjusted for input complexity
        if input_tokens < 5:  # Very short inputs
            base_ratio = 2.0
            size_threshold = 50
        elif input_tokens < 15:  # Short inputs
            base_ratio = 3.0
            size_threshold = 100
        elif input_tokens < 50:  # Medium inputs
            base_ratio = 4.0
            size_threshold = 200
        else:  # Long inputs
            base_ratio = 2.0
            size_threshold = 400
        
        # Adjust for input complexity
        complexity_multiplier = 1.0 + (input_complexity * 0.5)
        
        return {
            'expected_ratio': base_ratio * complexity_multiplier,
            'size_threshold': size_threshold,
            'time_threshold': 5.0 + (input_tokens * 0.1)
        }
    
    def _calculate_enhanced_score(self, token_ratio: float, generation_time: float,
                                input_tokens: int, output_tokens: int,
                                expected_ratio: float, repetition_score: float,
                                complexity_score: float, anomaly_score: float,
                                resource_score: float, thresholds: Dict) -> float:
        """Enhanced bloat score calculation with smooth continuity across 0-1 range"""
        
        # Factor 1: Smooth ratio scoring using sigmoid-like function
        adaptive_expected = thresholds['expected_ratio']
        ratio_factor = token_ratio / adaptive_expected
        
        # Smooth continuous scoring instead of step-based
        if ratio_factor <= 1.0:
            ratio_score = 0.0
        else:
            # Further optimized sigmoid for maximum smoothness
            k = 1.2  # Further reduced steepness
            offset = 1.6  # Even earlier start
            ratio_score = 1.0 / (1.0 + math.exp(-k * (ratio_factor - offset)))
        
        # Factor 2: Smooth size scoring with exponential growth
        size_threshold = thresholds['size_threshold']
        if output_tokens <= size_threshold:
            size_score = 0.0
        else:
            # Even gentler exponential growth
            size_factor = (output_tokens - size_threshold) / size_threshold
            size_score = 1.0 - math.exp(-0.5 * size_factor)  # Further reduced from 0.6
        
        # Factor 3: Smooth efficiency scoring
        if output_tokens > 0:
            time_per_token = generation_time / output_tokens
            # More gradual logarithmic scaling
            if time_per_token <= 0.2:
                efficiency_score = 0.0
            else:
                # Even gentler scaling
                efficiency_score = min(1.0, math.log(time_per_token / 0.2) / math.log(12))  # Increased from 8
        else:
            # Handle zero output case
            efficiency_score = min(1.0, generation_time / 5.0)  # Increased from 4.0
        
        # Factor 4: Enhanced repetition scoring with power scaling
        repetition_component = min(1.0, repetition_score ** 0.85)  # Increased from 0.8
        
        # Factor 5: Anomaly scoring with threshold smoothing
        anomaly_component = min(1.0, anomaly_score * 1.05)  # Further reduced from 1.1
        
        # Factor 6: Resource scoring with logarithmic scaling
        resource_component = min(1.0, resource_score ** 0.95)  # Increased from 0.9
        
        # Enhanced weighted combination with optimized weights
        base_score = (
            ratio_score * 0.26 +        # Further reduced from 0.28
            size_score * 0.24 +         # Increased from 0.22
            repetition_component * 0.22 + # Reduced from 0.23
            efficiency_score * 0.18 +    # Increased from 0.17
            anomaly_component * 0.06 +   # Kept same
            resource_component * 0.04    # Kept same
        )
        
        # Much gentler amplification to prevent large jumps
        # Using a more gradual tanh curve
        amplification_factor = 1.0 + 0.25 * math.tanh(2.0 * (base_score - 0.2))  # Reduced intensity further
        amplified_score = base_score * amplification_factor
        
        # Smoother power scaling for better distribution
        if amplified_score > 0.06:  # Further reduced threshold
            # Even gentler power scaling
            power = 0.92  # Increased from 0.88 for less compression
            final_score = amplified_score ** power
        else:
            final_score = amplified_score
        
        # Ensure smooth boundaries at extremes
        final_score = max(0.0, min(1.0, final_score))
        
        return round(final_score, 4)
    
    def _classify_risk_level(self, score: float) -> str:
        """Classify risk level based on final score with improved granularity"""
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.65:
            return "HIGH"
        elif score >= 0.45:
            return "MEDIUM"
        elif score >= 0.25:
            return "LOW"
        elif score >= 0.05:
            return "MINIMAL"
        else:
            return "NEGLIGIBLE"
    
    def _get_repetition_details(self, text: str) -> Dict:
        """Get detailed repetition analysis"""
        details = {}
        words = text.split()
        
        # Most repeated bigrams
        if len(words) >= 2:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            details['most_repeated_bigram'] = bigram_counts.most_common(1)
        
        # Character repetition patterns
        char_patterns = re.findall(r'(.)\1{3,}', text)
        details['character_repetitions'] = len(char_patterns)
        
        return details
    
    def _get_anomaly_details(self, token_ratio: float, generation_time: float) -> Dict:
        """Get detailed anomaly analysis"""
        return {
            'ratio_vs_baseline': token_ratio / self.baseline_stats['avg_ratio'],
            'time_vs_baseline': generation_time / max(self.baseline_stats['avg_time_per_token'], 0.001),
            'historical_samples': len(self.historical_metrics)
        }
    
    def _get_resource_details(self, input_tokens: int, output_tokens: int) -> Dict:
        """Get detailed resource analysis"""
        return {
            'total_tokens': input_tokens + output_tokens,
            'input_output_ratio': output_tokens / max(input_tokens, 1),
            'estimated_memory_mb': (input_tokens + output_tokens) * 0.004,  # Rough estimate
            'processing_complexity': 'high' if (input_tokens + output_tokens) > 1000 else 'normal'
        }
    
    def _update_historical_metrics(self, metrics: TokenBloatMetrics):
        """Update historical metrics for anomaly detection"""
        historical_metric = HistoricalMetrics(
            timestamp=time.time(),
            token_ratio=metrics.token_ratio,
            generation_time=metrics.generation_time,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            time_per_token=metrics.generation_time / max(metrics.output_tokens, 1)
        )
        
        self.historical_metrics.append(historical_metric)
        
        # Update baseline stats
        if len(self.historical_metrics) >= 10:
            recent_metrics = list(self.historical_metrics)[-20:]  # Last 20 samples
            
            try:
                recent_ratios = [m.token_ratio for m in recent_metrics]
                recent_times = [m.time_per_token for m in recent_metrics]
                
                self.baseline_stats['avg_ratio'] = statistics.mean(recent_ratios)
                self.baseline_stats['std_ratio'] = statistics.stdev(recent_ratios) if len(recent_ratios) > 1 else 0
                self.baseline_stats['avg_time_per_token'] = statistics.mean(recent_times)
                self.baseline_stats['std_time_per_token'] = statistics.stdev(recent_times) if len(recent_times) > 1 else 0
            except statistics.StatisticsError:
                # Handle edge cases
                pass

# ------------------- Endpoints -------------------
@app.get("/health")
def health_check():
    if model is None or tokenizer is None or generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}

@app.post("/detect/grammar", response_model=MetricReturnModel)
async def evaluate_text(request: GrammarRequest):
    start_time = time.time()
    
    try:
        logger.info("Analyzing grammatical correctness with T5-Base...")
        # Get prediction with details
        score, details = analyze_text(request.text)
        
        # Calculate processing time
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return MetricReturnModel(
            metric_name=EvaluationType.GRAMMATICAL_CORRECTNESS,
            actual_value=score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "inference_time_ms": processing_time_ms,
                "text_length": len(request.text),
                "correction_details": details
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/token-bloat-dos", response_model=MetricReturnModel)
async def detect_token_bloat_dos(req: TokenBloatRequest):
    """
    Enhanced token bloat and latency DoS detection with multi-dimensional analysis.
    """
    try:
        start_time = time.time()
        
        logger.info("Analyzing token bloat DoS patterns with enhanced detection...")
        
        # Initialize enhanced detector
        detector = TokenBloatDoSDetector(tokenizer, generator)
        
        # Perform comprehensive analysis
        metrics = detector.analyze_comprehensive(
            req.text, 
            req.max_output_length, 
            req.expected_ratio
        )
        
        # Create enhanced explanation
        explanation_parts = []
        if metrics.repetition_score > 0.3:
            explanation_parts.append("repetitive patterns detected")
        if metrics.anomaly_score > 0.3:
            explanation_parts.append("statistical anomalies identified")
        if metrics.resource_score > 0.3:
            explanation_parts.append("high resource consumption")
        if metrics.token_ratio > req.expected_ratio * 3:
            explanation_parts.append("excessive token generation")
        
        if not explanation_parts:
            explanation_parts.append("normal generation patterns")
        
        explanation = f"{metrics.risk_level.lower()} risk detected: {', '.join(explanation_parts)}"
        
        # Generate warnings based on component scores
        warnings = []
        if metrics.repetition_score > 0.5:
            warnings.append("High repetition patterns detected - possible generation loop")
        if metrics.anomaly_score > 0.5:
            warnings.append("Statistical anomaly detected - deviates significantly from baseline")
        if metrics.resource_score > 0.5:
            warnings.append("High resource consumption - potential DoS indicator")
        if metrics.token_ratio > req.expected_ratio * 3:
            warnings.append(f"Token ratio ({metrics.token_ratio:.1f}) exceeds expected by 3x")
        if metrics.generation_time > 15:  # 15+ seconds
            warnings.append("Generation time exceeds 15 seconds")
        
        return MetricReturnModel(
            metric_name=EvaluationType.TOKEN_BLOAT_DOS_EVALUATION,
            actual_value=metrics.final_score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "explanation": explanation,
                "risk_level": metrics.risk_level,
                "warnings": warnings,
                "component_scores": {
                    "repetition_score": metrics.repetition_score,
                    "complexity_score": metrics.complexity_score,
                    "anomaly_score": metrics.anomaly_score,
                    "resource_score": metrics.resource_score
                },
                "metrics": {
                    "token_ratio": metrics.token_ratio,
                    "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "generation_time": metrics.generation_time
                },
                "analysis_details": metrics.details,
                "processing_time": round(time.time() - start_time, 3)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced token bloat DoS detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 