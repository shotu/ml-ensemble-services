import os
import time
import logging
import sys
import threading
from enum import Enum
import re
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import spaCy - it's optional for some functionality
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = None  # Will be loaded in background
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# ------------------- Logging Configuration -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sentence_transformers_service")

# ------------------- Global Variables -------------------
model = None
model_loading = False
model_ready = False
startup_time = time.time()

# ------------------- FastAPI App Definition -------------------
app = FastAPI(
    title="Sentence Transformers Service",
    version="1.0.0",
    description="Computes semantic similarity between two texts using sentence transformers."
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
    SENTENCE_SIMILARITY_EVALUATION = "sentence_similarity_evaluation"
    SUPPLY_CHAIN_RISK_EVALUATION = "supply_chain_risk_evaluation"
    MEMBERSHIP_INFERENCE_RISK_EVALUATION = "membership_inference_risk_evaluation"
    CONTEXT_PRECISION_EVALUATION = "context_precision_evaluation"
    CONTEXT_RECALL_EVALUATION = "context_recall_evaluation"
    CONTEXT_ENTITIES_RECALL_EVALUATION = "context_entities_recall_evaluation"
    NOISE_SENSITIVITY_EVALUATION = "noise_sensitivity_evaluation"
    RESPONSE_RELEVANCY_EVALUATION = "response_relevancy_evaluation"
    CONTEXT_RELEVANCE_EVALUATION = "context_relevance_evaluation"
    # Agentic Metrics
    TOPIC_ADHERENCE_EVALUATION = "topic_adherence_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

# ------------------- Request Schema -------------------
class SimilarityRequest(BaseModel):
    text1: str
    text2: str

class SupplyChainRiskRequest(BaseModel):
    model_name: str
    plugin_sources: Optional[List[dict]] = []
    dataset_sources: Optional[List[dict]] = []

class MembershipInferenceRequest(BaseModel):
    text: str
    context: Optional[str] = None
    known_patterns: Optional[List[str]] = None

# New RAG Metrics Request Schemas
class RAGMetricRequest(BaseModel):
    llm_input_query: str
    llm_input_context: str
    llm_output: str

# Agentic Metrics Request Schema
class AgenticMetricsRequest(BaseModel):
    conversation_history: List[str]
    tool_calls: List[Dict[str, Any]]
    agent_responses: List[str]
    reference_data: Dict[str, Any]

# ------------------- RAG Metrics Implementation -------------------
class RAGMetricsEvaluator:
    """Enhanced RAG metrics evaluator using sentence transformers and spaCy"""
    
    def __init__(self, model, nlp_model=None):
        self.model = model
        self.nlp = nlp_model
        
        # Initialize TF-IDF vectorizer for syntactic analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        # Entity importance weights
        self.entity_weights = {
            "PERSON": 1.0,
            "ORG": 0.8,
            "GPE": 0.9,  # Geopolitical entity
            "DATE": 0.6,
            "TIME": 0.5,
            "MONEY": 0.7,
            "PERCENT": 0.6,
            "PRODUCT": 0.8,
            "EVENT": 0.7,
            "WORK_OF_ART": 0.6,
            "LAW": 0.8,
            "LANGUAGE": 0.5,
        }

    def split_context(self, context: str, method: str = "sentence") -> List[str]:
        """Split context into chunks for precision analysis"""
        if method == "sentence":
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', context.strip())
            return [s.strip() for s in sentences if len(s.strip()) > 10]
        elif method == "paragraph":
            paragraphs = context.split('\n\n')
            return [p.strip() for p in paragraphs if len(p.strip()) > 20]
        else:
            # Fixed-size chunks
            chunk_size = 200
            words = context.split()
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if len(chunk) > 50:
                    chunks.append(chunk)
            return chunks

    def get_adaptive_threshold(self, query: str) -> float:
        """Get adaptive similarity threshold based on query complexity"""
        query_length = len(query.split())
        query_complexity = len(set(query.lower().split()))
        
        # More complex queries need higher thresholds
        if query_length > 20 or query_complexity > 15:
            return 0.75
        elif query_length > 10 or query_complexity > 8:
            return 0.65
        else:
            return 0.55

    def evaluate_context_precision(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """Evaluate what fraction of context is relevant to the query"""
        start_time = time.time()
        
        try:
            # Split context into chunks
            context_chunks = self.split_context(context, method="sentence")
            
            if not context_chunks:
                return {
                    "precision_score": 0.0,
                    "relevant_chunks": 0,
                    "total_chunks": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Compute embeddings
            query_embedding = self.model.encode([query])
            chunk_embeddings = self.model.encode(context_chunks)
            
            # Calculate similarities
            similarities = sklearn_cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Get adaptive threshold
            threshold = self.get_adaptive_threshold(query)
            
            # Count relevant chunks
            relevant_chunks = sum(1 for sim in similarities if sim >= threshold)
            precision = relevant_chunks / len(context_chunks)
            
            processing_time = time.time() - start_time
            
            return {
                "precision_score": precision,
                "relevant_chunks": relevant_chunks,
                "total_chunks": len(context_chunks),
                "threshold_used": threshold,
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities)),
                "avg_similarity": float(np.mean(similarities)),
                "chunk_similarities": similarities.tolist(),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in context precision evaluation: {str(e)}")
            return {
                "precision_score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def extract_key_information(self, text: str) -> List[str]:
        """Extract key information/concepts from text"""
        # Use TF-IDF to extract important terms
        try:
            # Fit and transform the text
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = np.argsort(tfidf_scores)[-10:][::-1]
            key_terms = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            return key_terms
        except:
            # Fallback to simple word extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return list(set(words))[:10]

    def compute_semantic_coverage(self, expected_info: List[str], context_info: List[str]) -> float:
        """Compute semantic coverage between expected and context information"""
        if not expected_info or not context_info:
            return 0.0
        
        try:
            # Encode all terms
            all_terms = expected_info + context_info
            embeddings = self.model.encode(all_terms)
            
            expected_embeddings = embeddings[:len(expected_info)]
            context_embeddings = embeddings[len(expected_info):]
            
            # Calculate coverage
            coverage_scores = []
            for exp_emb in expected_embeddings:
                similarities = sklearn_cosine_similarity([exp_emb], context_embeddings)[0]
                max_sim = np.max(similarities) if len(similarities) > 0 else 0.0
                coverage_scores.append(max_sim)
            
            return float(np.mean(coverage_scores))
        except:
            return 0.0

    def evaluate_context_recall(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """Evaluate how well context covers the information needed for the response"""
        start_time = time.time()
        
        try:
            # Extract key information from response (what should be in context)
            expected_info = self.extract_key_information(response)
            context_info = self.extract_key_information(context)
            
            # Compute semantic coverage
            coverage_score = self.compute_semantic_coverage(expected_info, context_info)
            
            # Additional analysis: direct semantic similarity
            response_embedding = self.model.encode([response])
            context_embedding = self.model.encode([context])
            direct_similarity = sklearn_cosine_similarity(response_embedding, context_embedding)[0][0]
            
            # Weighted combination
            recall_score = 0.7 * coverage_score + 0.3 * direct_similarity
            
            processing_time = time.time() - start_time
            
            return {
                "recall_score": float(recall_score),
                "coverage_score": float(coverage_score),
                "direct_similarity": float(direct_similarity),
                "expected_concepts": expected_info,
                "context_concepts": context_info,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in context recall evaluation: {str(e)}")
            return {
                "recall_score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER"""
        if not self.nlp or not SPACY_AVAILABLE:
            # Fallback to simple pattern matching
            return self._extract_entities_fallback(text)
        
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "weight": self.entity_weights.get(ent.label_, 0.5)
                })
            return entities
        except:
            return self._extract_entities_fallback(text)

    def _extract_entities_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity extraction without spaCy"""
        entities = []
        
        # Simple patterns for common entity types
        patterns = {
            "PERSON": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "ORG": r'\b[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Organization))?\b',
            "GPE": r'\b(?:United States|USA|UK|Canada|France|Germany|Japan|China|India|Australia|New York|London|Paris|Tokyo)\b',
            "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b',
            "MONEY": r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
        }
        
        for label, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "weight": self.entity_weights.get(label, 0.5)
                })
        
        return entities

    def find_captured_entities(self, important_entities: List[Dict], context_entities: List[Dict]) -> List[Dict]:
        """Find which important entities are captured in context"""
        captured = []
        
        for imp_ent in important_entities:
            for ctx_ent in context_entities:
                # Check for exact match or semantic similarity
                if (imp_ent["text"].lower() == ctx_ent["text"].lower() or
                    imp_ent["text"].lower() in ctx_ent["text"].lower() or
                    ctx_ent["text"].lower() in imp_ent["text"].lower()):
                    captured.append({
                        "important_entity": imp_ent,
                        "context_entity": ctx_ent,
                        "match_type": "exact" if imp_ent["text"].lower() == ctx_ent["text"].lower() else "partial"
                    })
                    break
        
        return captured

    def calculate_weighted_recall(self, captured_entities: List[Dict], important_entities: List[Dict]) -> float:
        """Calculate weighted entity recall"""
        if not important_entities:
            return 1.0
        
        total_weight = sum(ent["weight"] for ent in important_entities)
        captured_weight = sum(cap["important_entity"]["weight"] for cap in captured_entities)
        
        return captured_weight / max(total_weight, 0.001)

    def evaluate_context_entities_recall(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """Evaluate entity coverage in context"""
        start_time = time.time()
        
        try:
            # Extract entities from all texts
            query_entities = self.extract_entities_spacy(query)
            context_entities = self.extract_entities_spacy(context)
            response_entities = self.extract_entities_spacy(response)
            
            # Important entities are from query and response
            important_entities = query_entities + response_entities
            
            # Remove duplicates
            seen_entities = set()
            unique_important = []
            for ent in important_entities:
                key = (ent["text"].lower(), ent["label"])
                if key not in seen_entities:
                    seen_entities.add(key)
                    unique_important.append(ent)
            
            # Find captured entities
            captured_entities = self.find_captured_entities(unique_important, context_entities)
            
            # Calculate weighted recall
            entity_recall = self.calculate_weighted_recall(captured_entities, unique_important)
            
            processing_time = time.time() - start_time
            
            return {
                "entity_recall_score": float(entity_recall),
                "query_entities": query_entities,
                "context_entities": context_entities,
                "response_entities": response_entities,
                "important_entities_count": len(unique_important),
                "captured_entities_count": len(captured_entities),
                "captured_entities": captured_entities,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in context entities recall evaluation: {str(e)}")
            return {
                "entity_recall_score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def inject_noise(self, context: str, noise_ratio: float = 0.3) -> str:
        """Inject noise into context by adding irrelevant sentences"""
        noise_sentences = [
            "The weather today is quite pleasant with clear skies.",
            "Technology continues to advance at a rapid pace.",
            "Many people enjoy reading books in their spare time.",
            "Coffee is one of the most popular beverages worldwide.",
            "Sports events attract millions of viewers globally.",
            "Music has the power to evoke strong emotions.",
            "Cooking is both an art and a science.",
            "Travel broadens one's perspective on life.",
        ]
        
        context_sentences = self.split_context(context, method="sentence")
        num_noise = max(1, int(len(context_sentences) * noise_ratio))
        
        # Add random noise sentences
        import random
        selected_noise = random.sample(noise_sentences, min(num_noise, len(noise_sentences)))
        
        # Mix original and noise sentences
        all_sentences = context_sentences + selected_noise
        random.shuffle(all_sentences)
        
        return ' '.join(all_sentences)

    def compute_relevancy_score(self, response: str, context: str) -> float:
        """Compute relevancy between response and context"""
        try:
            response_embedding = self.model.encode([response])
            context_embedding = self.model.encode([context])
            similarity = sklearn_cosine_similarity(response_embedding, context_embedding)[0][0]
            return float(similarity)
        except:
            return 0.0

    def evaluate_noise_sensitivity(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """Evaluate how sensitive the response is to noise in context"""
        start_time = time.time()
        
        try:
            # Compute relevancy with clean context
            clean_relevancy = self.compute_relevancy_score(response, context)
            
            # Create noisy version of context
            noisy_context = self.inject_noise(context, noise_ratio=0.3)
            noisy_relevancy = self.compute_relevancy_score(response, noisy_context)
            
            # Calculate sensitivity (lower is better)
            relevancy_drop = max(0, clean_relevancy - noisy_relevancy)
            sensitivity_score = 1.0 - min(1.0, relevancy_drop)
            
            processing_time = time.time() - start_time
            
            return {
                "noise_sensitivity_score": float(sensitivity_score),
                "clean_relevancy": float(clean_relevancy),
                "noisy_relevancy": float(noisy_relevancy),
                "relevancy_drop": float(relevancy_drop),
                "noise_injected": True,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in noise sensitivity evaluation: {str(e)}")
            return {
                "noise_sensitivity_score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def compute_syntactic_similarity(self, query: str, response: str) -> float:
        """Compute syntactic similarity using TF-IDF"""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([query, response])
            similarity = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

    def compute_pragmatic_similarity(self, query: str, response: str) -> float:
        """Compute pragmatic similarity based on question-answer patterns"""
        # Simple heuristics for pragmatic analysis
        query_lower = query.lower()
        response_lower = response.lower()
        
        score = 0.0
        
        # Question type matching
        if query_lower.startswith(('what', 'who', 'where', 'when', 'why', 'how')):
            if any(word in response_lower for word in ['is', 'are', 'was', 'were', 'because', 'due to']):
                score += 0.3
        
        # Keyword overlap
        query_words = set(query_lower.split())
        response_words = set(response_lower.split())
        overlap = len(query_words.intersection(response_words))
        score += min(0.4, overlap / max(len(query_words), 1) * 0.4)
        
        # Length appropriateness
        length_ratio = len(response) / max(len(query), 1)
        if 0.5 <= length_ratio <= 3.0:
            score += 0.3
        
        return min(1.0, score)

    def evaluate_response_relevancy(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """Evaluate how relevant the response is to the query"""
        start_time = time.time()
        
        try:
            # Multi-dimensional similarity analysis
            query_embedding = self.model.encode([query])
            response_embedding = self.model.encode([response])
            
            # Semantic similarity
            semantic_sim = sklearn_cosine_similarity(query_embedding, response_embedding)[0][0]
            
            # Syntactic similarity
            syntactic_sim = self.compute_syntactic_similarity(query, response)
            
            # Pragmatic similarity
            pragmatic_sim = self.compute_pragmatic_similarity(query, response)
            
            # Weighted combination
            relevancy_score = (0.6 * semantic_sim + 0.25 * syntactic_sim + 0.15 * pragmatic_sim)
            
            processing_time = time.time() - start_time
            
            return {
                "response_relevancy_score": float(relevancy_score),
                "semantic_similarity": float(semantic_sim),
                "syntactic_similarity": float(syntactic_sim),
                "pragmatic_similarity": float(pragmatic_sim),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in response relevancy evaluation: {str(e)}")
            return {
                "response_relevancy_score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def extract_query_aspects(self, query: str) -> List[str]:
        """Extract key aspects/information needs from the query"""
        try:
            # Use spaCy for entity and noun phrase extraction if available
            if self.nlp:
                doc = self.nlp(query)
                aspects = []
                
                # Extract named entities
                for ent in doc.ents:
                    aspects.append(ent.text)
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:  # Avoid very long phrases
                        aspects.append(chunk.text)
                
                # Extract key verbs and their objects
                for token in doc:
                    if token.pos_ == "VERB" and token.dep_ in ["ROOT", "aux"]:
                        verb_phrase = " ".join([child.text for child in token.children if child.dep_ in ["dobj", "pobj"]])
                        if verb_phrase:
                            aspects.append(f"{token.text} {verb_phrase}")
                
                # Remove duplicates and filter
                aspects = list(set([asp.strip() for asp in aspects if len(asp.strip()) > 2]))
                
                if aspects:
                    return aspects[:8]  # Limit to top 8 aspects
            
            # Fallback: Use TF-IDF and simple extraction
            key_terms = self.extract_key_information(query)
            
            # Extract question words and their context
            question_patterns = [
                r'what\s+(?:is|are|was|were|does|do|did)\s+([^?]+)',
                r'how\s+(?:does|do|did|can|could|would|will)\s+([^?]+)',
                r'when\s+(?:is|was|does|do|did|will)\s+([^?]+)',
                r'where\s+(?:is|are|was|were|does|do)\s+([^?]+)',
                r'why\s+(?:is|are|was|were|does|do|did)\s+([^?]+)',
                r'who\s+(?:is|are|was|were|does|do|did)\s+([^?]+)',
                r'which\s+([^?]+)',
            ]
            
            aspects = []
            query_lower = query.lower()
            for pattern in question_patterns:
                matches = re.findall(pattern, query_lower)
                aspects.extend(matches)
            
            # Combine with key terms
            aspects.extend(key_terms[:5])
            
            # Clean and deduplicate
            aspects = list(set([asp.strip() for asp in aspects if len(asp.strip()) > 2]))
            
            return aspects[:8] if aspects else [query]  # Fallback to full query
            
        except Exception as e:
            logger.warning(f"Error extracting query aspects: {str(e)}")
            return [query]

    def compute_aspect_coverage(self, query_aspects: List[str], context_chunks: List[str]) -> Dict[str, float]:
        """Compute how well context chunks cover each query aspect"""
        if not query_aspects or not context_chunks:
            return {}
        
        try:
            # Encode all aspects and chunks
            all_texts = query_aspects + context_chunks
            embeddings = self.model.encode(all_texts)
            
            aspect_embeddings = embeddings[:len(query_aspects)]
            chunk_embeddings = embeddings[len(query_aspects):]
            
            # Calculate coverage for each aspect
            aspect_coverage = {}
            for i, aspect in enumerate(query_aspects):
                similarities = sklearn_cosine_similarity([aspect_embeddings[i]], chunk_embeddings)[0]
                max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
                aspect_coverage[aspect] = max_similarity
            
            return aspect_coverage
            
        except Exception as e:
            logger.warning(f"Error computing aspect coverage: {str(e)}")
            return {}

    def evaluate_context_relevance(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """
        Evaluate how relevant the retrieved context is to the query's information needs.
        
        Context Relevance measures how well the context addresses the query's intent
        and information requirements, focusing on semantic alignment and coverage.
        """
        start_time = time.time()
        
        try:
            # Extract key aspects/information needs from the query
            query_aspects = self.extract_query_aspects(query)
            
            # Split context into semantic chunks
            context_chunks = self.split_context(context, method="sentence")
            
            if not context_chunks or not query_aspects:
                return {
                    "context_relevance_score": 0.0,
                    "query_aspects": query_aspects,
                    "context_chunks_count": len(context_chunks),
                    "processing_time": time.time() - start_time
                }
            
            # Compute overall query-context semantic similarity
            query_embedding = self.model.encode([query])
            context_embedding = self.model.encode([context])
            overall_similarity = float(sklearn_cosine_similarity(query_embedding, context_embedding)[0][0])
            
            # Compute aspect-level coverage
            aspect_coverage = self.compute_aspect_coverage(query_aspects, context_chunks)
            
            # Calculate aspect coverage scores
            coverage_scores = list(aspect_coverage.values())
            avg_aspect_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
            min_aspect_coverage = np.min(coverage_scores) if coverage_scores else 0.0
            max_aspect_coverage = np.max(coverage_scores) if coverage_scores else 0.0
            
            # Count well-covered aspects (using adaptive threshold)
            threshold = self.get_adaptive_threshold(query)
            well_covered_aspects = sum(1 for score in coverage_scores if score >= threshold)
            aspect_coverage_ratio = well_covered_aspects / len(query_aspects) if query_aspects else 0.0
            
            # Compute chunk-level relevance to query
            chunk_embeddings = self.model.encode(context_chunks)
            chunk_similarities = sklearn_cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Calculate relevance metrics
            relevant_chunks = sum(1 for sim in chunk_similarities if sim >= threshold)
            chunk_relevance_ratio = relevant_chunks / len(context_chunks)
            
            # Compute final context relevance score
            # Weighted combination of different relevance signals
            context_relevance_score = (
                0.4 * overall_similarity +           # Overall semantic alignment
                0.3 * avg_aspect_coverage +          # Average aspect coverage
                0.2 * aspect_coverage_ratio +        # Proportion of aspects covered
                0.1 * chunk_relevance_ratio          # Proportion of relevant chunks
            )
            
            # Apply quality adjustments
            if min_aspect_coverage < 0.3:  # Penalty for poorly covered aspects
                context_relevance_score *= 0.9
            
            if max_aspect_coverage > 0.8:  # Bonus for well-covered aspects
                context_relevance_score = min(1.0, context_relevance_score * 1.05)
            
            processing_time = time.time() - start_time
            
            return {
                "context_relevance_score": float(context_relevance_score),
                "overall_similarity": overall_similarity,
                "query_aspects": query_aspects,
                "aspect_coverage": aspect_coverage,
                "avg_aspect_coverage": float(avg_aspect_coverage),
                "min_aspect_coverage": float(min_aspect_coverage),
                "max_aspect_coverage": float(max_aspect_coverage),
                "well_covered_aspects": well_covered_aspects,
                "aspect_coverage_ratio": float(aspect_coverage_ratio),
                "context_chunks_count": len(context_chunks),
                "relevant_chunks": relevant_chunks,
                "chunk_relevance_ratio": float(chunk_relevance_ratio),
                "threshold_used": threshold,
                "chunk_similarities": chunk_similarities.tolist(),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in context relevance evaluation: {str(e)}")
            return {
                "context_relevance_score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

# ------------------- Supply Chain Risk Detection -------------------
class SupplyChainRiskAnalyzer:
    def __init__(self):
        # License trust scores (0.0 = risky, 1.0 = trusted)
        self.license_scores = {
            # High trust - permissive open source
            "mit": 1.0,
            "apache-2.0": 1.0,
            "bsd-3-clause": 0.95,
            "bsd-2-clause": 0.95,
            "isc": 0.9,
            "cc0-1.0": 0.85,
            
            # Medium trust - copyleft but established
            "gpl-3.0": 0.7,
            "gpl-2.0": 0.7,
            "lgpl-3.0": 0.8,
            "lgpl-2.1": 0.8,
            "agpl-3.0": 0.6,
            "mpl-2.0": 0.75,
            
            # Lower trust - restrictive or uncommon
            "cc-by-4.0": 0.6,
            "cc-by-sa-4.0": 0.5,
            "epl-2.0": 0.65,
            "cddl-1.0": 0.55,
            
            # Very low trust
            "proprietary": 0.2,
            "commercial": 0.3,
            "unknown": 0.1,
            "none": 0.0,
        }
        
        # Origin trust scores
        self.origin_scores = {
            # Highly trusted platforms
            "huggingface.co": 1.0,
            "github.com": 0.9,
            "gitlab.com": 0.85,
            "pypi.org": 0.8,
            "conda.anaconda.org": 0.8,
            
            # Academic/research institutions
            "arxiv.org": 0.9,
            "papers.nips.cc": 0.85,
            "openreview.net": 0.85,
            
            # Corporate but established
            "tensorflow.org": 0.9,
            "pytorch.org": 0.9,
            "microsoft.com": 0.75,
            "google.com": 0.75,
            "facebook.com": 0.7,
            
            # Medium trust
            "sourceforge.net": 0.6,
            "bitbucket.org": 0.7,
            
            # Lower trust
            "unknown": 0.3,
            "localhost": 0.1,
            "private": 0.2,
        }

    def analyze_huggingface_model(self, model_name: str) -> dict:
        """Analyze a Hugging Face model for supply chain risks."""
        try:
            from huggingface_hub import HfApi, ModelInfo
            
            api = HfApi()
            model_info = api.model_info(model_name)
            
            # Extract metadata
            license_info = getattr(model_info, 'license', 'unknown')
            author = getattr(model_info, 'author', 'unknown')
            downloads = getattr(model_info, 'downloads', 0)
            created_at = getattr(model_info, 'created_at', None)
            updated_at = getattr(model_info, 'last_modified', None)
            
            # Calculate license score
            license_key = str(license_info).lower() if license_info else 'unknown'
            license_score = self.license_scores.get(license_key, 0.3)
            
            # Calculate popularity score (downloads)
            if downloads > 100000:
                popularity_score = 1.0
            elif downloads > 10000:
                popularity_score = 0.8
            elif downloads > 1000:
                popularity_score = 0.6
            elif downloads > 100:
                popularity_score = 0.4
            else:
                popularity_score = 0.2
            
            # Calculate recency score
            import datetime
            if updated_at:
                days_since_update = (datetime.datetime.now(datetime.timezone.utc) - updated_at).days
                if days_since_update < 30:
                    recency_score = 1.0
                elif days_since_update < 90:
                    recency_score = 0.8
                elif days_since_update < 365:
                    recency_score = 0.6
                elif days_since_update < 730:
                    recency_score = 0.4
                else:
                    recency_score = 0.2
            else:
                recency_score = 0.3
            
            return {
                "model_name": model_name,
                "license": license_info,
                "license_score": license_score,
                "author": author,
                "downloads": downloads,
                "popularity_score": popularity_score,
                "recency_score": recency_score,
                "created_at": str(created_at) if created_at else None,
                "updated_at": str(updated_at) if updated_at else None,
                "origin": "huggingface.co",
                "origin_score": self.origin_scores.get("huggingface.co", 0.8)
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze HuggingFace model {model_name}: {e}")
            return {
                "model_name": model_name,
                "license": "unknown",
                "license_score": 0.1,
                "error": str(e),
                "origin": "unknown",
                "origin_score": 0.3
            }

    def analyze_github_source(self, repo_url: str) -> dict:
        """Analyze a GitHub repository for supply chain risks."""
        try:
            # Extract owner/repo from URL
            if "github.com" in repo_url:
                parts = repo_url.replace("https://", "").replace("http://", "").split("/")
                if len(parts) >= 3:
                    owner, repo = parts[1], parts[2]
                    
                    # GitHub API call
                    api_url = f"https://api.github.com/repos/{owner}/{repo}"
                    response = requests.get(api_url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        license_info = data.get("license", {})
                        license_key = license_info.get("key", "unknown").lower() if license_info else "unknown"
                        license_score = self.license_scores.get(license_key, 0.3)
                        
                        stars = data.get("stargazers_count", 0)
                        forks = data.get("forks_count", 0)
                        
                        # Calculate popularity from stars
                        if stars > 10000:
                            popularity_score = 1.0
                        elif stars > 1000:
                            popularity_score = 0.8
                        elif stars > 100:
                            popularity_score = 0.6
                        elif stars > 10:
                            popularity_score = 0.4
                        else:
                            popularity_score = 0.2
                        
                        return {
                            "repo_url": repo_url,
                            "license": license_key,
                            "license_score": license_score,
                            "stars": stars,
                            "forks": forks,
                            "popularity_score": popularity_score,
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                            "origin": "github.com",
                            "origin_score": self.origin_scores.get("github.com", 0.9)
                        }
            
            return {
                "repo_url": repo_url,
                "license": "unknown",
                "license_score": 0.2,
                "error": "Could not analyze repository",
                "origin": "unknown",
                "origin_score": 0.3
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze GitHub repo {repo_url}: {e}")
            return {
                "repo_url": repo_url,
                "license": "unknown", 
                "license_score": 0.1,
                "error": str(e),
                "origin": "unknown",
                "origin_score": 0.3
            }

    def analyze_plugin_source(self, plugin_info: dict) -> dict:
        """Analyze a generic plugin source."""
        name = plugin_info.get("name", "unknown")
        origin = plugin_info.get("origin", "unknown").lower()
        license_info = plugin_info.get("license", "unknown").lower()
        
        # Score the origin
        origin_score = self.origin_scores.get(origin, 0.3)
        
        # Score the license
        license_score = self.license_scores.get(license_info, 0.3)
        
        return {
            "name": name,
            "origin": origin,
            "origin_score": origin_score,
            "license": license_info,
            "license_score": license_score,
            "trust_score": (origin_score + license_score) / 2
        }

    def calculate_overall_risk(self, analyses: List[dict]) -> dict:
        """Calculate overall supply chain risk from individual analyses."""
        if not analyses:
            return {
                "overall_risk_score": 0.5,
                "risk_level": "medium",
                "explanation": "No sources to analyze"
            }
        
        # Extract trust scores
        trust_scores = []
        for analysis in analyses:
            if "trust_score" in analysis:
                trust_scores.append(analysis["trust_score"])
            else:
                # Calculate trust score from license and origin
                license_score = analysis.get("license_score", 0.3)
                origin_score = analysis.get("origin_score", 0.3)
                popularity_score = analysis.get("popularity_score", 0.5)
                recency_score = analysis.get("recency_score", 0.5)
                
                # Weighted average
                trust_score = (
                    license_score * 0.4 +
                    origin_score * 0.3 +
                    popularity_score * 0.2 +
                    recency_score * 0.1
                )
                trust_scores.append(trust_score)
        
        # Calculate overall trust (average with bias toward lowest scores)
        if trust_scores:
            avg_trust = sum(trust_scores) / len(trust_scores)
            min_trust = min(trust_scores)
            
            # Weight toward the minimum to be conservative
            overall_trust = (avg_trust * 0.7) + (min_trust * 0.3)
        else:
            overall_trust = 0.3
        
        # Convert trust to risk (inverse)
        overall_risk = 1.0 - overall_trust
        
        # Determine risk level
        if overall_risk < 0.3:
            risk_level = "low"
        elif overall_risk < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "overall_risk_score": round(overall_risk, 3),
            "overall_trust_score": round(overall_trust, 3),
            "risk_level": risk_level,
            "individual_trust_scores": [round(score, 3) for score in trust_scores],
            "min_trust_score": round(min(trust_scores), 3) if trust_scores else 0.3,
            "avg_trust_score": round(sum(trust_scores) / len(trust_scores), 3) if trust_scores else 0.3
        }

# ------------------- Enhanced Similarity Functions -------------------
def cosine_similarity(vec1, vec2):
    """Calculate raw cosine similarity between two vectors"""
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity

def preprocess_text(text):
    """Preprocess text for better similarity calculation"""
    if not text or not text.strip():
        return ""
    return text.strip().lower()

def detect_empty_or_nonsemantic(text1, text2):
    """Detect cases that should have very low similarity scores"""
    # Check for empty texts
    if not text1.strip() or not text2.strip():
        return True, 0.0
    
    # Check for very short texts (likely non-semantic)
    if len(text1.strip()) < 3 or len(text2.strip()) < 3:
        return True, 0.1
    
    # Check if texts are just numbers or single characters
    if (text1.strip().isdigit() and text2.strip().isalpha()) or \
       (text1.strip().isalpha() and text2.strip().isdigit()):
        return True, 0.0
    
    # Check if one text is just punctuation/symbols
    if not any(c.isalnum() for c in text1) or not any(c.isalnum() for c in text2):
        return True, 0.05
    
    return False, None

def calibrate_similarity_score(raw_similarity, text1, text2):
    """
    Calibrate similarity score to better use the full 0-1 range
    and handle edge cases appropriately
    """
    # Check for special cases first
    is_special, special_score = detect_empty_or_nonsemantic(text1, text2)
    if is_special:
        return special_score
    
    # Convert from [-1, 1] to [0, 1] range
    normalized_score = (raw_similarity + 1) / 2
    
    # Apply calibration to expand the range and reduce baseline
    # This transforms the typical 0.4-1.0 range to better use 0.0-1.0
    
    # Step 1: Shift and scale to reduce high baseline
    # Typical sentence similarity rarely goes below 0.4, so we remap
    baseline_threshold = 0.4  # Typical minimum for sentence embeddings
    
    if normalized_score < baseline_threshold:
        # Very low similarity - map to 0.0-0.2 range
        calibrated = (normalized_score / baseline_threshold) * 0.2
    else:
        # Above baseline - map to 0.2-1.0 range
        calibrated = 0.2 + ((normalized_score - baseline_threshold) / (1.0 - baseline_threshold)) * 0.8
    
    # Step 2: Apply non-linear transformation to increase differentiation
    # Use a sigmoid-like curve to better separate different similarity levels
    calibrated = apply_differentiation_curve(calibrated)
    
    # Step 3: Ensure bounds
    calibrated = np.clip(calibrated, 0.0, 1.0)
    
    return float(calibrated)

def apply_differentiation_curve(score):
    """
    Apply a curve to better differentiate similarity levels
    This helps separate medium similarities from high similarities
    """
    # Use a modified sigmoid curve for better differentiation
    # This makes the middle range more sensitive to differences
    
    # Apply power transformation to increase differentiation
    if score < 0.5:
        # Lower scores get compressed more (to push unrelated content lower)
        return score ** 1.5
    else:
        # Higher scores get expanded (to maintain high similarity detection)
        return 1.0 - (1.0 - score) ** 0.7

def enhanced_similarity_analysis(text1, text2, raw_similarity):
    """
    Perform additional semantic analysis to adjust similarity
    """
    # Length difference penalty (very different lengths often indicate different content)
    len_diff = abs(len(text1) - len(text2))
    max_len = max(len(text1), len(text2))
    
    if max_len > 0:
        length_penalty = min(len_diff / max_len, 0.5)  # Max 50% penalty
    else:
        length_penalty = 0
    
    # Word overlap bonus/penalty
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) > 0 and len(words2) > 0:
        word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # If cosine similarity is high but word overlap is very low, reduce score
        if raw_similarity > 0.8 and word_overlap < 0.1:
            length_penalty += 0.2  # Additional penalty for high cosine but low word overlap
    
    return length_penalty

# ------------------- Async Model Loading -------------------
def load_model_in_background():
    global model, model_loading, model_ready, nlp
    model_loading = True
    try:
        # Set up model cache directory
        model_dir = "/app/model_cache"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # Ensure environment variables are set for the model cache
        os.environ["TRANSFORMERS_CACHE"] = model_dir
        os.environ["HF_HOME"] = model_dir
        
        # Load the sentence transformer model
        model_name = "sentence-transformers/all-mpnet-base-v2"
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name, cache_folder=model_dir)
        
        # Test the model with a simple encoding to ensure it works
        model.encode(["test query"])
        logger.info("Sentence transformer model loaded successfully!")
        
        # Load spaCy model if available
        if SPACY_AVAILABLE:
            try:
                logger.info("Loading spaCy transformer model...")
                nlp = spacy.load("en_core_web_trf")
                
                # Test spaCy model
                doc = nlp("Test entity recognition.")
                logger.info(f"spaCy model loaded successfully! Found {len(doc.ents)} entities in test.")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {str(e)}")
                nlp = None
        else:
            logger.warning("spaCy not available - entity-based metrics will use fallback methods")
        
        # Mark model as ready
        model_ready = True
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        logger.error(f"Stack trace: {sys.exc_info()}")
    finally:
        model_loading = False

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
    
    # Default payload for similarity computation
    payload = {
        "text1": os.getenv("PING_TEXT1", "Hello world"),
        "text2": os.getenv("PING_TEXT2", "Hello world")
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

# -------------------- MEMBERSHIP INFERENCE RISK DETECTION --------------------

class MembershipInferenceDetector:
    """
    Detects membership inference risk using semantic similarity-based approaches.
    Inspired by PETAL (Per-Token Semantic Similarity) and modern MIA research.
    """
    
    def __init__(self, model):
        self.model = model
        
    def compute_perplexity_proxy(self, text: str, context: str = None) -> float:
        """
        Compute a perplexity proxy using semantic similarity.
        Based on the insight that members have higher semantic coherence.
        """
        if not text.strip():
            return 1.0
            
        # Split text into sentences for analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5
        
        # Compute embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        # Calculate semantic coherence (average pairwise similarity)
        total_similarity = 0.0
        pairs = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j])
                total_similarity += float(similarity)
                pairs += 1
        
        if pairs == 0:
            return 0.5
            
        avg_similarity = total_similarity / pairs
        
        # Convert similarity to perplexity-like score (higher similarity = lower perplexity = higher membership likelihood)
        perplexity_proxy = max(0.0, 1.0 - avg_similarity)
        
        return float(perplexity_proxy)
    
    def detect_semantic_patterns(self, text: str, known_patterns: List[str] = None) -> float:
        """
        Detect semantic patterns that might indicate training data membership.
        """
        if not known_patterns:
            known_patterns = []
        
        # Generic patterns that might indicate training data
        training_indicators = [
            "according to", "research shows", "studies indicate", 
            "it is well known", "as mentioned earlier", "in conclusion",
            "furthermore", "moreover", "however", "therefore",
            "for example", "such as", "including", "namely"
        ]
        
        all_patterns = known_patterns + training_indicators
        text_lower = text.lower()
        
        # Count pattern matches
        pattern_matches = sum(1 for pattern in all_patterns if pattern.lower() in text_lower)
        
        # Normalize by text length
        words = len(text.split())
        if words == 0:
            return 0.0
            
        pattern_density = pattern_matches / words
        return min(1.0, pattern_density * 10)  # Scale appropriately
    
    def analyze_context_similarity(self, text: str, context: str = None) -> float:
        """
        Analyze similarity between text and context using ReCaLL-inspired approach.
        """
        if not context or not context.strip():
            return 0.3  # Neutral score when no context
        
        # Compute embeddings for text and context
        text_embedding = self.model.encode([text])
        context_embedding = self.model.encode([context])
        
        # Calculate semantic similarity
        similarity = np.dot(text_embedding[0], context_embedding[0])
        
        # High similarity with context might indicate membership
        # (text and context from same distribution/source)
        return float(max(0.0, min(1.0, similarity)))
    
    def detect_memorization_indicators(self, text: str) -> float:
        """
        Detect indicators of potential memorization based on text characteristics.
        """
        if not text.strip():
            return 0.0
        
        words = text.split()
        if len(words) < 5:
            return 0.2
        
        memorization_score = 0.0
        
        # Check for repetitive patterns
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # High word repetition might indicate memorization
        max_freq = max(word_freq.values()) if word_freq else 1
        repetition_score = min(0.3, (max_freq - 1) / len(words))
        memorization_score += repetition_score
        
        # Check for unusual specificity (numbers, dates, proper nouns)
        specific_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\d+\.\d+\b',  # Decimal numbers
        ]
        
        specific_count = 0
        for pattern in specific_patterns:
            specific_count += len(re.findall(pattern, text))
        
        specificity_score = min(0.4, specific_count / len(words))
        memorization_score += specificity_score
        
        # Check for formal/academic language patterns
        formal_indicators = [
            "furthermore", "moreover", "nevertheless", "consequently",
            "therefore", "thus", "hence", "accordingly", "subsequently"
        ]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text.lower())
        formal_score = min(0.3, formal_count / len(words) * 20)
        memorization_score += formal_score
        
        return min(1.0, memorization_score)
    
    def detect(self, text: str, context: str = None, known_patterns: List[str] = None) -> tuple:
        """
        Main detection method combining multiple membership inference signals.
        """
        # Calculate individual component scores
        perplexity_proxy = self.compute_perplexity_proxy(text, context)
        pattern_score = self.detect_semantic_patterns(text, known_patterns)
        context_similarity = self.analyze_context_similarity(text, context)
        memorization_score = self.detect_memorization_indicators(text)
        
        # Weighted combination inspired by modern MIA research
        weights = {
            'perplexity_proxy': 0.35,    # Primary signal - semantic coherence
            'pattern_detection': 0.25,   # Training data patterns
            'context_similarity': 0.25,  # ReCaLL-inspired context analysis
            'memorization': 0.15         # Memorization indicators
        }
        
        # Combine scores (lower perplexity proxy = higher membership likelihood)
        membership_score = (
            (1.0 - perplexity_proxy) * weights['perplexity_proxy'] +
            pattern_score * weights['pattern_detection'] +
            context_similarity * weights['context_similarity'] +
            memorization_score * weights['memorization']
        )
        
        # Ensure minimum score if any component is detected
        if any([pattern_score > 0.1, context_similarity > 0.5, memorization_score > 0.1]):
            membership_score = max(0.05, membership_score)
        
        # Create detailed breakdown
        details = {
            'perplexity_proxy': round(perplexity_proxy, 3),
            'pattern_detection_score': round(pattern_score, 3),
            'context_similarity_score': round(context_similarity, 3),
            'memorization_indicators': round(memorization_score, 3),
            'component_breakdown': {
                'text_length_words': len(text.split()),
                'context_length_words': len(context.split()) if context else 0,
                'known_patterns_count': len(known_patterns) if known_patterns else 0,
                'has_context': bool(context and context.strip())
            }
        }
        
        explanation = self._generate_explanation(membership_score, details)
        
        return round(membership_score, 3), details, explanation
    
    def _generate_explanation(self, score: float, details: dict) -> str:
        """
        Generate human-readable explanation of the membership inference score.
        """
        if score == 0.0:
            return "No indicators of training data membership detected"
        
        explanations = []
        
        if details['perplexity_proxy'] < 0.5:
            explanations.append("high semantic coherence detected")
        
        if details['pattern_detection_score'] > 0.1:
            explanations.append("training data patterns identified")
        
        if details['context_similarity_score'] > 0.5:
            explanations.append("high context similarity")
        
        if details['memorization_indicators'] > 0.1:
            explanations.append("memorization indicators present")
        
        risk_level = "low" if score < 0.3 else "medium" if score < 0.7 else "high"
        
        base_explanation = f"Membership inference risk: {risk_level} ({score:.3f})"
        if explanations:
            base_explanation += f" - {', '.join(explanations)}"
            
        return base_explanation

@app.on_event("startup")
async def startup_event():
    global startup_time
    startup_time = time.time()
    
    # Start model loading in background to avoid blocking the app startup
    thread = threading.Thread(target=load_model_in_background)
    thread.daemon = True
    thread.start()
    logger.info("Model loading started in background thread")

    # Start background ping thread if enabled
    if os.getenv("ENABLE_PING", "false").lower() == "true":
        threading.Thread(target=background_ping, daemon=True).start()
        logger.info("Background ping service started")

# ------------------- Middleware for Request Handling -------------------
@app.middleware("http")
async def check_model_readiness(request: Request, call_next):
    # Always allow health checks to proceed
    if request.url.path == "/health":
        return await call_next(request)
        
    # For all other endpoints, check if model is ready
    global model, model_ready
    if not model_ready:
        if model_loading:
            return Response(
                content="Model is still loading, please try again later",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "10"}
            )
        else:
            # If not loading and not ready, something went wrong
            return Response(
                content="Model failed to load, service is unavailable",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    # If model is ready, proceed with the request
    return await call_next(request)

# ------------------- Health Check -------------------
@app.get("/health")
def health_check():
    uptime = time.time() - startup_time
    
    # If model is loading, return 200 but with a message
    if model_loading:
        return {
            "status": "initializing", 
            "message": "Model is still loading",
            "uptime_seconds": uptime
        }
    
    # If model failed to load
    if not model_ready and not model_loading:
        return {
            "status": "error", 
            "message": "Model failed to load",
            "uptime_seconds": uptime
        }
    
    # If model is loaded, test it
    if model_ready:
        try:
            # Test the model with a simple encoding
            test_result = model.encode(["test query"])
            return {
                "status": "ok", 
                "message": "Model loaded and working",
                "uptime_seconds": uptime
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "error", 
                "message": f"Model loaded but failed test: {str(e)}",
                "uptime_seconds": uptime
            }
    
    # Fallback response (should never reach here)
    return {
        "status": "unknown", 
        "message": "Unknown state",
        "uptime_seconds": uptime
    }

# ------------------- Enhanced Similarity Computation -------------------
def compute_similarity(text1: str, text2: str) -> float:
    """Compute enhanced semantic similarity between two texts"""
    try:
        # Preprocess texts
        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)
        
        # Handle identical texts
        if processed_text1 == processed_text2:
            return 1.0
        
        # Encode both texts
        embeddings = model.encode([text1, text2])
        
        # Calculate raw cosine similarity
        raw_similarity = cosine_similarity(embeddings[0], embeddings[1])
        
        # Apply enhanced similarity analysis
        length_penalty = enhanced_similarity_analysis(text1, text2, raw_similarity)
        
        # Calibrate the similarity score
        calibrated_score = calibrate_similarity_score(raw_similarity, text1, text2)
        
        # Apply length penalty
        final_score = calibrated_score * (1.0 - length_penalty)
        
        # Ensure bounds
        final_score = np.clip(final_score, 0.0, 1.0)
        
        logger.info(f"Similarity computation: raw={raw_similarity:.3f}, "
                   f"calibrated={calibrated_score:.3f}, final={final_score:.3f}")
        
        return float(final_score)
        
    except Exception as e:
        logger.error(f"Error in similarity calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

# ------------------- Inference Endpoint -------------------
@app.post("/compute/similarity", response_model=MetricReturnModel)
async def evaluate_similarity(req: SimilarityRequest):
    """
    Compute semantic similarity between two texts using sentence transformers.
    
    Returns a similarity score between 0 and 1, where 1 indicates identical meaning.
    """
    start_time = time.time()
    
    try:
        # Compute similarity
        similarity_score = compute_similarity(req.text1, req.text2)
        
        # Calculate processing time
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Create response
        result = MetricReturnModel(
            metric_name=EvaluationType.SENTENCE_SIMILARITY_EVALUATION,
            actual_value=similarity_score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": processing_time_ms,
                "text1_length": len(req.text1),
                "text2_length": len(req.text2),
                "model_name": "sentence-transformers/all-mpnet-base-v2"
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in similarity evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/supply-chain-risk", response_model=MetricReturnModel)
async def detect_supply_chain_risk(req: SupplyChainRiskRequest):
    """
    Analyze supply chain risks for models, plugins, and datasets.
    """
    start_time = time.time()
    
    try:
        logger.info("Analyzing supply chain risks...")
        analyzer = SupplyChainRiskAnalyzer()
        
        all_analyses = []
        
        # Analyze the main model if provided
        if req.model_name:
            logger.info(f"Analyzing model: {req.model_name}")
            model_analysis = analyzer.analyze_huggingface_model(req.model_name)
            model_analysis["source_type"] = "model"
            all_analyses.append(model_analysis)
        
        # Analyze plugin sources
        for plugin in req.plugin_sources:
            logger.info(f"Analyzing plugin: {plugin.get('name', 'unknown')}")
            if "github.com" in plugin.get("origin", ""):
                plugin_analysis = analyzer.analyze_github_source(plugin.get("origin"))
            else:
                plugin_analysis = analyzer.analyze_plugin_source(plugin)
            plugin_analysis["source_type"] = "plugin"
            all_analyses.append(plugin_analysis)
        
        # Analyze dataset sources
        for dataset in req.dataset_sources:
            logger.info(f"Analyzing dataset: {dataset.get('name', 'unknown')}")
            if "github.com" in dataset.get("origin", ""):
                dataset_analysis = analyzer.analyze_github_source(dataset.get("origin"))
            else:
                dataset_analysis = analyzer.analyze_plugin_source(dataset)
            dataset_analysis["source_type"] = "dataset"
            all_analyses.append(dataset_analysis)
        
        # Calculate overall risk
        risk_summary = analyzer.calculate_overall_risk(all_analyses)
        
        # Calculate processing time
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Create response
        result = MetricReturnModel(
            metric_name=EvaluationType.SUPPLY_CHAIN_RISK_EVALUATION,
            actual_value=risk_summary["overall_risk_score"],
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": processing_time_ms,
                "sources_analyzed": len(all_analyses),
                "risk_level": risk_summary["risk_level"],
                "overall_trust_score": risk_summary["overall_trust_score"],
                "min_trust_score": risk_summary["min_trust_score"],
                "avg_trust_score": risk_summary["avg_trust_score"],
                "detailed_analyses": all_analyses,
                "risk_summary": risk_summary
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in supply chain risk analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/status")
def debug_status():
    """Debug endpoint to get system status"""
    return {
        "model_loading": model_loading,
        "model_ready": model_ready,
        "uptime_seconds": time.time() - startup_time,
        "environment": {
            "PORT": os.getenv("PORT", "8080"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "production"),
            "HF_HOME": os.getenv("HF_HOME", ""),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE", "")
        },
        "model_cache": {
            "exists": os.path.exists("/app/model_cache")
        },
        "python_version": sys.version,
        "model_info": str(model)
    }

@app.post("/detect/membership-inference-risk", response_model=MetricReturnModel)
async def detect_membership_inference_risk(req: MembershipInferenceRequest):
    """
    Detect potential membership inference risk using semantic similarity-based approaches.
    
    Estimates likelihood that a particular sample was part of training data using
    modern MIA techniques including semantic similarity analysis, context comparison,
    and memorization indicators with continuous scoring from 0.0 to 1.0.
    """
    try:
        start_time = time.time()
        
        # Check if model is ready
        if not model_ready:
            raise HTTPException(status_code=503, detail="Model is still loading. Please try again later.")
        
        detector = MembershipInferenceDetector(model)
        score, details, explanation = detector.detect(
            text=req.text,
            context=req.context,
            known_patterns=req.known_patterns
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return MetricReturnModel(
            metric_name=EvaluationType.MEMBERSHIP_INFERENCE_RISK_EVALUATION,
            actual_value=score,
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "explanation": explanation,
                "processing_time_ms": round(processing_time, 2),
                "text_length": len(req.text),
                "perplexity_proxy": details['perplexity_proxy'],
                "pattern_detection_score": details['pattern_detection_score'],
                "context_similarity_score": details['context_similarity_score'],
                "memorization_indicators": details['memorization_indicators'],
                "component_breakdown": details['component_breakdown']
            }
        )
        
    except Exception as e:
        logger.error(f"Membership inference detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# ------------------- RAG Metrics Endpoints -------------------

@app.post("/evaluate/context-precision", response_model=MetricReturnModel)
async def evaluate_context_precision_endpoint(req: RAGMetricRequest):
    """
    Evaluate context precision - what fraction of context is relevant to the query
    
    This metric measures how much of the provided context is actually relevant
    to answering the query, helping identify noise and irrelevant information.
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.llm_input_context.strip():
            raise HTTPException(status_code=400, detail="Context cannot be empty")
        if not req.llm_input_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Initialize evaluator
        evaluator = RAGMetricsEvaluator(model, nlp)
        
        # Perform evaluation
        result = evaluator.evaluate_context_precision(
            query=req.llm_input_query,
            context=req.llm_input_context,
            response=req.llm_output
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.CONTEXT_PRECISION_EVALUATION,
            "actual_value": result["precision_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "relevant_chunks": result.get("relevant_chunks", 0),
                "total_chunks": result.get("total_chunks", 0),
                "threshold_used": result.get("threshold_used", 0.0),
                "max_similarity": result.get("max_similarity", 0.0),
                "min_similarity": result.get("min_similarity", 0.0),
                "avg_similarity": result.get("avg_similarity", 0.0),
                "input_lengths": {
                    "query_length": len(req.llm_input_query),
                    "context_length": len(req.llm_input_context),
                    "response_length": len(req.llm_output)
                },
                "model_name": "all-mpnet-base-v2",
                "evaluation_method": "semantic_similarity_chunking"
            }
        }
        
        logger.info(f"Context precision evaluation completed in {processing_time:.4f}s - Score: {result['precision_score']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in context precision evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/evaluate/context-recall", response_model=MetricReturnModel)
async def evaluate_context_recall_endpoint(req: RAGMetricRequest):
    """
    Evaluate context recall - how well context covers information needed for the response
    
    This metric measures whether the context contains sufficient information
    to support the generated response.
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.llm_input_context.strip():
            raise HTTPException(status_code=400, detail="Context cannot be empty")
        if not req.llm_output.strip():
            raise HTTPException(status_code=400, detail="LLM output cannot be empty")
        
        # Initialize evaluator
        evaluator = RAGMetricsEvaluator(model, nlp)
        
        # Perform evaluation
        result = evaluator.evaluate_context_recall(
            query=req.llm_input_query,
            context=req.llm_input_context,
            response=req.llm_output
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.CONTEXT_RECALL_EVALUATION,
            "actual_value": result["recall_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "coverage_score": result.get("coverage_score", 0.0),
                "direct_similarity": result.get("direct_similarity", 0.0),
                "expected_concepts": result.get("expected_concepts", []),
                "context_concepts": result.get("context_concepts", []),
                "input_lengths": {
                    "query_length": len(req.llm_input_query),
                    "context_length": len(req.llm_input_context),
                    "response_length": len(req.llm_output)
                },
                "model_name": "all-mpnet-base-v2",
                "evaluation_method": "semantic_coverage_analysis"
            }
        }
        
        logger.info(f"Context recall evaluation completed in {processing_time:.4f}s - Score: {result['recall_score']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in context recall evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/evaluate/context-entities-recall", response_model=MetricReturnModel)
async def evaluate_context_entities_recall_endpoint(req: RAGMetricRequest):
    """
    Evaluate context entities recall - how well context covers important entities
    
    This metric measures whether the context contains the named entities
    (persons, organizations, locations, etc.) that are important for the query and response.
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.llm_input_context.strip():
            raise HTTPException(status_code=400, detail="Context cannot be empty")
        if not req.llm_input_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Initialize evaluator
        evaluator = RAGMetricsEvaluator(model, nlp)
        
        # Perform evaluation
        result = evaluator.evaluate_context_entities_recall(
            query=req.llm_input_query,
            context=req.llm_input_context,
            response=req.llm_output
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.CONTEXT_ENTITIES_RECALL_EVALUATION,
            "actual_value": result["entity_recall_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "important_entities_count": result.get("important_entities_count", 0),
                "captured_entities_count": result.get("captured_entities_count", 0),
                "query_entities": result.get("query_entities", []),
                "context_entities": result.get("context_entities", []),
                "response_entities": result.get("response_entities", []),
                "captured_entities": result.get("captured_entities", []),
                "spacy_available": SPACY_AVAILABLE,
                "input_lengths": {
                    "query_length": len(req.llm_input_query),
                    "context_length": len(req.llm_input_context),
                    "response_length": len(req.llm_output)
                },
                "model_name": "all-mpnet-base-v2",
                "ner_model": "en_core_web_trf" if SPACY_AVAILABLE and nlp else "fallback_patterns",
                "evaluation_method": "named_entity_recognition"
            }
        }
        
        logger.info(f"Context entities recall evaluation completed in {processing_time:.4f}s - Score: {result['entity_recall_score']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in context entities recall evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/evaluate/noise-sensitivity", response_model=MetricReturnModel)
async def evaluate_noise_sensitivity_endpoint(req: RAGMetricRequest):
    """
    Evaluate noise sensitivity - how robust the response is to noise in context
    
    This metric measures how much the quality of the response degrades
    when irrelevant information is added to the context.
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.llm_input_context.strip():
            raise HTTPException(status_code=400, detail="Context cannot be empty")
        if not req.llm_output.strip():
            raise HTTPException(status_code=400, detail="LLM output cannot be empty")
        
        # Initialize evaluator
        evaluator = RAGMetricsEvaluator(model, nlp)
        
        # Perform evaluation
        result = evaluator.evaluate_noise_sensitivity(
            query=req.llm_input_query,
            context=req.llm_input_context,
            response=req.llm_output
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.NOISE_SENSITIVITY_EVALUATION,
            "actual_value": result["noise_sensitivity_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "clean_relevancy": result.get("clean_relevancy", 0.0),
                "noisy_relevancy": result.get("noisy_relevancy", 0.0),
                "relevancy_drop": result.get("relevancy_drop", 0.0),
                "noise_injected": result.get("noise_injected", False),
                "input_lengths": {
                    "query_length": len(req.llm_input_query),
                    "context_length": len(req.llm_input_context),
                    "response_length": len(req.llm_output)
                },
                "model_name": "all-mpnet-base-v2",
                "evaluation_method": "noise_injection_analysis"
            }
        }
        
        logger.info(f"Noise sensitivity evaluation completed in {processing_time:.4f}s - Score: {result['noise_sensitivity_score']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in noise sensitivity evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/evaluate/response-relevancy", response_model=MetricReturnModel)
async def evaluate_response_relevancy_endpoint(req: RAGMetricRequest):
    """
    Evaluate response relevancy - how relevant the response is to the query
    
    This metric measures how well the generated response addresses the original query
    using semantic, syntactic, and pragmatic analysis.
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.llm_input_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        if not req.llm_output.strip():
            raise HTTPException(status_code=400, detail="LLM output cannot be empty")
        
        # Initialize evaluator
        evaluator = RAGMetricsEvaluator(model, nlp)
        
        # Perform evaluation
        result = evaluator.evaluate_response_relevancy(
            query=req.llm_input_query,
            context=req.llm_input_context,
            response=req.llm_output
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.RESPONSE_RELEVANCY_EVALUATION,
            "actual_value": result["response_relevancy_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "semantic_similarity": result.get("semantic_similarity", 0.0),
                "syntactic_similarity": result.get("syntactic_similarity", 0.0),
                "pragmatic_similarity": result.get("pragmatic_similarity", 0.0),
                "input_lengths": {
                    "query_length": len(req.llm_input_query),
                    "context_length": len(req.llm_input_context),
                    "response_length": len(req.llm_output)
                },
                "model_name": "all-mpnet-base-v2",
                "evaluation_method": "multi_dimensional_similarity"
            }
        }
        
        logger.info(f"Response relevancy evaluation completed in {processing_time:.4f}s - Score: {result['response_relevancy_score']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in response relevancy evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/evaluate/context-relevance", response_model=MetricReturnModel)
async def evaluate_context_relevance_endpoint(req: RAGMetricRequest):
    """
    Evaluate context relevance - how relevant the retrieved context is to the query's information needs.
    
    Context Relevance measures how well the context addresses the query's intent
    and information requirements, focusing on semantic alignment and coverage.
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.llm_input_context.strip():
            raise HTTPException(status_code=400, detail="Context cannot be empty")
        if not req.llm_output.strip():
            raise HTTPException(status_code=400, detail="LLM output cannot be empty")
        
        # Initialize evaluator
        evaluator = RAGMetricsEvaluator(model, nlp)
        
        # Perform evaluation
        result = evaluator.evaluate_context_relevance(
            query=req.llm_input_query,
            context=req.llm_input_context,
            response=req.llm_output
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.CONTEXT_RELEVANCE_EVALUATION,
            "actual_value": result["context_relevance_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "overall_similarity": result.get("overall_similarity", 0.0),
                "query_aspects": result.get("query_aspects", []),
                "aspect_coverage": result.get("aspect_coverage", {}),
                "avg_aspect_coverage": result.get("avg_aspect_coverage", 0.0),
                "min_aspect_coverage": result.get("min_aspect_coverage", 0.0),
                "max_aspect_coverage": result.get("max_aspect_coverage", 0.0),
                "well_covered_aspects": result.get("well_covered_aspects", 0),
                "aspect_coverage_ratio": result.get("aspect_coverage_ratio", 0.0),
                "context_chunks_count": result.get("context_chunks_count", 0),
                "relevant_chunks": result.get("relevant_chunks", 0),
                "chunk_relevance_ratio": result.get("chunk_relevance_ratio", 0.0),
                "threshold_used": result.get("threshold_used", 0.0),
                "chunk_similarities": result.get("chunk_similarities", []),
                "input_lengths": {
                    "query_length": len(req.llm_input_query),
                    "context_length": len(req.llm_input_context),
                    "response_length": len(req.llm_output)
                },
                "model_name": "all-mpnet-base-v2",
                "evaluation_method": "semantic_alignment_analysis"
            }
        }
        
        logger.info(f"Context relevance evaluation completed in {processing_time:.4f}s - Score: {result['context_relevance_score']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in context relevance evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# ------------------- Agentic Metrics Endpoints -------------------

@app.post("/evaluate/topic-adherence", response_model=MetricReturnModel)
async def evaluate_topic_adherence_endpoint(req: AgenticMetricsRequest):
    """
    Evaluate if the agent stays on topic over time by evaluating conversation alignment with reference topics.
    
    This metric measures how well the agent maintains focus on the expected topics
    throughout the conversation using semantic similarity analysis.
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not req.conversation_history:
            raise HTTPException(status_code=400, detail="Conversation history cannot be empty")
        if not req.reference_data.get("expected_topics"):
            raise HTTPException(status_code=400, detail="Expected topics must be provided in reference_data")
        
        # Initialize evaluator
        evaluator = TopicAdherenceEvaluator(model, nlp)
        
        # Perform evaluation
        result = evaluator.evaluate_topic_adherence(
            conversation_history=req.conversation_history,
            expected_topics=req.reference_data["expected_topics"]
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "metric_name": EvaluationType.TOPIC_ADHERENCE_EVALUATION,
            "actual_value": result["topic_adherence_score"],
            "actual_value_type": ActualValueDtype.FLOAT,
            "others": {
                "processing_time_ms": processing_time * 1000,
                "matched_topics": result.get("matched_topics", []),
                "precision": result.get("precision", 0.0),
                "recall": result.get("recall", 0.0),
                "f1_score": result.get("f1_score", 0.0),
                "conversation_topics": result.get("conversation_topics", []),
                "expected_topics": result.get("expected_topics", []),
                "topic_similarities": result.get("topic_similarities", []),
                "threshold_used": result.get("threshold_used", 0.0),
                "input_lengths": {
                    "conversation_messages": len(req.conversation_history),
                    "expected_topics": len(req.reference_data.get("expected_topics", [])),
                    "total_conversation_length": sum(len(msg) for msg in req.conversation_history)
                },
                "model_name": "all-mpnet-base-v2",
                "evaluation_method": "semantic_topic_matching"
            }
        }
        
        logger.info(f"Topic adherence evaluation completed in {processing_time:.4f}s - Score: {result['topic_adherence_score']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in topic adherence evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# ------------------- Agentic Metrics Implementation -------------------
class TopicAdherenceEvaluator:
    """Topic adherence evaluator using sentence transformers and spaCy"""
    
    def __init__(self, model, nlp):
        self.model = model
        self.nlp = nlp
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def extract_conversation_topics(self, conversation_history: List[str]) -> List[str]:
        """Extract topics from conversation using spaCy NER and noun phrases"""
        # Join all conversation messages
        conversation_text = " ".join(conversation_history)
        topics = []
        
        try:
            if self.nlp:
                doc = self.nlp(conversation_text)
                
                # Extract named entities
                entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 2]
                topics.extend(entities)
                
                # Extract noun phrases
                noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 2]
                topics.extend(noun_phrases)
                
                # Extract key verbs and their objects for action topics
                for token in doc:
                    if token.pos_ == "VERB" and token.dep_ in ["ROOT"]:
                        verb_phrase = " ".join([child.text for child in token.children if child.dep_ in ["dobj", "pobj"]])
                        if verb_phrase:
                            topics.append(f"{token.text} {verb_phrase}".lower())
            
            # Fallback: extract using TF-IDF
            if not topics:
                try:
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform([conversation_text])
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    tfidf_scores = tfidf_matrix.toarray()[0]
                    
                    # Get top terms
                    top_indices = np.argsort(tfidf_scores)[-15:][::-1]
                    topics = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                except:
                    # Final fallback: simple word extraction
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', conversation_text.lower())
                    topics = list(set(words))[:15]
            
            # Remove duplicates and clean
            topics = list(set([topic.strip() for topic in topics if len(topic.strip()) > 2]))
            
        except Exception as e:
            logger.error(f"Error extracting conversation topics: {str(e)}")
            # Fallback to simple word extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', conversation_text.lower())
            topics = list(set(words))[:15]
        
        return topics[:20]  # Limit to top 20 topics
    
    def evaluate_topic_adherence(self, conversation_history: List[str], expected_topics: List[str]) -> Dict[str, Any]:
        """
        Evaluate if the agent stays on topic over time by evaluating conversation alignment with reference topics.
        
        Args:
            conversation_history: List of conversation messages
            expected_topics: Expected topics the conversation should cover
            
        Returns:
            Dict containing F1 score and topic matching details
        """
        start_time = time.time()
        
        try:
            if not conversation_history or not expected_topics:
                return {
                    "topic_adherence_score": 0.0,
                    "matched_topics": [],
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "processing_time": time.time() - start_time
                }
            
            # Extract topics from conversation
            conversation_topics = self.extract_conversation_topics(conversation_history)
            
            # Calculate semantic similarity between conversation topics and expected topics
            matched_topics = []
            topic_similarities = []
            
            if conversation_topics and expected_topics:
                # Encode all topics
                all_topics = conversation_topics + expected_topics
                topic_embeddings = self.model.encode(all_topics)
                
                conv_embeddings = topic_embeddings[:len(conversation_topics)]
                exp_embeddings = topic_embeddings[len(conversation_topics):]
                
                # Find matches for each expected topic
                threshold = 0.6  # Semantic similarity threshold
                
                for i, expected_topic in enumerate(expected_topics):
                    exp_embedding = exp_embeddings[i:i+1]
                    similarities = sklearn_cosine_similarity(exp_embedding, conv_embeddings)[0]
                    max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
                    topic_similarities.append(float(max_similarity))
                    
                    if max_similarity >= threshold:
                        # Find the best matching conversation topic
                        best_match_idx = int(np.argmax(similarities))
                        matched_topics.append({
                            "expected_topic": expected_topic,
                            "matched_topic": conversation_topics[best_match_idx],
                            "similarity": float(max_similarity)
                        })
            
            # Calculate precision, recall, and F1
            total_expected = len(expected_topics)
            total_matched = len(matched_topics)

            precision = float(total_matched / total_expected) if total_expected > 0 else 0.0
            recall = precision  # In this context, recall equals precision
            f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Continuous score: average of topic_similarities (max similarity for each expected topic)
            if topic_similarities:
                continuous_score = float(np.mean(topic_similarities))
            else:
                continuous_score = 0.0

            processing_time = time.time() - start_time

            return {
                "topic_adherence_score": continuous_score,  # Use continuous score as main
                "matched_topics": matched_topics,
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "conversation_topics": conversation_topics,
                "expected_topics": expected_topics,
                "topic_similarities": topic_similarities,
                "threshold_used": float(threshold),
                "processing_time": float(processing_time)
            }
            
        except Exception as e:
            logger.error(f"Error in topic adherence evaluation: {str(e)}")
            return {
                "topic_adherence_score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

# If running as a script, start the server
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable, default to 8080
    port = int(os.getenv("PORT", "8080"))
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port) 