# Sentence Transformers Service

This service computes semantic similarity between two texts, analyzes supply chain risks, and provides comprehensive RAG (Retrieval-Augmented Generation) evaluation metrics using sentence transformers (SBERT).

## Overview

The Sentence Transformers service provides multiple capabilities:

1. **Semantic Similarity Analysis**: Analyzes two pieces of text to determine their semantic similarity using the `all-mpnet-base-v2` model
2. **Supply Chain Risk Analysis**: Evaluates the security risks of models, plugins, and datasets from various sources
3. **RAG Evaluation Metrics**: Comprehensive evaluation of RAG systems with 5 specialized metrics
4. **Membership Inference Detection**: Detects potential training data membership risks

The similarity service returns a continuous similarity score between 0 and 1:
- **1.0**: Texts have identical or nearly identical meaning
- **0.5**: Texts are somewhat related but have different meanings
- **0.0**: Texts have completely different or opposite meanings

## API Endpoints

### Semantic Similarity
`POST /compute/similarity`

### Request Format

```json
{
  "text1": "The first text to compare",
  "text2": "The second text to compare"
}
```

### Response Format

```json
{
  "metric_name": "sentence_similarity_evaluation",
  "actual_value": 0.85,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 142.5,
    "text1_length": 25,
    "text2_length": 27,
    "model_name": "sentence-transformers/all-mpnet-base-v2"
  }
}
```

## RAG Evaluation Metrics

The service provides 5 comprehensive RAG evaluation metrics that assess different aspects of retrieval-augmented generation systems. All metrics use the same request format and return scores between 0.0 and 1.0.

### Common Request Format for RAG Metrics

```json
{
  "llm_input_query": "What is the capital of France?",
  "llm_input_context": "Paris is the capital and largest city of France. It is located in the north-central part of the country.",
  "llm_output": "The capital of France is Paris."
}
```

### 1. Context Precision (`POST /evaluate/context-precision`)

**Purpose**: Measures what fraction of the provided context is actually relevant to answering the query.

**Algorithm**:
1. **Context Chunking**: Splits context into sentences using regex patterns
2. **Semantic Embedding**: Encodes query and each context chunk using sentence-transformers
3. **Adaptive Thresholding**: Calculates similarity threshold based on query complexity:
   - Complex queries (>20 words): threshold = 0.75
   - Medium queries (10-20 words): threshold = 0.65
   - Simple queries (<10 words): threshold = 0.55
4. **Relevance Scoring**: Counts chunks above threshold as relevant
5. **Precision Calculation**: `relevant_chunks / total_chunks`

**Implementation Logic**:
```python
# Query complexity analysis
query_length = len(query.split())
query_complexity = len(set(query.lower().split()))

# Adaptive threshold
if query_length > 20 or query_complexity > 15:
    threshold = 0.75  # Stricter for complex queries
elif query_length > 10 or query_complexity > 8:
    threshold = 0.65
else:
    threshold = 0.55  # More lenient for simple queries

# Semantic similarity calculation
query_embedding = model.encode([query])
chunk_embeddings = model.encode(context_chunks)
similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

# Count relevant chunks
relevant_chunks = sum(1 for sim in similarities if sim >= threshold)
precision = relevant_chunks / len(context_chunks)
```

**Response Example**:
```json
{
  "metric_name": "context_precision_evaluation",
  "actual_value": 0.75,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 245.3,
    "relevant_chunks": 3,
    "total_chunks": 4,
    "threshold_used": 0.65,
    "max_similarity": 0.89,
    "min_similarity": 0.23,
    "avg_similarity": 0.67,
    "evaluation_method": "semantic_similarity_chunking"
  }
}
```

### 2. Context Recall (`POST /evaluate/context-recall`)

**Purpose**: Measures how well the context covers the information needed to generate the response.

**Algorithm**:
1. **Concept Extraction**: Uses TF-IDF to extract key concepts from the response
2. **Context Analysis**: Extracts key concepts from the context
3. **Semantic Coverage**: Measures how well context concepts cover response concepts
4. **Direct Similarity**: Calculates direct semantic similarity between response and context
5. **Weighted Combination**: `0.7 * coverage_score + 0.3 * direct_similarity`

**Implementation Logic**:
```python
# TF-IDF concept extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
response_concepts = extract_key_concepts(response)  # Top 10 TF-IDF terms
context_concepts = extract_key_concepts(context)

# Semantic coverage calculation
all_terms = response_concepts + context_concepts
embeddings = model.encode(all_terms)
response_embeddings = embeddings[:len(response_concepts)]
context_embeddings = embeddings[len(response_concepts):]

# For each response concept, find best match in context
coverage_scores = []
for resp_emb in response_embeddings:
    similarities = cosine_similarity([resp_emb], context_embeddings)[0]
    max_sim = np.max(similarities) if len(similarities) > 0 else 0.0
    coverage_scores.append(max_sim)

coverage_score = np.mean(coverage_scores)

# Direct similarity
direct_similarity = cosine_similarity(
    model.encode([response]), 
    model.encode([context])
)[0][0]

# Final score
recall_score = 0.7 * coverage_score + 0.3 * direct_similarity
```

**Response Example**:
```json
{
  "metric_name": "context_recall_evaluation",
  "actual_value": 0.82,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 189.7,
    "coverage_score": 0.78,
    "direct_similarity": 0.94,
    "expected_concepts": ["paris", "capital", "france", "city"],
    "context_concepts": ["paris", "capital", "largest", "city", "france", "north", "central"],
    "evaluation_method": "semantic_coverage_analysis"
  }
}
```

### 3. Context Entities Recall (`POST /evaluate/context-entities-recall`)

**Purpose**: Measures how well the context covers important named entities from the query and response.

**Algorithm**:
1. **Entity Extraction**: Uses spaCy transformer NER (`en_core_web_trf`) or regex fallback
2. **Entity Weighting**: Assigns importance weights by entity type:
   - PERSON: 1.0, ORG: 0.8, GPE: 0.9, DATE: 0.6, etc.
3. **Entity Matching**: Finds exact and partial matches between important and context entities
4. **Weighted Recall**: `captured_weight / total_important_weight`

**Implementation Logic**:
```python
# Entity extraction using spaCy transformer NER
nlp = spacy.load("en_core_web_trf")
query_entities = extract_entities_spacy(query)
context_entities = extract_entities_spacy(context)
response_entities = extract_entities_spacy(response)

# Entity importance weights
entity_weights = {
    "PERSON": 1.0, "ORG": 0.8, "GPE": 0.9,
    "DATE": 0.6, "MONEY": 0.7, "PRODUCT": 0.8
}

# Important entities from query + response
important_entities = query_entities + response_entities

# Remove duplicates
unique_important = remove_duplicate_entities(important_entities)

# Find captured entities (exact + partial matching)
captured = find_captured_entities(unique_important, context_entities)

# Calculate weighted recall
total_weight = sum(ent["weight"] for ent in unique_important)
captured_weight = sum(cap["important_entity"]["weight"] for cap in captured)
entity_recall = captured_weight / max(total_weight, 0.001)
```

**Fallback Strategy** (when spaCy unavailable):
```python
# Regex patterns for common entity types
patterns = {
    "PERSON": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
    "ORG": r'\b[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd))?\b',
    "GPE": r'\b(?:United States|USA|UK|Canada|France|Germany|Japan|China)\b',
    "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b',
    "MONEY": r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
}
```

**Response Example**:
```json
{
  "metric_name": "context_entities_recall_evaluation",
  "actual_value": 0.91,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 312.1,
    "important_entities_count": 3,
    "captured_entities_count": 3,
    "query_entities": [{"text": "France", "label": "GPE", "weight": 0.9}],
    "context_entities": [{"text": "Paris", "label": "GPE", "weight": 0.9}, {"text": "France", "label": "GPE", "weight": 0.9}],
    "response_entities": [{"text": "France", "label": "GPE", "weight": 0.9}, {"text": "Paris", "label": "GPE", "weight": 0.9}],
    "spacy_available": true,
    "ner_model": "en_core_web_trf",
    "evaluation_method": "named_entity_recognition"
  }
}
```

### 4. Noise Sensitivity (`POST /evaluate/noise-sensitivity`)

**Purpose**: Measures how robust the response quality is when irrelevant information is added to the context.

**Algorithm**:
1. **Baseline Measurement**: Calculate response-context relevancy with clean context
2. **Noise Injection**: Add irrelevant sentences (30% of original context size)
3. **Degraded Measurement**: Calculate response-context relevancy with noisy context
4. **Sensitivity Calculation**: `1.0 - max(0, clean_relevancy - noisy_relevancy)`

**Implementation Logic**:
```python
# Noise sentences to inject
noise_sentences = [
    "The weather today is quite pleasant with clear skies.",
    "Technology continues to advance at a rapid pace.",
    "Many people enjoy reading books in their spare time.",
    "Coffee is one of the most popular beverages worldwide.",
    "Sports events attract millions of viewers globally."
]

# Calculate baseline relevancy
clean_relevancy = cosine_similarity(
    model.encode([response]), 
    model.encode([context])
)[0][0]

# Inject noise (30% ratio)
context_sentences = split_context(context)
num_noise = max(1, int(len(context_sentences) * 0.3))
selected_noise = random.sample(noise_sentences, min(num_noise, len(noise_sentences)))

# Mix and shuffle
all_sentences = context_sentences + selected_noise
random.shuffle(all_sentences)
noisy_context = ' '.join(all_sentences)

# Calculate degraded relevancy
noisy_relevancy = cosine_similarity(
    model.encode([response]), 
    model.encode([noisy_context])
)[0][0]

# Sensitivity score (higher = more robust)
relevancy_drop = max(0, clean_relevancy - noisy_relevancy)
sensitivity_score = 1.0 - min(1.0, relevancy_drop)
```

**Response Example**:
```json
{
  "metric_name": "noise_sensitivity_evaluation",
  "actual_value": 0.87,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 198.4,
    "clean_relevancy": 0.92,
    "noisy_relevancy": 0.79,
    "relevancy_drop": 0.13,
    "noise_injected": true,
    "evaluation_method": "noise_injection_analysis"
  }
}
```

### 5. Response Relevancy (`POST /evaluate/response-relevancy`)

**Purpose**: Measures how relevant the generated response is to the original query using multi-dimensional analysis.

**Algorithm**:
1. **Semantic Analysis** (60% weight): Sentence-transformer embeddings similarity
2. **Syntactic Analysis** (25% weight): TF-IDF cosine similarity for lexical overlap
3. **Pragmatic Analysis** (15% weight): Heuristic question-answer pattern matching
4. **Weighted Combination**: Final score combines all three dimensions

**Implementation Logic**:
```python
# 1. Semantic similarity (60% weight)
query_embedding = model.encode([query])
response_embedding = model.encode([response])
semantic_sim = cosine_similarity(query_embedding, response_embedding)[0][0]

# 2. Syntactic similarity (25% weight)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform([query, response])
syntactic_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# 3. Pragmatic similarity (15% weight)
query_lower = query.lower()
response_lower = response.lower()
pragmatic_score = 0.0

# Question type matching
if query_lower.startswith(('what', 'who', 'where', 'when', 'why', 'how')):
    if any(word in response_lower for word in ['is', 'are', 'was', 'were', 'because', 'due to']):
        pragmatic_score += 0.3

# Keyword overlap
query_words = set(query_lower.split())
response_words = set(response_lower.split())
overlap = len(query_words.intersection(response_words))
pragmatic_score += min(0.4, overlap / max(len(query_words), 1) * 0.4)

# Length appropriateness
length_ratio = len(response) / max(len(query), 1)
if 0.5 <= length_ratio <= 3.0:
    pragmatic_score += 0.3

pragmatic_sim = min(1.0, pragmatic_score)

# Final weighted combination
relevancy_score = (0.6 * semantic_sim + 0.25 * syntactic_sim + 0.15 * pragmatic_sim)
```

**Response Example**:
```json
{
  "metric_name": "response_relevancy_evaluation",
  "actual_value": 0.89,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 167.8,
    "semantic_similarity": 0.94,
    "syntactic_similarity": 0.76,
    "pragmatic_similarity": 0.85,
    "evaluation_method": "multi_dimensional_similarity"
  }
}
```

## RAG Metrics Technical Details

### Model Architecture
- **Primary Model**: `sentence-transformers/all-mpnet-base-v2`
  - 768-dimensional embeddings
  - Trained on diverse text pairs
  - Excellent semantic understanding
- **NER Model**: `en_core_web_trf` (spaCy transformer)
  - High-accuracy named entity recognition
  - Supports 18+ entity types
  - Fallback to regex patterns when unavailable

### Performance Optimizations
- **Batch Encoding**: Multiple texts encoded simultaneously
- **Background Loading**: Models loaded asynchronously
- **Efficient Similarity**: Scikit-learn optimized cosine similarity
- **Memory Management**: Proper cleanup and resource handling

### Error Handling
- **Graceful Degradation**: Fallback methods when components fail
- **Input Validation**: Comprehensive request validation
- **Timeout Protection**: Processing time limits
- **Detailed Logging**: Performance and error tracking

### Use Cases for RAG Metrics

#### Context Precision
- **Document Filtering**: Remove irrelevant passages before generation
- **Retrieval Optimization**: Improve retrieval system precision
- **Cost Reduction**: Reduce tokens sent to LLM by filtering noise

#### Context Recall
- **Knowledge Gap Detection**: Identify missing information in context
- **Retrieval Evaluation**: Assess if retrieval provides sufficient information
- **Response Quality Prediction**: Predict response quality before generation

#### Context Entities Recall
- **Entity-Critical Tasks**: Ensure important entities are available
- **Factual Accuracy**: Verify entity information is present
- **Domain-Specific RAG**: Critical for legal, medical, financial domains

#### Noise Sensitivity
- **Robustness Testing**: Test RAG system robustness to irrelevant information
- **Retrieval Quality**: Assess impact of low-quality retrieval
- **Production Monitoring**: Monitor real-world performance degradation

#### Response Relevancy
- **Answer Quality**: Evaluate how well responses address queries
- **User Satisfaction**: Predict user satisfaction with responses
- **System Comparison**: Compare different RAG implementations

### Supply Chain Risk Detection
`POST /detect/supply-chain-risk`

**Request Format:**
```json
{
  "model_name": "sentence-transformers/all-mpnet-base-v2",
  "plugin_sources": [
    {
      "name": "custom-plugin",
      "origin": "https://github.com/user/plugin-repo",
      "license": "MIT"
    }
  ],
  "dataset_sources": [
    {
      "name": "training-data",
      "origin": "https://huggingface.co/datasets/example",
      "license": "Apache-2.0"
    }
  ]
}
```

**Response Format:**
```json
{
  "metric_name": "supply_chain_risk_evaluation",
  "actual_value": 0.25,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 1250.5,
    "sources_analyzed": 3,
    "risk_level": "low",
    "overall_trust_score": 0.75,
    "min_trust_score": 0.65,
    "avg_trust_score": 0.82,
    "detailed_analyses": [
      {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "license": "apache-2.0",
        "license_score": 1.0,
        "downloads": 2500000,
        "popularity_score": 1.0,
        "origin": "huggingface.co",
        "origin_score": 1.0,
        "source_type": "model"
      }
    ],
    "risk_summary": {
      "overall_risk_score": 0.25,
      "risk_level": "low"
    }
  }
}
```

## Health Endpoint

`GET /health`

Returns service health status:

```json
{
  "status": "ok",
  "message": "Model loaded and working",
  "uptime_seconds": 3600.5
}
```

## Supply Chain Risk Analysis

The supply chain risk analyzer evaluates multiple factors:

### Trust Scoring Factors

#### License Trust Scores (0.0 = risky, 1.0 = trusted)
- **High Trust**: MIT (1.0), Apache-2.0 (1.0), BSD licenses (0.95)
- **Medium Trust**: GPL licenses (0.7), LGPL licenses (0.8)
- **Low Trust**: Proprietary (0.2), Commercial (0.3), Unknown (0.1)

#### Origin Trust Scores
- **Highly Trusted**: Hugging Face (1.0), GitHub (0.9), GitLab (0.85)
- **Academic**: arXiv (0.9), Papers (0.85)
- **Corporate**: TensorFlow/PyTorch (0.9), Microsoft/Google (0.75)
- **Unknown/Private**: Low trust scores (0.1-0.3)

#### Popularity Scores
- **Models**: Based on download counts (100K+ downloads = 1.0)
- **Repositories**: Based on GitHub stars (10K+ stars = 1.0)

### Risk Levels
- **Low Risk (0.0-0.3)**: Well-established sources with trusted licenses
- **Medium Risk (0.3-0.6)**: Mixed trust factors, requires review
- **High Risk (0.6-1.0)**: Untrusted sources, unknown licenses, or suspicious patterns

### Supported Source Types
- **Hugging Face Models**: Automatic metadata extraction and analysis
- **GitHub Repositories**: License, popularity, and maintenance analysis
- **Generic Sources**: Configurable analysis based on provided metadata

## Local Development

Build and run the Docker container:

```bash
make build
make run
```

Test the service:

```bash
./test_script.sh
```

## Deployment

Deploy to Google Cloud Run:

```bash
# Authenticate with Google Cloud
gcloud auth login

# Build and push the Docker image
make push

# Deploy to Cloud Run
gcloud run deploy sentence-transformers \
  --image gcr.io/atri-raime/sentence-transformers:latest \
  --platform managed \
  --region us-central1 \
  --memory 16Gi \
  --cpu 8 \
  --timeout 300s \
  --min-instances 1 \
  --max-instances 10
```

## Model Information

This service uses the "sentence-transformers/all-mpnet-base-v2" model:

- **Model Type**: Sentence transformer based on MPNet
- **Training Data**: Large corpus of sentence pairs from various domains
- **Output**: Dense vector embeddings (768 dimensions)
- **Model Size**: ~420MB
- **Score Range**: 0.0 to 1.0 (cosine similarity normalized to [0,1])

## Resource Requirements

- **Memory**: Minimum 16GB recommended for optimal performance (8GB minimum for RAG metrics)
- **CPU**: At least 8 CPUs for faster inference (4 CPU minimum)
- **Storage**: The model download requires approximately 2GB of storage (sentence-transformers + spaCy models)

## Use Cases

### Semantic Similarity
This service is ideal for:
- **Document similarity**: Compare documents, articles, or paragraphs
- **Question-answer matching**: Find the best answer for a given question
- **Duplicate detection**: Identify similar or duplicate content
- **Semantic search**: Find semantically similar content
- **Content recommendation**: Recommend similar content based on user preferences

### RAG Evaluation
This service helps with:
- **RAG System Evaluation**: Comprehensive assessment of retrieval-augmented generation systems
- **Retrieval Quality Assessment**: Evaluate the quality and relevance of retrieved context
- **Response Quality Measurement**: Assess how well generated responses address queries
- **System Optimization**: Identify areas for improvement in RAG pipelines
- **Production Monitoring**: Monitor RAG system performance in real-time
- **A/B Testing**: Compare different RAG implementations and configurations

### Supply Chain Risk Analysis
This service helps with:
- **Model Security Assessment**: Evaluate risks of third-party models
- **Plugin Vetting**: Analyze security risks of external plugins
- **Dataset Validation**: Assess trustworthiness of training datasets
- **Compliance Checking**: Ensure sources meet security standards
- **Risk Reporting**: Generate detailed risk assessments for auditing

## Notes on Model Behavior

The all-mpnet-base-v2 model:
- Performs well across various domains and text types
- Handles both short phrases and longer paragraphs effectively
- Provides robust semantic understanding beyond keyword matching
- Works well for cross-lingual similarity (though primarily trained on English) 