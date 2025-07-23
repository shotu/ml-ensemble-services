# NLTK Analysis Service

This service provides comprehensive text analysis capabilities using NLTK and machine learning libraries. It combines functionality from multiple text analysis services into a single, efficient service.

## Features

- **Readability Analysis**: Evaluates text readability using Flesch Reading Ease score
- **Clarity Analysis**: Combines readability, conciseness, and diversity metrics
- **Diversity Analysis**: Measures lexical diversity using Type-Token Ratio (TTR)
- **Creativity Analysis**: Analyzes linguistic creativity through diversity metrics
- **BLEU Score**: Computes BLEU score between reference and predicted texts
- **Compression Score**: Calculates compression ratio between texts
- **Cosine Similarity**: Measures semantic similarity using Word2Vec embeddings
- **Fuzzy Score**: Measures string similarity using rapidfuzz for fuzzy matching
- **ROUGE Score**: Measures text similarity using ROUGE score
- **METEOR Score**: Measures text similarity using METEOR metric with semantic matching

## API Endpoints

### Health Check
```
GET /health
```
Response:
```json
{
    "status": "ok"
}
```

### Text Analysis Endpoints

#### Readability Analysis
```
POST /detect/readability
```
Request:
```json
{
    "text": "The text to analyze for readability"
}
```
Response:
```json
{
    "metric_name": "readability_evaluation",
    "actual_score": 0.75,
    "processing_time": 0.05
}
```

#### Clarity Analysis
```
POST /detect/clarity
```
Request:
```json
{
    "text": "The text to analyze for clarity"
}
```
Response:
```json
{
    "metric_name": "clarity_evaluation",
    "actual_score": 0.75,
    "token_count": 42,
    "readability_score": 0.75,
    "conciseness_score": 0.8,
    "diversity_score": 0.65,
    "processing_time": 0.05
}
```

#### Diversity Analysis
```
POST /detect/diversity
```
Request:
```json
{
    "text": "The text to analyze for lexical diversity"
}
```
Response:
```json
{
    "metric_name": "diversity_evaluation",
    "actual_score": 0.65,
    "token_count": 120,
    "unique_token_count": 78,
    "processing_time": 0.15
}
```

#### Creativity Analysis
```
POST /detect/creativity-similarity
```
Request:
```json
{
    "text": "The text to analyze for creativity"
}
```
Response:
```json
{
    "metric_name": "creativity_evaluation",
    "actual_value": 0.68,
    "actual_value_type": "float",
    "others": {
        "token_count": 120,
        "unique_token_count": 82,
        "processing_time": 0.12
    }
}
```

### Computation Endpoints

#### BLEU Score
```
POST /compute/bleu-score
```
Request:
```json
{
    "references": ["The quick brown fox jumps over the lazy dog"],
    "predictions": ["A quick brown fox leaps over the lazy dog"]
}
```
Response:
```json
{
    "metric_name": "bleu_score_evaluation",
    "actual_score": 0.7524,
    "actual_value_type": "float",
    "others": {
        "processing_time": 0.05
    }
}
```

#### Compression Score
```
POST /compute/compression-score
```
Request:
```json
{
    "references": ["This is a long reference text"],
    "predictions": ["Short text"]
}
```
Response:
```json
{
    "metric_name": "compression_score_evaluation",
    "actual_score": 0.4286,
    "actual_value_type": "float",
    "others": {
        "processing_time": 0.03
    }
}
```

#### Cosine Similarity
```
POST /compute/cosine-similarity
```
Request:
```json
{
    "references": ["The weather is nice today"],
    "predictions": ["Today has beautiful weather"]
}
```
Response:
```json
{
    "metric_name": "cosine_similarity_evaluation",
    "actual_score": 0.8542,
    "actual_value_type": "float",
    "others": {
        "processing_time": 0.12
    }
}
```

#### Fuzzy Score
```
POST /compute/fuzzy-score
```
Request:
```json
{
    "references": ["The quick brown fox jumps over the lazy dog"],
    "predictions": ["The fast brown fox leaps over the sleepy dog"]
}
```
Response:
```json
{
    "metric_name": "fuzzy_score",
    "actual_score": 0.8235,
    "actual_value_type": "float",
    "others": {
        "processing_time": 0.0012
    }
}
```

### 9. ROUGE Score

**Endpoint**: `POST /compute/rouge-score`

**Description**: Computes ROUGE score between reference and prediction texts using ROUGE-L metric.

**Request Body**:
```json
{
  "references": ["The quick brown fox jumps over the lazy dog"],
  "predictions": ["The fast brown fox leaps over the sleepy dog"]
}
```

**Response**:
```json
{
  "metric_name": "rouge_score",
  "actual_score": 0.7368,
  "actual_value_type": "float",
  "others": {
    "precision": 0.7778,
    "recall": 0.7000,
    "fmeasure": 0.7368,
    "processing_time": 0.0015
  }
}
```

### 10. METEOR Score

**Endpoint**: `POST /compute/meteor-score`

**Description**: Computes METEOR score between reference and prediction texts using METEOR metric with semantic matching.

**Request Body**:
```json
{
  "references": ["The quick brown fox jumps over the lazy dog"],
  "predictions": ["The fast brown fox leaps over the sleepy dog"]
}
```

**Response**:
```json
{
  "metric_name": "meteor_score_evaluation",
  "actual_score": 0.8125,
  "actual_value_type": "float",
  "others": {
    "processing_time": 0.0023
  }
}
```

## Score Descriptions

### Readability Score (0-1)
- Based on Flesch Reading Ease score
- Normalized to 0-1 range
- Higher scores indicate more readable text

### Clarity Score (0-1)
Combines three components:
1. **Readability**: Flesch Reading Ease score
2. **Conciseness**: Based on sentence and word length
3. **Diversity**: Type-Token Ratio of non-stop words

### Diversity Score (0-1)
- Type-Token Ratio (TTR) of non-stop words
- Measures lexical diversity
- Higher scores indicate more diverse vocabulary

### Creativity Score (0-1)
- Based on linguistic diversity metrics
- Uses Type-Token Ratio of filtered tokens
- Higher scores indicate more creative language use

### BLEU Score (0-1)
- Measures n-gram overlap between reference and prediction texts. Higher scores indicate better similarity.

### Compression Score (0-∞)
- Ratio of prediction length to reference length. Values > 1 indicate expansion, < 1 indicate compression.

### Cosine Similarity (-1 to 1)
- Measures semantic similarity using Word2Vec embeddings. Range: -1 to 1, where 1 is most similar.

### Fuzzy Score (0-1)
- Measures string similarity using rapidfuzz for fuzzy matching. Range: 0 to 1, where 1 is exact match.

### ROUGE Score (0-1)
- Measures text similarity using ROUGE-L metric, focusing on longest common subsequences. Range: 0 to 1, where 1 is perfect match.

### METEOR Score (0-1)
- Measures text similarity using METEOR metric with semantic matching. Range: 0 to 1, where 1 is perfect match.

## Development

### Prerequisites
- Python 3.10+
- Docker (for containerized development)

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```bash
python download_model.py
```

4. Run the service:
```bash
uvicorn main:app --reload
```

### Docker Build and Run

Build the Docker image:
```bash
make build
```

Run the service:
```bash
make run
```

## Configuration

The service uses environment variables for configuration:
- `PORT`: Port number for the service (default: 8080)
- `HOST`: Host to bind to (default: 0.0.0.0)
- `NLTK_DATA`: Path to NLTK data directory (default: /app/nltk_data)

## Dependencies

- fastapi==0.104.1
- uvicorn==0.24.0
- nltk==3.8.1
- pydantic==2.4.2
- textstat==0.7.3
- python-multipart==0.0.6
- gensim==4.3.2
- scikit-learn==1.3.2
- numpy==1.24.4
- rapidfuzz==3.5.2

## NLTK Data Requirements

The service automatically downloads required NLTK packages:
- punkt (sentence tokenization)
- stopwords (stopword filtering)
- cmudict (syllable counting) 