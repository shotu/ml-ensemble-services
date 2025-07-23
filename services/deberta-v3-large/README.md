# DeBERTa v3 Large Service

This service provides text classification using the DeBERTa v3 Large cross-encoder model for evaluating relationships between two text inputs.

## Overview

The DeBERTa v3 Large Service uses the `cross-encoder/nli-deberta-v3-large` model to analyze the relationship between two pieces of text. It provides a raw score indicating the strength of the relationship without any labeling or threshold logic.

## Model Information

- **Model**: `cross-encoder/nli-deberta-v3-large`
- **Type**: Cross-encoder for natural language inference
- **Output**: Raw confidence scores
- **Use Cases**: Text similarity, relevance scoring, relationship analysis

## API Endpoints

### Health Check
```
GET /health
```

### Evaluate Text Relationship
```
POST /detect/answer_relevance
```

Request body:
```json
{
  "input_text": "What is quantum computing?",
  "output_text": "Quantum computing uses quantum mechanics principles to process information using quantum bits or qubits."
}
```

Response:
```json
{
  "metric_name": "deberta_v3_large_evaluation",
  "actual_value": 4.23,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 450.25,
    "input_text_length": 25,
    "output_text_length": 98,
    "model_name": "cross-encoder/nli-deberta-v3-large"
  }
}
```

## Example curl Commands

### Basic Text Relationship Analysis
```bash
curl -X POST https://deberta-v3-large-production-drnc7zg5yq-uc.a.run.app/detect/answer_relevance \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "What is the capital of France?",
    "output_text": "The capital of France is Paris."
  }'
```

### Complex Text Analysis
```bash
curl -X POST https://deberta-v3-large-production-drnc7zg5yq-uc.a.run.app/detect/answer_relevance \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "Explain the process of photosynthesis in plants",
    "output_text": "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."
  }'
```

### Health Check
```bash
curl -X GET https://deberta-v3-large-production-drnc7zg5yq-uc.a.run.app/health
```

## Understanding Scores

The service returns raw scores from the DeBERTa v3 Large model:
- Higher scores indicate stronger relationships between the texts
- Scores can vary significantly based on text content and length
- No predefined thresholds or labels are applied

## Use Cases

- **Answer Relevance**: Evaluate if an answer addresses a question
- **Text Similarity**: Measure semantic similarity between texts
- **Content Matching**: Assess if content matches requirements
- **Quality Assessment**: Score text relationships for quality control

## Performance

- Processing times typically range from 400-600ms
- Performance depends on text length and complexity
- Optimized for production workloads

## Development

### Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the service:
```bash
uvicorn main:app --reload
```

### Docker

1. Build the Docker image:
```bash
docker build -t deberta-v3-large .
```

2. Run the container:
```bash
docker run -p 8080:8080 deberta-v3-large
```

## Deployment

The service is deployed to Google Cloud Run using the GitHub Actions workflow in `.github/workflows/deberta_v3_large_deploy.yml`.

### Resource Requirements

- **Memory**: 16Gi (recommended)
- **CPU**: 8 cores (recommended)
- **Timeout**: 300 seconds
- **Concurrency**: 80 requests per instance

## Model Behavior

The DeBERTa v3 Large cross-encoder model:
- Processes text pairs jointly for better understanding
- Provides nuanced scoring based on semantic relationships
- Handles various text lengths and complexities
- Optimized for natural language inference tasks 