# Flan-T5 Service

This service evaluates factual consistency between text and context using the Flan-T5 model with Vectara's hallucination evaluation capabilities.

## Overview

The Flan-T5 service analyzes whether the information in a given text aligns with the provided context. It detects potential hallucinations or fabricated facts not supported by the context.

The service uses Vectara's hallucination evaluation model with Flan-T5 tokenizer, which is designed to detect whether a text contains unsupported claims. The model outputs a confidence score between 0 and 1:
- Higher scores indicate better factual consistency
- Lower scores suggest potential hallucinations or inconsistencies

## Endpoint

`POST /detect/factual_consistency`

### Request Format

```json
{
  "text": "The text that needs to be evaluated for factual consistency",
  "context": "The context against which the text should be evaluated"
}
```

### Response Format

```json
{
  "metric_name": "factual_consistency_evaluation",
  "actual_value": 0.85,
  "actual_value_type": "float",
  "others": {
    "inference_time_ms": 425.67,
    "text_length": 156,
    "context_length": 342
  }
}
```

## Health Endpoint

`GET /health`

Returns service health status:

```json
{
  "status": "ok"
}
```

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
gcloud run deploy flan-t5 \
  --image gcr.io/atri-raime/flan-t5:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s \
  --min-instances 1 \
  --max-instances 10
```

## Resource Requirements

This service requires significant resources due to the size of the model:
- Memory: Minimum 4GB recommended
- CPU: At least 2 CPUs
- Storage: The model download requires approximately 1GB of storage

## Notes on Model Behavior

The Flan-T5 based factual consistency evaluation:
- Works best when comparing specific factual claims against clear context
- May give less reliable results for highly abstract or subjective content
- Handles short to medium-length texts better than very long ones 