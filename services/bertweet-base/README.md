# BERTweet-Base Service

This service evaluates the tone/sentiment of text using the BERTweet sentiment analysis model.

## Overview

The BERTweet-Base service analyzes text to determine its emotional tone or sentiment. It uses the BERTweet model, which is specifically trained on Twitter data for sentiment analysis.

The service returns a continuous score between 0 and 1:
- Higher scores indicate more positive sentiment
- Lower scores indicate more negative sentiment
- The model also provides a raw label (POS, NEG, NEU) for categorical interpretation

## Endpoint

`POST /detect/response_tone`

### Request Format

```json
{
  "text": "The text content to analyze for tone/sentiment"
}
```

### Response Format

```json
{
  "metric_name": "response_tone_evaluation",
  "actual_value": 0.85,
  "actual_value_type": "float",
  "others": {
    "inference_time_ms": 142.5,
    "text_length": 45,
    "raw_label": "POS",
    "raw_score": 0.85
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
gcloud run deploy bertweet-base \
  --image gcr.io/atri-raime/bertweet-base:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s \
  --min-instances 1 \
  --max-instances 10
```

## Model Information

This service uses the "finiteautomata/bertweet-base-sentiment-analysis" model:

- **Model Type**: BERTweet-based sentiment classifier
- **Training Data**: Twitter data
- **Output Classes**: POS (positive), NEG (negative), NEU (neutral)
- **Model Size**: ~500MB
- **Score Range**: 0.0 to 1.0 (higher values indicate more positive tone)

## Resource Requirements

- Memory: Minimum 4GB recommended
- CPU: At least 2 CPUs
- Storage: The model download requires approximately 500MB of storage

## Notes on Model Behavior

The BERTweet-based sentiment analysis:
- Works particularly well on informal text and social media content
- Handles emojis, hashtags, and Twitter-style text effectively
- May be less accurate on formal or technical text compared to general-purpose models 