# DistilBERT NSFW Text Classifier Service

This service uses the `eliasalbouzidi/distilbert-nsfw-text-classifier` model to detect whether input text is NSFW (Not Safe For Work) or safe.

## Features

- **Fast Inference**: Utilizes a distilled BERT model for quick predictions.
- **High Accuracy**: Achieves an F1 score of approximately 0.974 on the NSFW-Safe-Dataset.
- **Easy Deployment**: Dockerized for seamless deployment.

## API Endpoints

### Health Check

**GET** `/health`

Returns the status of the service.

**Response:**
```json
{
  "status": "ok"
}
```

### NSFW Detection

**POST** `/detect/nsfw`

**Request Body:**
```json
{
  "text": "Your input text here"
}
```

**Response:**
```json
{
  "metric_name": "nsfw_evaluation",
  "actual_value": 0.95,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 45.2,
    "text_length": 25,
    "model_label": "NSFW",
    "model_raw_score": 0.95,
    "model_name": "eliasalbouzidi/distilbert-nsfw-text-classifier"
  }
}
```

- `metric_name`: Always "nsfw_evaluation"
- `actual_value`: NSFW probability score between 0 and 1 (higher values indicate more NSFW content)
- `actual_value_type`: Data type (always "float")
- `others`: Additional information including:
  - `processing_time_ms`: Processing time in milliseconds
  - `text_length`: Length of input text
  - `model_label`: Original model classification ("SAFE" or "NSFW")
  - `model_raw_score`: Raw confidence score from the model
  - `model_name`: Model identifier

## Model Information

- **Model**: `eliasalbouzidi/distilbert-nsfw-text-classifier`
- **Base Model**: DistilBERT
- **Task**: Binary text classification (NSFW vs SAFE)
- **Performance**: F1 score ~0.974

## Local Development

### Prerequisites

- Python 3.10+
- Docker (for containerized development)

### Setup

1. Clone the repository
2. Navigate to the distilbert-nsfw-text-classifier service directory:
   ```bash
   cd services/distilbert-nsfw-text-classifier
   ```
3. Set up a virtual environment:
   ```bash
   make dev
   ```

### Build and Run

Build the Docker image:
```bash
make build
```

Run the service:
```bash
make run
```

Access the service at `http://localhost:8080`

### Development Commands

```bash
make build    # Build the Docker image
make run      # Run the service locally using Docker
make test     # Run tests
make clean    # Clean up Docker resources
make dev      # Set up development environment
make lint     # Run linters
make format   # Format the code
```

## Testing

### Example NSFW Detection

```bash
curl -X POST "http://localhost:8080/detect/nsfw" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a normal, safe text."}'
```

Response:
```json
{
  "metric_name": "nsfw_evaluation",
  "actual_value": 0.9876,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 45.2,
    "text_length": 25,
    "model_label": "SAFE",
    "model_raw_score": 0.9876,
    "model_name": "eliasalbouzidi/distilbert-nsfw-text-classifier"
  }
}
```

### Example NSFW Content

```bash
curl -X POST "http://localhost:8080/detect/nsfw" \
  -H "Content-Type: application/json" \
  -d '{"text": "Some inappropriate content here..."}'
```

Response:
```json
{
  "metric_name": "nsfw_evaluation",
  "actual_value": 0.05,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 45.2,
    "text_length": 29,
    "model_label": "SAFE",
    "model_raw_score": 0.95,
    "model_name": "eliasalbouzidi/distilbert-nsfw-text-classifier"
  }
}
```

## Deployment

This service can be deployed to Google Cloud Run using GitHub Actions workflow.

### Environment Variables

- `PORT` - Port for the service to listen on (default: 8080)
- `HOST` - Host to bind to (default: 0.0.0.0)
- `TRANSFORMERS_CACHE` - Cache directory for transformers models
- `HF_HOME` - Hugging Face cache directory

## Error Handling

The service handles various error scenarios:

- **Model not loaded**: Returns 500 with appropriate error message
- **Invalid input**: Returns 422 for validation errors
- **Inference errors**: Returns 500 with error details

## Performance

- **Model Size**: ~67MB (DistilBERT)
- **Inference Time**: ~50-100ms per request
- **Memory Usage**: ~200-300MB

## License

See the [LICENSE](../../LICENSE) file in the root directory of this repository. 