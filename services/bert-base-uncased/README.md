# BERT Base Uncased Service

This service evaluates text similarity and relevance using the BERT Base Uncased model. It analyzes two key components:

1. **Semantic Similarity** - Using BERT embeddings to measure similarity between context and response
2. **Relevance Score** - Contextual relevance between the input texts
3. **Novelty Score** - Derived from semantic distance to measure creative deviation

## Features

- Evaluates text similarity using BERT Base Uncased embeddings
- Provides similarity and relevance scores
- Returns standardized evaluation format consistent with other metrics
- Simple API with no configuration required

## API

### Health Check

```
GET /health
```

Verifies the service is running and the model is loaded correctly.

### Creativity Detection

```
POST /detect/creativity
```

#### Request Body

```json
{
  "context": "Context text for comparison",
  "response": "Response text to evaluate"
}
```

#### Response

```json
{
  "metric_name": "creativity_evaluation",
  "actual_value": 0.68,
  "actual_value_type": "float",
  "relevance_score": 0.72,
  "novelty_score": 0.45,
  "others": {
    "processing_time": 0.25
  }
}
```

## How It Works

The creativity score is calculated using a weighted combination of:

1. **Similarity Score** (50%): Measures semantic similarity between context and response using BERT embeddings and cosine similarity.

2. **Novelty Score** (50%): Derived from semantic distance to measure creative deviation from the context while maintaining relevance.

## Model Information

- **Model**: `bert-base-uncased`
- **Fallback**: BERTScore for semantic evaluation (if available)

## Local Development

### Prerequisites

- Python 3.10+
- Docker (for containerized development)

### Setup

1. Clone the repository
2. Navigate to the bert-base-uncased service directory:
   ```
   cd services/bert-base-uncased
   ```
3. Set up a virtual environment:
   ```
   make dev
   ```

### Build and Run

Build the Docker image:
```
make build
```

Run the service:
```
make run
```

Access the service at `http://localhost:8080`

## Deployment

This service is deployed to Google Cloud Run using the GitHub Actions workflow in `.github/workflows/bert_base_uncased_deploy.yml`.

### Environment Variables

- `ENVIRONMENT` - Deployment environment (production, staging, development)
- `PORT` - Port for the service to listen on (default: 8080)
- `HOST` - Host to bind to (default: 0.0.0.0)

## License

See the [LICENSE](../../LICENSE) file in the root directory of this repository. 