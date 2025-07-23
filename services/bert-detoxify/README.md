# Toxicity Detection Service

A FastAPI-based service for detecting toxic content in text using the Detoxify model from Unitary AI.

## Overview

This service provides toxicity detection capabilities by analyzing text for various types of toxic content:
- Toxicity
- Severe Toxicity
- Obscene
- Identity Attack
- Insult
- Threat

## Features

- Real-time toxicity detection
- Multiple toxicity categories
- Configurable thresholds
- Health monitoring
- Debug endpoints
- Docker support
- Cloud Run deployment ready

## API Endpoints

### Health Check
```http
GET /health
```
Returns the service health status and model loading state.

### Debug Status
```http
GET /debug/status
```
Returns detailed service status including model loading progress and cache information.

### Toxicity Detection
```http
POST /detect/toxicity
```

Request body:
```json
{
  "input_data": {
    "llm_input_query": "string",
    "llm_input_context": "string",
    "llm_output": "string"
  },
  "config_input": {
    "threshold": 0.5,
    "custom_labels": ["non-toxic", "toxic"],
    "label_thresholds": [0, 1]
  }
}
```

Response:
```json
{
  "metric_name": "toxicity_evaluation",
  "actual_value": 0.75,
  "actual_value_type": "float",
  "metric_labels": ["toxic"],
  "threshold": ["Fail"],
  "threshold_score": 0.5,
  "others": {
    "raw_label": "toxic",
    "raw_score": 0.75,
    "inference_time_ms": 150.2
  }
}
```

## Model Details

The service uses the Detoxify model from Unitary AI, which is a RoBERTa-based model fine-tuned for toxicity detection. The model is loaded during service startup and cached for subsequent requests.

## Deployment

### Local Development

1. Build the Docker image:
```bash
docker build -t toxicity-service -f services/toxicity/Dockerfile services/toxicity
```

2. Run the container:
```bash
docker run -p 8080:8080 toxicity-service
```

### Cloud Run Deployment

The service is configured for deployment to Google Cloud Run with the following specifications:

- Memory: 8GB (configurable)
- CPU: 4 cores (configurable)
- Timeout: 600 seconds
- Min instances: 1
- Max instances: 10
- Concurrency: 80 requests per instance

Environment variables:
- `TRANSFORMERS_CACHE=/tmp/cache`
- `HF_HOME=/tmp/cache`
- `IN_DOCKER=true`
- `TORCH_HOME=/tmp/cache`

## Performance Considerations

1. **Model Loading**:
   - Model is pre-downloaded during build time
   - Loaded in background during service startup
   - Cached for subsequent requests
   - Cache directories are properly configured in both build and runtime stages

2. **Resource Requirements**:
   - Minimum 8GB RAM recommended
   - 4 CPU cores recommended
   - 600s timeout for initial model load

3. **Caching**:
   - Model cache directory: `/tmp/cache`
   - Hugging Face cache directory: `/tmp/cache`
   - Torch cache directory: `/tmp/cache`
   - All cache directories are created during build and preserved in runtime

## Health Checks

The service implements health checks to monitor:
- Model loading status
- Service availability
- Cache directory status
- Memory usage

## Error Handling

The service handles various error scenarios:
- Model loading failures
- Invalid input
- Timeout conditions
- Resource constraints
- Cache directory issues

## Monitoring

The service provides monitoring through:
- Health check endpoint
- Debug status endpoint
- Detailed logging
- Performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 