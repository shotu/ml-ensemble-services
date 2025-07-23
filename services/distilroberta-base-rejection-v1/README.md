# Model Rejection Detection Service using ProtectAI/distilroberta-base-rejection-v1

This service exposes a REST API to detect whether a given text contains a rejection or refusal message.

## API Endpoints

### GET `/health`
Returns the health status of the service.

**Response:**
```json
{
  "status": "ok"
}
```

### POST `/detect/refusal`
Detects if the input text contains a model refusal or rejection message.

**Request Body:**
```json
{
  "text": "I'm sorry, I cannot help with that request."
}
```

**Response:**
```json
{
  "metric_name": "model_refusal_evaluation",
  "actual_value": 0.95,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 45.2,
    "text_length": 42,
    "model_label": "REJECTION",
    "model_raw_score": 0.95,
    "model_name": "ProtectAI/distilroberta-base-rejection-v1"
  }
}
```

## Model

This service uses the Hugging Face model: **ProtectAI/distilroberta-base-rejection-v1**
which is a DistilRoBERTa model fine-tuned for rejection/refusal detection.

## Test Results Summary

| Test Case | Input Text | Model Label | Actual Value | Interpretation |
|-----------|------------|-------------|--------------|----------------|
| **Clear Refusal** | "I apologize, but I cannot assist with that request." | `REJECTION` | **0.9994** | ✅ **Very High Rejection** |
| **Unable to Help** | "I am not able to help with that." | `REJECTION` | **0.9851** | ✅ **High Rejection** |
| **Helpful Response** | "I would be happy to help you with that request." | `NORMAL` | **0.0005** | ✅ **Very Low Rejection** |
| **Neutral Info** | "The weather today is sunny with a temperature of 75 degrees." | `NORMAL` | **0.000029** | ✅ **Extremely Low Rejection** |
| **Polite Refusal** | "I am sorry, I cannot provide information on that topic." | `REJECTION` | **0.9988** | ✅ **Very High Rejection** |

## Key Observations

- **Excellent Performance**: The model correctly identifies refusal/rejection patterns with very high confidence (98-99%)
- **Case Sensitivity Fix Working**: The scoring logic with `label.upper() == "REJECTION"` is working correctly:
  - `REJECTION` labels → High actual_value (rejection probability)
  - `NORMAL` labels → Low actual_value (rejection probability)
- **Response Structure**: Perfectly matches the standard ecosystem format with:
  - `metric_name`: "model_refusal_evaluation"
  - `actual_value`: Rejection probability (0-1)
  - `others`: Detailed metadata including processing time, model info
- **Performance**: Processing times are fast (~50ms after first request)
- **Model Labels**: The model returns:
  - `"REJECTION"` for refusal/rejection content
  - `"NORMAL"` for regular/helpful content

The service is working perfectly and ready for integration into the API Gateway! The model shows excellent discrimination between refusal patterns and normal responses, making it highly suitable for detecting when AI models refuse to answer requests.

## Usage

### Build Docker image:
```bash
docker build -t rejection-detection-service .
```

### Run Docker container:
```bash
docker run -p 8080:8080 rejection-detection-service
```

### Test endpoint:
```bash
curl -X POST "http://localhost:8080/detect/refusal" \
  -H "Content-Type: application/json" \
  -d '{"text":"I am sorry, I cannot help with that."}'
```

## Response Structure

- **metric_name**: Always "model_refusal_evaluation"
- **actual_value**: Float between 0 and 1 indicating likelihood of rejection/refusal (higher = more likely to be a rejection)
- **actual_value_type**: Always "float"
- **others**: Additional metadata including:
  - **processing_time_ms**: Time taken to process the request
  - **text_length**: Length of input text
  - **model_label**: Raw label from the model
  - **model_raw_score**: Raw confidence score from the model
  - **model_name**: Name of the underlying model

## Development

### Local Development:
```bash
# Install dependencies
pip install -r requirements.txt

# Download model (optional, will be done during Docker build)
python download_model.py

# Run the service
python main.py
```

### Using Makefile:
```bash
# Build the service
make build

# Run the service
make run

# Test the service
make test

# Clean up
make clean
```

## Notes

- The model and tokenizer are downloaded and cached at `/app/model_cache` during image build
- The service uses FastAPI and runs on port 8080 by default
- The scoring logic converts model outputs to a consistent "rejection probability" where higher values indicate higher likelihood of refusal/rejection 