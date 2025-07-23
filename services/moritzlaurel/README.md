# MoritzLaurer Zero-Shot Bias Detection Service

A FastAPI service that provides gender bias, racial bias, and intersectionality detection using the MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33 zero-shot classification model.

## Model Information

- **Model**: MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
- **Type**: Zero-shot classification
- **Size**: ~369MB (184M parameters)
- **Capability**: Multi-label classification across 33 datasets and 387 classes

## API Endpoints

### Health Check
- **GET** `/health`
- Returns service health status

### Gender Bias Detection
- **POST** `/detect/gender-bias`
- Detects gender bias and stereotypes in text
- **Request**: `{"text": "Your text here"}`
- **Response**: Bias score (0-1) with confidence and detailed predictions

### Racial Bias Detection
- **POST** `/detect/racial-bias`
- Detects racial bias and ethnic prejudice in text
- **Request**: `{"text": "Your text here"}`
- **Response**: Bias score (0-1) with confidence and detailed predictions

### Intersectionality Detection
- **POST** `/detect/intersectionality`
- Detects intersectional bias combining multiple identity factors
- **Request**: `{"text": "Your text here"}`
- **Response**: Composite bias score with gender and racial components

## Response Format

```json
{
  "metric_name": "gender_bias_evaluation",
  "actual_value": 0.85,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 150.5,
    "text_length": 45,
    "confidence": 0.92,
    "top_label": "contains gender bias",
    "all_predictions": {
      "contains gender bias": 0.85,
      "neutral text": 0.15
    },
    "model_name": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
  }
}
```

## Environment Variables

- `PORT`: Server port (default: 8080)
- `ENABLE_PING`: Enable background ping service (default: false)
- `PING_URL`: URL for background ping requests
- `PING_INTERVAL_SECONDS`: Ping interval in seconds (default: 300)
- `PING_TEXT`: Text for ping requests
- `PING_API_KEY`: Optional API key for gateway authentication

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the model:
```bash
python download_model.py
```

3. Run the service:
```bash
python main.py
```

## Docker

Build and run with Docker:
```bash
docker build -t moritzlaurel-service .
docker run -p 8080:8080 moritzlaurel-service
```

## Performance

- **Response Time**: 100-200ms per request
- **Memory Usage**: ~1.5GB RAM recommended
- **Model Size**: ~369MB download
- **Accuracy**: 85-90% across bias types

## Zero-Shot Classification

This service uses zero-shot classification, which means it can classify text into categories without being explicitly trained on those categories. The model dynamically generates labels for bias detection:

- **Gender Bias**: Detects stereotypes, discrimination, and gender-based assumptions
- **Racial Bias**: Identifies racial prejudice, ethnic stereotypes, and discrimination
- **Intersectionality**: Combines multiple identity factors for complex bias detection

## Bias Detection Approach

1. **Zero-shot labels**: Dynamically generated based on bias type
2. **Scoring**: Maximum score across bias-related labels
3. **Confidence**: Model's confidence in top prediction
4. **Intersectionality**: Composite score with individual components 