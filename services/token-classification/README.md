# Token Classification Service

This service detects personal information in text using a specialized token classification model with optimized scoring logic.

## Overview

The token classification service analyzes text to identify personally identifiable information (PII) and sensitive data. It uses an advanced continuous scoring algorithm that provides nuanced risk assessment based on entity types, confidence levels, and text context.

### Model & Architecture

- **Base Model**: `iiiorg/piiranha-v1-detect-personal-information`
- **Hybrid Detection**: Combines ML model predictions with conservative regex patterns
- **Confidence Filtering**: Filters model results with minimum 0.3 confidence threshold
- **Overlap Resolution**: Removes overlapping entities, keeping highest confidence matches

### Scoring Algorithm

The service calculates continuous scores (0.0-1.0) using a sophisticated multi-factor approach:

#### Entity Weight Hierarchy
```python
# Critical PII (highest risk)
SOCIALNUM: 1.0          # Social Security Numbers
CREDITCARDNUMBER: 0.95  # Credit Card Numbers
PASSPORT: 0.9           # Passport Numbers
DRIVERLICENSE: 0.85     # Driver License Numbers

# High sensitivity PII
TELEPHONENUM: 0.7       # Phone Numbers
EMAIL: 0.65             # Email Addresses

# Medium sensitivity PII
PERSON: 0.5             # Person Names
ZIPCODE: 0.4            # ZIP Codes
STREET: 0.35            # Street Addresses
BUILDINGNUM: 0.3        # Building Numbers

# Lower sensitivity (geographic/organizational)
ORGANIZATION: 0.2       # Organizations
CITY: 0.15              # City Names
STATE: 0.1              # State Names
COUNTRY: 0.05           # Country Names
```

#### Scoring Formula
1. **Base Weighted Score**: `Σ(entity_weight × model_confidence) / max_possible_weight`
2. **Density Adjustment**: Light boost for high entity density (>1 entity per 20 words)
3. **Granularity Enhancement**: Small variance based on entity count for better continuity
4. **Minimum Threshold**: Ensures 0.05 minimum for any detected PII

#### Conservative Regex Detection
Additional high-confidence pattern matching for:
- **Email**: `[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}`
- **Phone**: US formats only (555-123-4567, (555) 123-4567, 555.123.4567)
- **Validation**: Length checks, structure validation, fake number detection

## API Endpoints

### Detection Endpoint

`POST /detect/data_leakage`

#### Request Format
```json
{
  "text": "Contact John Smith at john.smith@example.com or call (555) 123-4567"
}
```

#### Response Format
```json
{
  "metric_name": "data_leakage_evaluation",
  "actual_value": 0.842,
  "actual_value_type": "float",
  "others": {
    "inference_time_ms": 756.32,
    "detected_entities": [
      {
        "word": "john.smith@example.com",
        "label": "I-EMAIL",
        "confidence": 0.85,
        "start": 29,
        "end": 51
      },
      {
        "word": "(555) 123-4567",
        "label": "I-TELEPHONENUM", 
        "confidence": 0.8,
        "start": 60,
        "end": 74
      }
    ],
    "num_entities": 2,
    "entity_types_detected": ["EMAIL", "TELEPHONENUM"],
    "model_entities": 1,
    "regex_entities": 1,
    "text_word_count": 12
  }
}
```

### Health Endpoint

`GET /health`
```json
{
  "status": "ok"
}
```

## Optimization Features

### False Positive Prevention
- **Conservative Patterns**: Only high-confidence regex matches
- **Context Filtering**: Removes low-confidence model predictions (<0.3)
- **Overlap Resolution**: Prevents double-counting of same entities
- **Validation Logic**: Email/phone structure and fake number detection

### Performance Optimizations
- **Efficient Processing**: ~800ms average response time
- **Memory Management**: Transformer cache optimization
- **Minimal Dependencies**: Streamlined imports and processing

### Score Distribution
The optimized algorithm provides better continuous scoring:
- **0.0**: No PII detected (clean text)
- **0.05-0.3**: Low-risk entities (locations, organizations)
- **0.3-0.7**: Medium-risk entities (names, addresses)
- **0.7-1.0**: High-risk entities (SSN, credit cards, emails, phones)

## Local Development

### Build and Run
```bash
# Build Docker container
make build

# Run locally
make run

# Test the service
./test_script.sh
```

### Testing Examples
```bash
# No PII - should return 0.0
curl -X POST "http://localhost:8080/detect/data_leakage" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a generic text about technology trends."}'

# Email detection - should return ~0.9
curl -X POST "http://localhost:8080/detect/data_leakage" \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact support at help@example.com"}'

# High PII - should return 1.0
curl -X POST "http://localhost:8080/detect/data_leakage" \
  -H "Content-Type: application/json" \
  -d '{"text": "SSN: 123-45-6789, Credit Card: 4532-1234-5678-9012"}'
```

## Deployment

### Google Cloud Run
```bash
# Authenticate
gcloud auth login

# Build and push
make push

# Deploy
gcloud run deploy token-classification \
  --image [IMAGE_URL] \
  --platform managed \
  --allow-unauthenticated
```

## Service Characteristics

- **Reliability**: Eliminates false positive explosion from v2.0
- **Accuracy**: Conservative approach ensures high precision
- **Performance**: Consistent sub-second response times
- **Scalability**: Stateless design suitable for Cloud Run auto-scaling
- **Security**: No data persistence, request-response only processing 