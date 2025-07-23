# T5-Base Service

This service evaluates grammatical correctness and detects token bloat DoS attacks using the T5-Base model. It provides standardized evaluation formats with detailed analysis.

## Overview

The T5-Base service provides two main capabilities:

1. **Grammatical Correctness Analysis**: Analyzes text for grammatical correctness by generating corrected versions and comparing them with the original text
2. **Token Bloat DoS Detection**: Identifies potential denial-of-service attacks through excessive token generation or processing latency

The grammar service uses a sophisticated scoring algorithm that considers multiple error types and provides detailed correction analysis, producing a continuous score between 0 and 1 where higher scores indicate better grammatical correctness.

## API Endpoints

### Grammar Detection
`POST /detect/grammar`

### Request Format

```json
{
  "text": "The text that needs to be evaluated for grammatical correctness"
}
```

### Response Format

```json
{
  "metric_name": "grammatical_correctness_evaluation",
  "actual_value": 0.85,
  "actual_value_type": "float",
  "others": {
    "inference_time_ms": 425.67,
    "text_length": 156,
    "correction_details": {
      "basic_similarity": 0.95,
      "corrections": {
        "punctuation": 1,
        "spelling": 0,
        "grammar": 0,
        "word_order": 0,
        "subject_verb": 0,
        "tense": 0,
        "article": 0,
        "preposition": 0,
        "total_changes": 1
      },
      "penalty": 0.15,
      "error_density": 0.1,
      "corrected_text": "The corrected version of the input text"
    }
  }
}
```

### Token Bloat DoS Detection
`POST /detect/token-bloat-dos`

**Request Format:**
```json
{
  "text": "Input text to analyze for token bloat patterns",
  "max_output_length": 512,
  "expected_ratio": 3.0
}
```

**Response Format:**
```json
{
  "metric_name": "token_bloat_dos_evaluation",
  "actual_value": 0.65,
  "actual_value_type": "float",
  "others": {
    "explanation": "Medium risk of token bloat DoS attack detected (score: 0.650)",
    "risk_level": "medium",
    "warnings": [
      "Excessive token generation detected",
      "Output/input ratio (8.5) exceeds expected by 2x"
    ],
    "analysis_details": {
      "input_text": "Input text to analyze for token bloat patterns",
      "output_text": "Generated response text...",
      "input_tokens": 10,
      "output_tokens": 85,
      "token_ratio": 8.5,
      "generation_time_ms": 2500.0,
      "tokens_per_second": 34.0,
      "expected_ratio": 3.0,
      "bloat_score": 0.7,
      "latency_score": 0.6,
      "combined_score": 0.65,
      "total_processing_time": 2.5
    },
    "processing_time": 2.5
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

## Token Bloat DoS Detection

The token bloat DoS detector analyzes potential denial-of-service attacks through:

### Detection Methods

#### 1. Token Bloat Analysis
- **Token Ratio Monitoring**: Compares output/input token ratios against expected baselines
- **Absolute Output Size**: Detects unusually large outputs that could consume resources
- **Generation Efficiency**: Monitors tokens generated per second for anomalies
- **Input/Output Disproportion**: Identifies cases where tiny inputs generate massive outputs

#### 2. Latency DoS Analysis
- **Absolute Latency Thresholds**: Flags generation times exceeding normal ranges
- **Statistical Deviation**: Compares current latency against historical baselines
- **Efficiency Scoring**: Analyzes time-per-token generation patterns
- **Baseline Learning**: Maintains rolling averages for anomaly detection

### Risk Scoring

#### Combined Risk Score (0.0-1.0)
- **Bloat Score (60% weight)**: Based on token generation patterns
- **Latency Score (40% weight)**: Based on processing time patterns

#### Risk Levels
- **Minimal (0.0-0.1)**: Normal operation, no concerning patterns
- **Low (0.1-0.4)**: Minor inefficiencies, monitoring recommended
- **Medium (0.4-0.7)**: Concerning patterns detected, investigation needed
- **High (0.7-1.0)**: Likely DoS attack, immediate action required

#### Warning Indicators
- **Excessive Token Generation**: Output significantly exceeds normal patterns
- **Suspicious Generation Latency**: Processing time exceeds thresholds
- **Ratio Violations**: Output/input ratio exceeds expected by 2x or more
- **Generation Timeout**: Processing exceeds 10+ seconds

### Configuration Parameters
- **max_output_length**: Maximum allowed output tokens (default: 512)
- **expected_ratio**: Expected output/input token ratio (default: 3.0)
- **Statistical Window**: Rolling window for baseline calculations (100 samples)

## Scoring System

The service uses an enhanced scoring system that considers multiple factors:

### Error Categories
- Punctuation errors
- Spelling errors
- Grammar errors
- Word order errors
- Subject-verb agreement errors
- Tense errors
- Article usage errors
- Preposition errors

### Error Weights
Each error type has a specific weight in the final score:
- Subject-verb agreement: 0.4 (40% impact)
- Grammar: 0.35 (35% impact)
- Tense: 0.3 (30% impact)
- Spelling: 0.25 (25% impact)
- Word order: 0.2 (20% impact)
- Preposition: 0.2 (20% impact)
- Punctuation: 0.15 (15% impact)
- Article: 0.15 (15% impact)

### Scoring Algorithm
1. Basic similarity score between original and corrected text
2. Error detection and categorization
3. Weighted penalty calculation with exponential scaling
4. Error density consideration
5. Non-linear final score calculation

### Score Interpretation
- 1.0: Perfect grammar (no corrections needed)
- 0.8-0.99: Minor errors (mostly correct)
- 0.6-0.79: Moderate errors (needs improvement)
- 0.4-0.59: Significant errors (needs major improvement)
- 0.0-0.39: Major errors (needs complete revision)

## Local Development

Build and run the Docker container:

```bash
docker build -t t5-base-service .
docker run -p 8080:8080 t5-base-service
```

Test the service:

```bash
curl -X POST "http://localhost:8080/detect/grammar" \
  -H "Content-Type: application/json" \
  -d '{"text": "This are a test sentence with grammar error."}'

curl -X POST "http://localhost:8080/detect/token-bloat-dos" \
  -H "Content-Type: application/json" \
  -d '{"text": "Analyze this text for potential DoS patterns", "max_output_length": 256, "expected_ratio": 2.0}'
```

## Deployment

Deploy to Google Cloud Run:

```bash
# Authenticate with Google Cloud
gcloud auth login

# Build and push the Docker image
docker build -t gcr.io/atri-raime/t5-base:latest .
docker push gcr.io/atri-raime/t5-base:latest

# Deploy to Cloud Run
gcloud run deploy t5-base \
  --image gcr.io/atri-raime/t5-base:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s \
  --min-instances 1 \
  --max-instances 10
```

## Resource Requirements

This service requires significant resources due to the size of the T5 model:
- Memory: Minimum 4GB recommended
- CPU: At least 2 CPUs
- Storage: The model download requires approximately 1GB of storage

## Model Information

This service uses the `vennify/t5-base-grammar-correction` model, which is a T5 model fine-tuned for grammar correction. The model:
- Takes input text and generates a corrected version
- Provides detailed error analysis and categorization
- Returns a continuous score between 0 and 1 based on multiple factors

## Use Cases

### Grammar Detection
- **Content Quality Assessment**: Evaluate writing quality in documents
- **Educational Tools**: Provide grammar feedback for learning
- **Content Moderation**: Ensure published content meets quality standards
- **Automated Proofreading**: Integrate into writing workflows

### Token Bloat DoS Detection
- **Security Monitoring**: Detect potential DoS attacks on text generation systems
- **Resource Protection**: Prevent excessive resource consumption
- **Performance Optimization**: Identify inefficient text processing patterns
- **Compliance**: Ensure text generation stays within acceptable bounds

## Notes on Model Behavior

The T5-Base grammatical correctness evaluation:
- Works best with complete sentences and paragraphs
- May give less reliable results for very short text fragments
- Handles various types of grammatical errors including punctuation, spelling, and syntax
- Provides detailed breakdown of correction types for analysis

The Token Bloat DoS detection:
- Learns from historical patterns to improve accuracy
- Adapts to different types of text inputs
- Provides detailed analysis for security investigation
- Can be configured for different risk tolerance levels 