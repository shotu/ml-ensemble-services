# RoBERTa-large-mnli Faithfulness Evaluation Service

A production-ready microservice for evaluating the faithfulness of LLM-generated responses using Natural Language Inference (NLI) with RoBERTa-large-mnli.

## 🎯 Purpose

This service evaluates how well an LLM's generated response is supported by the provided context. It uses the RoBERTa-large-mnli model to perform entailment analysis, determining if the response is:
- **Entailed** by the context (faithful)
- **Contradicted** by the context (unfaithful)  
- **Neutral** with respect to the context (partially faithful)

## 🚀 Features

- **Faithfulness Evaluation**: Comprehensive analysis using Natural Language Inference
- **Claim Decomposition**: Breaks down responses into individual claims for fine-grained analysis
- **Contradiction Detection**: Identifies contradictory statements
- **Evidence Support Assessment**: Categorizes evidence strength (strong/moderate/weak/insufficient)
- **Production Ready**: Async FastAPI with health checks and monitoring
- **Background Model Loading**: Non-blocking startup with model loading status
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📊 Input Format

The service expects the following input structure:

```json
{
    "llm_input_query": "What is the capital of France?",
    "llm_input_context": "Paris is the capital and largest city of France, located in the north-central part of the country.",
    "llm_output": "The capital of France is Paris, which is also its largest city."
}
```

## 📈 Output Format

```json
{
    "metric_name": "faithfulness",
    "actual_value": 0.95,
    "actual_value_type": "float",
    "others": {
        "processing_time_ms": 245.7,
        "entailment_probability": 0.95,
        "contradiction_probability": 0.02,
        "neutral_probability": 0.03,
        "predicted_label": "entailment",
        "claims_analyzed": 2,
        "average_claim_score": 0.94,
        "claim_details": [
            {
                "claim": "The capital of France is Paris",
                "entailment_prob": 0.96,
                "predicted_label": "entailment"
            },
            {
                "claim": "Paris is the largest city",
                "entailment_prob": 0.92,
                "predicted_label": "entailment"
            }
        ],
        "contradiction_detected": false,
        "evidence_support": "strong",
        "input_lengths": {
            "context_length": 89,
            "response_length": 67,
            "query_length": 29
        },
        "model_name": "roberta-large-mnli",
        "evaluation_method": "natural_language_inference"
    }
}
```

## 🏗️ API Endpoints

### Primary Endpoint

**POST** `/evaluate/faithfulness`
- Evaluates faithfulness of LLM output against context
- Returns comprehensive analysis with claim-level breakdown

### Utility Endpoints

**GET** `/health`
- Health check endpoint
- Returns model loading status

**GET** `/`
- Service information and capabilities

**GET** `/debug/status`
- Debug information including model status and system info

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- 8GB+ RAM (for RoBERTa-large model)
- GPU recommended for faster inference

### Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Download Model**
```bash
python download_model.py
```

3. **Run Service**
```bash
python main.py
```

4. **Test the Service**
```bash
curl -X POST "http://localhost:8080/evaluate/faithfulness" \
     -H "Content-Type: application/json" \
     -d '{
       "llm_input_query": "What is the capital of France?",
       "llm_input_context": "Paris is the capital of France.",
       "llm_output": "The capital of France is Paris."
     }'
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t roberta-mnli-faithfulness .

# Run the container
docker run -p 8080:8080 roberta-mnli-faithfulness
```

### Environment Variables

- `PORT`: Service port (default: 8080)
- `ENABLE_PING`: Enable background health pinging (default: false)
- `PING_URL`: URL to ping for health checks
- `PING_INTERVAL_SECONDS`: Ping interval (default: 300)
- `PING_API_KEY`: Optional API key for ping requests

## 📏 Evaluation Metrics

### Faithfulness Score (0.0 - 1.0)
- **0.8 - 1.0**: Strong evidence support
- **0.6 - 0.8**: Moderate evidence support  
- **0.4 - 0.6**: Weak evidence support
- **0.0 - 0.4**: Insufficient evidence support

### Analysis Components
- **Overall Entailment**: Holistic entailment probability
- **Claim-level Analysis**: Individual claim evaluation
- **Contradiction Detection**: Identifies contradictory statements
- **Evidence Support**: Qualitative assessment of support strength

## 🔬 Technical Details

### Model Information
- **Model**: `roberta-large-mnli`
- **Task**: Natural Language Inference (NLI)
- **Labels**: 
  - 0: Contradiction
  - 1: Neutral
  - 2: Entailment
- **Max Sequence Length**: 512 tokens

### Claim Decomposition
The service automatically decomposes responses into factual claims using:
- Sentence boundary detection
- Factual indicator patterns
- Opinion vs. fact classification

### Performance Considerations
- **Model Size**: ~1.3GB
- **Memory Usage**: ~4-6GB during inference
- **Inference Time**: 100-500ms per request (depending on text length)
- **Concurrent Requests**: Supports multiple concurrent evaluations

## 🚨 Error Handling

The service handles various error conditions:
- Empty context or output
- Model loading failures
- Tokenization errors
- CUDA/memory issues

All errors are logged with detailed information for debugging.

## 📊 Monitoring and Logging

### Health Checks
- `/health` endpoint for service status
- Docker health checks included
- Model loading status tracking

### Logging
- Structured logging with timestamps
- Request/response logging
- Performance metrics
- Error tracking with stack traces

## 🔄 Integration with RAG Systems

This service is designed to integrate seamlessly with RAG (Retrieval-Augmented Generation) evaluation pipelines:

1. **Context**: Retrieved documents/passages
2. **Query**: User's original question
3. **Output**: LLM-generated response
4. **Evaluation**: Faithfulness score and detailed analysis

## 🤝 Contributing

When contributing to this service:
1. Maintain backward compatibility with the API
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Follow the existing code style and logging patterns

## 📝 License

This service is part of the atri RAG evaluation framework. 