# ML Evaluation Microservices

This repository contains a suite of standalone FastAPI microservices for evaluating text and AI agent outputs. Each service is focused on a specific evaluation metric or security/privacy check, and is designed for modular deployment and easy integration.

## Available Services

- **similarity**: Semantic similarity and agentic metrics (goal accuracy, intent resolution)
- **token-classification**: PII detection, identity disclosure, and privacy risk scoring
- **rule-based**: Information leakage, invisible/control character detection, insecure output, plugin execution risk, and narrative flow analysis
- **bert-detoxify**: Toxicity, hate speech, and content moderation
- **roberta**: Political bias detection
- **t5-base**: Grammatical correctness and token bloat DoS detection
- **deberta-v3-base**: Prompt injection detection
- **deberta-v3-large**: Answer relevance and text relationship scoring
- **bert-base-uncased**: Creativity, semantic similarity, and relevance
- ...and more (see the `services/` directory)

Each service lives under `services/<name>/` and includes:
- FastAPI app (`main.py`)
- Model download/setup scripts
- `requirements.txt` and `Dockerfile`
- Service-specific README with API details and usage examples

## Usage

- **Local Development:**
  - Build and run any service with its Makefile or Dockerfile
  - Example: `cd services/token-classification && make build && make run`
- **All Services:**
  - Bring up all services with Docker Compose (if provided):
    ```bash
    docker-compose up
    ```
- **Health Checks:**
  - Each service exposes a `/health` endpoint for readiness checks

## Monitoring & Scripts
- Monitoring and alerting configs are in the `monitoring/` directory
- Deployment and utility scripts are in the `scripts/` directory

## Contributing
- Each service is self-contained and can be extended independently
- See the service-specific README files for API details, models, and contribution guidelines

---
For more information, see the README in each service directory.