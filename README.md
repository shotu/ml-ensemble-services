# atri RAIME ML Services

This repo holds standalone FastAPI microservices for RAIME evaluations:
- **similarity**: SBERT‐based text similarity  
- **bert_toxicity**: BERT‐based toxicity classification  

Each lives under `services/<name>/` with:
  - FastAPI `app/main.py` + Pydantic `schemas.py`  
  - `requirements.txt` & `Dockerfile`  
  - a service‐specific `README.md`

Bring them up locally with:
\`\`\`
docker-compose up
\`\`\`
