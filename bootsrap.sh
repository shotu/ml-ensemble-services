#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# bootstrap.sh
# Bootstraps atri RAIME ML Services (two services: similarity & bert_toxicity)
# ──────────────────────────────────────────────────────────────────────────────

echo "🛠  Bootstrapping atri RAIME ML Services…"

# 1) Top‐level README.md
cat > README.md <<'EOF'
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
EOF

# 2) .gitignore
cat > .gitignore <<'EOF'
# Python artifacts
__pycache__/
*.py[cod]

# Virtual envs & env files
venv/
.env

# IDE
.vscode/
.idea/

# Docker
docker-compose*.yml

# Logs
*.log
EOF

# 3) docker-compose.yml
cat > docker-compose.yml <<'EOF'
version: "3.8"
services:
  similarity:
    build: ./services/similarity
    ports:
      - "8000:8000"
  bert_toxicity:
    build: ./services/bert_toxicity
    ports:
      - "8001:8000"
EOF

# 4) Create service directories
mkdir -p services/similarity/app
mkdir -p services/bert_toxicity/app

# ──────────────────────────────────────────────────────────────────────────────
# 5) similarity service
# ──────────────────────────────────────────────────────────────────────────────

# schemas.py
cat > services/similarity/app/schemas.py <<'EOF'
from pydantic import BaseModel

class SimilarityRequest(BaseModel):
    sources: list[str]
    targets: list[str]
    model_name: str = "sbert-mini"
    params: dict[str, float] = {}

class PairSimilarity(BaseModel):
    source: str
    target: str
    similarity: float

class SimilarityResponse(BaseModel):
    metric_name: str
    similarities: list[list[float]]
    pairs: list[PairSimilarity]
    diagnostics: dict
    model_info: dict
EOF

# main.py
cat > services/similarity/app/main.py <<'EOF'
import time, logging
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
from .schemas import SimilarityRequest, SimilarityResponse, PairSimilarity

log = logging.getLogger("similarity")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Text Similarity Service", version="1.0.0")
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/predict", response_model=SimilarityResponse)
async def predict(req: SimilarityRequest):
    t0 = time.time()
    emb_s = model.encode(req.sources, convert_to_tensor=True)
    emb_t = model.encode(req.targets, convert_to_tensor=True)
    sims = util.cos_sim(emb_s, emb_t).tolist()

    pairs = [
        PairSimilarity(source=src, target=tgt, similarity=sims[i][j])
        for i, src in enumerate(req.sources)
        for j, tgt in enumerate(req.targets)
    ]
    info = {
        "model_name": req.model_name,
        "model_version": "1.0.0",
        "compute_time_ms": (time.time() - t0) * 1000
    }
    return SimilarityResponse(
        metric_name="text_similarity",
        similarities=sims,
        pairs=pairs,
        diagnostics={"num_sources": len(req.sources),"num_targets": len(req.targets)},
        model_info=info,
    )
EOF

# requirements.txt
cat > services/similarity/requirements.txt <<'EOF'
fastapi
uvicorn[standard]
sentence-transformers
torch
EOF

# Dockerfile
cat > services/similarity/Dockerfile <<'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0","--port","8000"]
EOF

# service README.md
cat > services/similarity/README.md <<'EOF'
# Similarity Service

**POST** `/predict`

**Request**: `SimilarityRequest`  
**Response**: `SimilarityResponse`

Example:
\`\`\`
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sources":["foo"],"targets":["bar"]}'
\`\`\`
EOF

# ──────────────────────────────────────────────────────────────────────────────
# 6) bert_toxicity service
# ──────────────────────────────────────────────────────────────────────────────

# schemas.py
cat > services/bert_toxicity/app/schemas.py <<'EOF'
from pydantic import BaseModel

class ToxicityRequest(BaseModel):
    text: str

class ToxicityResponse(BaseModel):
    metric_name: str
    score: float
EOF

# main.py (stub)
cat > services/bert_toxicity/app/main.py <<'EOF'
import logging
from fastapi import FastAPI
from .schemas import ToxicityRequest, ToxicityResponse

log = logging.getLogger("bert_toxicity")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="BERT Toxicity Service", version="1.0.0")

@app.post("/predict", response_model=ToxicityResponse)
async def predict(req: ToxicityRequest):
    # TODO: plug in real BERT-based toxicity pipeline
    return ToxicityResponse(metric_name="bert_toxicity", score=0.0)
EOF

# requirements.txt
cat > services/bert_toxicity/requirements.txt <<'EOF'
fastapi
uvicorn[standard]
# transformers, torch, etc., for real implementation
EOF

# Dockerfile
cat > services/bert_toxicity/Dockerfile <<'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0","--port","8000"]
EOF

# service README.md
cat > services/bert_toxicity/README.md <<'EOF'
# BERT Toxicity Service

**POST** `/predict`

**Request**: `ToxicityRequest`  
**Response**: `ToxicityResponse`

Example:
\`\`\`
curl -X POST localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"some text"}'
\`\`\`
EOF

echo "✅ Bootstrap complete!"
echo "Next steps:"
echo " 1) cd services/similarity && docker build -t raime-similarity ."
echo " 2) cd services/bert_toxicity && docker build -t raime-bert-toxicity ."
echo " 3) docker-compose up"
