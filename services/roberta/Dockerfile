# 1. Build stage
FROM python:3.10-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

# 2. Final stage
FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /usr/local /usr/local

RUN pip install --no-cache-dir uvicorn[standard]

COPY --from=builder /app/model_cache /app/model_cache

COPY main.py .

ENV PORT=8080
ENV HOST=0.0.0.0
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 