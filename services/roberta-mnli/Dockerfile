# RoBERTa-large-mnli Faithfulness Evaluation Service Dockerfile
# This service evaluates faithfulness of LLM outputs using Natural Language Inference

# 1. Build stage - Download and cache the model
FROM python:3.10-slim AS builder
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download RoBERTa-large-mnli model
COPY download_model.py .
RUN python download_model.py

# 2. Final stage - Runtime environment
FROM python:3.10-slim
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local /usr/local

# Install uvicorn for production serving
RUN pip install --no-cache-dir uvicorn[standard]

# Copy the cached model from builder stage
COPY --from=builder /app/model_cache /app/model_cache

# Copy application code
COPY main.py .

# Environment variables
ENV PORT=8080
ENV HOST=0.0.0.0
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TOKENIZERS_PARALLELISM=false

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"] 