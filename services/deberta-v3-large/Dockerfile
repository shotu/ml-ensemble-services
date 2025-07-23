# Build stage
FROM python:3.10-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages early
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
RUN mkdir -p /app/model_cache

# Download model and tokenizer
COPY download_model.py .
RUN python download_model.py

# Final runtime image
FROM python:3.10-slim
WORKDIR /app

# Install runtime essentials and debugging tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
    uvicorn[standard] \
    gunicorn

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/model_cache /app/model_cache

# Copy app code
COPY main.py .

# Set env vars
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV PORT=8080
ENV TIMEOUT=600
ENV WORKERS=1
ENV LOG_LEVEL=info

# Verify model cache exists and print size
RUN echo "Model cache files:" && ls -la /app/model_cache && \
    echo "Total model cache size: $(du -sh /app/model_cache | cut -f1)"

# Health check configuration
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

# Use gunicorn with extended timeouts for large model loading
CMD gunicorn main:app --workers ${WORKERS} --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT} --timeout ${TIMEOUT} --log-level ${LOG_LEVEL} \
    --graceful-timeout 120 --keep-alive 120 