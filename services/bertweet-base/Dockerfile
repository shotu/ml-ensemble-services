# Build stage
FROM python:3.9-slim AS builder

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/tmp/cache \
    HF_HOME=/tmp/cache \
    TORCH_HOME=/tmp/cache

WORKDIR /app

# Install git and dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Create cache directories
RUN mkdir -p /tmp/cache && \
    mkdir -p /app/model_cache && \
    chmod -R 777 /tmp/cache && \
    chmod -R 777 /app/model_cache

# Install Python packages early
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Download model and tokenizer
COPY download_model.py .
RUN python download_model.py

# Final runtime image
FROM python:3.9-slim
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/tmp/cache \
    HF_HOME=/tmp/cache \
    TORCH_HOME=/tmp/cache \
    PORT=8080

# Create cache directories
RUN mkdir -p /tmp/cache && \
    mkdir -p /app/model_cache && \
    chmod -R 777 /tmp/cache && \
    chmod -R 777 /app/model_cache

# Install runtime essentials
RUN pip install --no-cache-dir uvicorn[standard]

# Copy from builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/model_cache /app/model_cache

# Copy app code
COPY main.py .

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"] 