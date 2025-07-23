# Build stage
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/tmp/cache \
    HF_HOME=/tmp/cache \
    TORCH_HOME=/tmp/cache \
    PYTHONPATH=/app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create cache directories with proper permissions
RUN mkdir -p /tmp/cache && \
    mkdir -p /app/model_cache && \
    chmod -R 777 /tmp/cache && \
    chmod -R 777 /app/model_cache

# Copy specific files first
COPY requirements.txt .
COPY download_model.py .
COPY main.py .

# Install dependencies
RUN pip install --user -r requirements.txt

# Download model
RUN python download_model.py

# Final stage
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/tmp/cache \
    HF_HOME=/tmp/cache \
    TORCH_HOME=/tmp/cache \
    IN_DOCKER=true \
    PYTHONPATH=/app

# Create cache directories with proper permissions
RUN mkdir -p /tmp/cache && \
    mkdir -p /app/model_cache && \
    chmod -R 777 /tmp/cache && \
    chmod -R 777 /app/model_cache

# Copy specific files from builder stage
COPY --from=builder /app/main.py /app/
COPY --from=builder /app/download_model.py /app/
COPY --from=builder /app/requirements.txt /app/
COPY --from=builder /app/model_cache /app/model_cache
COPY --from=builder /root/.local /root/.local

# Set permissions for application files
RUN chmod -R 755 /app

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting toxicity service..."\n\
echo "Cache directories:"\n\
ls -la /tmp/cache\n\
ls -la /app/model_cache\n\
echo "Application files:"\n\
ls -la /app\n\
echo "Python path:"\n\
python -c "import sys; print(sys.path)"\n\
echo "Current directory:"\n\
pwd\n\
cd /app && python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run the application
CMD ["/app/start.sh"] 