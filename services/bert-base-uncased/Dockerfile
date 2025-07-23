# Build stage
FROM python:3.10-slim AS builder
WORKDIR /app

# Install build dependencies (e.g., git)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies early
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model downloader and download model
COPY download_model.py .
RUN python download_model.py

# Final stage
FROM python:3.10-slim
WORKDIR /app

# Install only runtime essentials
RUN pip install --no-cache-dir uvicorn[standard]

# Copy installed packages and model files from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/model_cache /app/model_cache

# Copy application code
COPY main.py .

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

# Add a script to wait for startup
RUN echo '#!/bin/bash\npython -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" --log-level info' > /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/app/start.sh"]
