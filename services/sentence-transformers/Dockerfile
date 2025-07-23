FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for spaCy and other packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy and run model download script
COPY download_model.py .
RUN python download_model.py

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them in final stage
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy model cache and spaCy data from builder
COPY --from=builder /app/model_cache /app/model_cache
COPY --from=builder /usr/local/lib/python3.10/site-packages/en_core_web_trf /usr/local/lib/python3.10/site-packages/en_core_web_trf

# Copy application code
COPY main.py .
COPY download_model.py .

# Set environment variables
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Create startup script
RUN echo '#!/bin/bash\npython -m uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8080} --log-level info' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose port
EXPOSE 8080

# Run the application
CMD ["/app/start.sh"] 