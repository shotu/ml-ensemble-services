# Build stage
FROM python:3.10-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directory
RUN mkdir -p /app/model_cache 

# Download and cache the model
COPY download_model.py .
RUN python download_model.py

# Final stage
FROM python:3.10-slim
WORKDIR /app

RUN pip install --no-cache-dir uvicorn[standard]

# Copy installed packages and model cache from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/model_cache /app/model_cache

# Copy application code
COPY main.py .

ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

RUN echo '#!/bin/bash\npython -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" --log-level info' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose port
EXPOSE 8080

CMD ["/app/start.sh"] 