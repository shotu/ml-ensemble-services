# Build stage
FROM python:3.10-slim AS builder
WORKDIR /app

# Install git and dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python packages early
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model and tokenizer
COPY download_model.py .
RUN python download_model.py

# Final runtime image
FROM python:3.10-slim
WORKDIR /app

# Install runtime essentials
RUN pip install --no-cache-dir uvicorn[standard]

# Copy from builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/model_cache /app/model_cache

# Copy app code
COPY main.py .

# Set env vars
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 