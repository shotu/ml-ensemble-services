# 1. build stage
FROM python:3.10-slim AS builder
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download model into cache
COPY download_model.py .
RUN python download_model.py

# 2. final stage
FROM python:3.10-slim
WORKDIR /app

# Copy full Python installation from builder
COPY --from=builder /usr/local /usr/local

# Install uvicorn directly in the final stage
RUN pip install --no-cache-dir uvicorn[standard]

# Copy the cached model
COPY --from=builder /app/model_cache /app/model_cache

# copy app code
COPY main.py .

# Set environment variables
ENV PORT=8080
ENV HOST=0.0.0.0
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Expose the port
EXPOSE 8080

# Run with the Python executable directly
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
