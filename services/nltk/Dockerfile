# Build stage
FROM python:3.10-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

# Final stage
FROM python:3.10-slim
WORKDIR /app

RUN pip install --no-cache-dir uvicorn[standard]

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local
# Copy NLTK data from builder
COPY --from=builder /app/nltk_data /app/nltk_data

COPY main.py .

# Set environment variables
ENV PORT=8080
ENV HOST=0.0.0.0
ENV NLTK_DATA=/app/nltk_data
ENV SERVICE_NAME=nltk

EXPOSE 8080

RUN echo '#!/bin/bash\npython -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" --log-level info' > /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/app/start.sh"] 