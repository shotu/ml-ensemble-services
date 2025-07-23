# Makefile for DistilRoBERTa Rejection Detection Service

# Variables
SERVICE_NAME = distilroberta-rejection-service
IMAGE_NAME = distilroberta-rejection-service
CONTAINER_NAME = distilroberta-rejection-container
PORT = 8080

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build      - Build Docker image"
	@echo "  run        - Run Docker container"
	@echo "  stop       - Stop Docker container"
	@echo "  clean      - Remove Docker container and image"
	@echo "  test       - Test the service endpoints"
	@echo "  logs       - Show container logs"
	@echo "  shell      - Open shell in running container"
	@echo "  dev        - Run service in development mode"

# Build Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

# Run Docker container
.PHONY: run
run:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(IMAGE_NAME)
	@echo "Service running at http://localhost:$(PORT)"

# Stop Docker container
.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Clean up Docker resources
.PHONY: clean
clean: stop
	docker rmi $(IMAGE_NAME) || true

# Test the service
.PHONY: test
test:
	@echo "Testing health endpoint..."
	curl -s http://localhost:$(PORT)/health | python -m json.tool
	@echo "\nTesting refusal detection with rejection text..."
	curl -s -X POST http://localhost:$(PORT)/detect/refusal \
		-H "Content-Type: application/json" \
		-d '{"text":"I apologize, but I cannot assist with that request."}' | python -m json.tool
	@echo "\nTesting refusal detection with normal text..."
	curl -s -X POST http://localhost:$(PORT)/detect/refusal \
		-H "Content-Type: application/json" \
		-d '{"text":"Hello, how can I help you today?"}' | python -m json.tool

# Show container logs
.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Open shell in running container
.PHONY: shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Run in development mode (without Docker)
.PHONY: dev
dev:
	python main.py

# Install dependencies for local development
.PHONY: install
install:
	pip install -r requirements.txt

# Download model for local development
.PHONY: download-model
download-model:
	python download_model.py 