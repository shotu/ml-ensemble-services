.PHONY: build run test clean

# Variables
SERVICE_NAME = roberta
DOCKER_IMAGE = roberta-service
PORT = 8080

# Build the Docker image
build:
	docker build -t $(DOCKER_IMAGE) .

# Run the service locally using Docker
run:
	docker run -p $(PORT):$(PORT) $(DOCKER_IMAGE)

# Run tests
test:
	pytest tests/

# Clean up Docker resources
clean:
	docker rmi $(DOCKER_IMAGE) || true

# Development commands
dev:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	uvicorn main:app --reload

# Lint the code
lint:
	flake8 .
	black . --check
	isort . --check-only

# Format the code
format:
	black .
	isort .

# Help command
help:
	@echo "Available commands:"
	@echo "  make build    - Build the Docker image"
	@echo "  make run      - Run the service locally using Docker"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean up Docker resources"
	@echo "  make dev      - Set up development environment"
	@echo "  make lint     - Run linters"
	@echo "  make format   - Format the code" 