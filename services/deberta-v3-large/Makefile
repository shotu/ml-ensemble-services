.PHONY: build run test clean

SERVICE_NAME = deberta-v3-large
DOCKER_IMAGE = deberta-v3-large
PORT = 8080

build:
	docker build -t $(DOCKER_IMAGE) .

run:
	docker run -p $(PORT):$(PORT) $(DOCKER_IMAGE)

test:
	pytest tests/

clean:
	docker rmi $(DOCKER_IMAGE) || true

dev:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	uvicorn main:app --reload

lint:
	flake8 .
	black . --check
	isort . --check-only

format:
	black .
	isort .

help:
	@echo "Available commands:"
	@echo "  make build    - Build the Docker image"
	@echo "  make run      - Run the service locally using Docker"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean up Docker resources"
	@echo "  make dev      - Set up development environment"
	@echo "  make lint     - Run linters"
	@echo "  make format   - Format code" 