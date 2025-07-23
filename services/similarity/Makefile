# Makefile for Similarity Service

IMAGE_NAME    ?= raime-similarity
PORT          ?= 8000
VENV_DIR      ?= .venv
PYTHON        ?= $(VENV_DIR)/bin/python
PIP           ?= $(VENV_DIR)/bin/pip
UVICORN       ?= $(VENV_DIR)/bin/uvicorn

.PHONY: help setup install download-model run-dev docker-build docker-run clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtualenv and install Python dependencies
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install --no-cache-dir -r requirements.txt


.PHONY: help install download-model run-dev docker-build docker-run clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install --upgrade pip
	pip install --no-cache-dir -r requirements.txt

download-model: ## Pre-download SBERT model into local cache
	python download_model.py

run-dev: download-model ## Run service locally with auto-reload
	uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME) .

docker-run: ## Run Docker container
	docker run --rm -p $(PORT):8000 $(IMAGE_NAME)

clean: ## Remove Python cache files
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
