.PHONY: build run test clean push

IMAGE_NAME = rule-based
PROJECT_ID ?= atri-dev
GCR_HOSTNAME ?= us-docker.pkg.dev
TAG ?= latest
PORT ?= 8080

build:
	@echo "Building Docker image for $(IMAGE_NAME)..."
	docker build -t $(IMAGE_NAME):$(TAG) .

run: build
	@echo "Running container on port $(PORT)..."
	docker run -p $(PORT):8080 --name $(IMAGE_NAME) $(IMAGE_NAME):$(TAG)

run-background: build
	@echo "Running container in background on port $(PORT)..."
	docker run -d -p $(PORT):8080 --name $(IMAGE_NAME) $(IMAGE_NAME):$(TAG)

test:
	@echo "Running tests..."
	chmod +x test_script.sh
	./test_script.sh http://localhost:$(PORT)

stop:
	@echo "Stopping and removing container..."
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true

clean: stop
	@echo "Removing image..."
	docker rmi $(IMAGE_NAME):$(TAG) || true

push: build
	@echo "Tagging image for $(GCR_HOSTNAME)/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG)..."
	docker tag $(IMAGE_NAME):$(TAG) $(GCR_HOSTNAME)/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG)
	@echo "Pushing image to container registry..."
	docker push $(GCR_HOSTNAME)/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG)

deploy: push
	@echo "Deploying to Cloud Run..."
	gcloud run deploy $(IMAGE_NAME) \
		--image $(GCR_HOSTNAME)/$(PROJECT_ID)/$(IMAGE_NAME):$(TAG) \
		--platform managed \
		--region us-central1 \
		--allow-unauthenticated 