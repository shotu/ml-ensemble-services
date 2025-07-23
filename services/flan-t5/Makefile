.PHONY: build run clean test

# Service name
SERVICE_NAME = flan-t5

# Docker image info
IMAGE_NAME = flan-t5
REPO = gcr.io/atri-raime
TAG = latest

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Tag the Docker image
tag:
	docker tag $(IMAGE_NAME) $(REPO)/$(IMAGE_NAME):$(TAG)

# Push the Docker image to Google Container Registry
push: tag
	docker push $(REPO)/$(IMAGE_NAME):$(TAG)

# Run the Docker container locally
run: build
	docker run -p 8080:8080 $(IMAGE_NAME)

# Run the Docker container locally with the debug flag
run-debug: build
	docker run -e DEBUG=True -p 8080:8080 $(IMAGE_NAME)

# Clean up Docker images
clean:
	docker rmi $(IMAGE_NAME) $(REPO)/$(IMAGE_NAME):$(TAG) || true

# Test the service locally
test:
	./test_script.sh 