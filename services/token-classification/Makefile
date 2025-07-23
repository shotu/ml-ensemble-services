PROJECT := atri-dev
IMAGE_NAME := token-classification
TAG := latest
REGION := us-central1
SERVICE_NAME := token-classification

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

run:
	docker run -p 8080:8080 $(IMAGE_NAME):$(TAG)

push:
	docker tag $(IMAGE_NAME):$(TAG) $(REGION)-docker.pkg.dev/$(PROJECT)/ml-services/$(IMAGE_NAME):$(TAG)
	docker push $(REGION)-docker.pkg.dev/$(PROJECT)/ml-services/$(IMAGE_NAME):$(TAG)

deploy:
	gcloud run deploy $(SERVICE_NAME) \
		--image $(REGION)-docker.pkg.dev/$(PROJECT)/ml-services/$(IMAGE_NAME):$(TAG) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--memory 2Gi \
		--cpu 1 \
		--min-instances 0 \
		--project $(PROJECT)

.PHONY: build run push deploy 