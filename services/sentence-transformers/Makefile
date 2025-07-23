.PHONY: build run push all clean

SERVICE_NAME = sentence-transformers
DOCKER_IMAGE = sentence-transformers-service
PROJECT_ID = atri-dev
REGION = us-central1

build:
	docker build -t $(DOCKER_IMAGE) .

run: build
	docker run -p 8080:8080 $(DOCKER_IMAGE)

push: build
	docker tag $(DOCKER_IMAGE) gcr.io/$(PROJECT_ID)/$(DOCKER_IMAGE)
	docker push gcr.io/$(PROJECT_ID)/$(DOCKER_IMAGE)

deploy: push
	gcloud run deploy $(SERVICE_NAME) \
		--image gcr.io/$(PROJECT_ID)/$(DOCKER_IMAGE) \
		--platform managed \
		--region $(REGION) \
		--memory 4Gi \
		--cpu 2 \
		--min-instances 0 \
		--max-instances 10 \
		--concurrency 80 \
		--timeout 300 \
		--allow-unauthenticated \
		--project $(PROJECT_ID)

all: build push deploy

clean:
	docker rmi $(DOCKER_IMAGE) || true
	docker rmi gcr.io/$(PROJECT_ID)/$(DOCKER_IMAGE) || true 