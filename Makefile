.PHONY: build-ner run-ner

build-ner:
	docker build -t ner-service:latest ./services/ner_service

run-ner:
	docker run --rm -p 8001:8001 ner-service:latest