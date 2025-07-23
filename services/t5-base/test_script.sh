#!/bin/bash

# Set variables
SERVICE_URL="https://t5-base-production-drnc7zg5yq-uc.a.run.app"
API_GATEWAY_URL="https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1"
API_KEY="YOUR_API_KEY_HERE"  # Replace with your actual API key

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Testing T5-Base Service ===${NC}"

# Health check
echo -e "\n${BLUE}Testing health endpoint:${NC}"
curl -s -X GET ${SERVICE_URL}/health | jq

# Testing direct service
echo -e "\n${BLUE}Testing direct service with grammatically correct text:${NC}"
curl -s -X POST "${SERVICE_URL}/detect/grammar" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog. This sentence is grammatically correct."
  }' | jq

echo -e "\n${BLUE}Testing direct service with grammatically incorrect text:${NC}"
curl -s -X POST "${SERVICE_URL}/detect/grammar" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This are a test sentence with grammar error and wrong verb agreement."
  }' | jq

echo -e "\n${BLUE}Testing direct service with punctuation errors:${NC}"
curl -s -X POST "${SERVICE_URL}/detect/grammar" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world how are you today i am fine thank you"
  }' | jq

echo -e "\n${BLUE}Testing direct service with mixed errors:${NC}"
curl -s -X POST "${SERVICE_URL}/detect/grammar" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Me and him was going to store yesterday but we didnt had enough money."
  }' | jq

# Testing through API Gateway
if [ "$API_KEY" != "YOUR_API_KEY_HERE" ]; then
  echo -e "\n${BLUE}Testing through API Gateway:${NC}"
  curl -s -X POST "${API_GATEWAY_URL}/detect/grammar" \
    -H "Content-Type: application/json" \
    -H "x-api-key: ${API_KEY}" \
    -d '{
      "text": "This are a test sentence with grammar error."
    }' | jq
else
  echo -e "\n${RED}Skipping API Gateway test. Replace 'YOUR_API_KEY_HERE' with your actual API key to test.${NC}"
fi

echo -e "\n${GREEN}Testing completed!${NC}" 