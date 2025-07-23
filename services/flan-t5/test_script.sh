#!/bin/bash

# Set variables
SERVICE_URL="https://flan-t5-production-drnc7zg5yq-uc.a.run.app"
API_GATEWAY_URL="https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1"
API_KEY="YOUR_API_KEY_HERE"  # Replace with your actual API key

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Testing Flan-T5 Service ===${NC}"

# Health check
echo -e "\n${BLUE}Testing health endpoint:${NC}"
curl -s -X GET ${SERVICE_URL}/health | jq

# Testing direct service
echo -e "\n${BLUE}Testing direct service with factually consistent text:${NC}"
curl -s -X POST "${SERVICE_URL}/detect/factual_consistency" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Water is a colorless substance with the chemical formula H2O that is essential for life on Earth. It is found in oceans, lakes, and streams.",
    "context": "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance. It is the main constituent of Earth'\''s streams, lakes, and oceans, and the fluids of most living organisms. Its chemical formula is H2O, meaning that each of its molecules contains one oxygen and two hydrogen atoms."
  }' | jq

echo -e "\n${BLUE}Testing direct service with factually inconsistent text:${NC}"
curl -s -X POST "${SERVICE_URL}/detect/factual_consistency" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Water is a blue liquid with a chemical formula CO2 and is mainly used for industrial purposes.",
    "context": "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance. It is the main constituent of Earth'\''s streams, lakes, and oceans, and the fluids of most living organisms. Its chemical formula is H2O, meaning that each of its molecules contains one oxygen and two hydrogen atoms."
  }' | jq

echo -e "\n${BLUE}Testing direct service with partially consistent text:${NC}"
curl -s -X POST "${SERVICE_URL}/detect/factual_consistency" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Water has the chemical formula H2O and is used for drinking and swimming.",
    "context": "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance. It is the main constituent of Earth'\''s streams, lakes, and oceans, and the fluids of most living organisms. Its chemical formula is H2O, meaning that each of its molecules contains one oxygen and two hydrogen atoms."
  }' | jq

# Testing through API Gateway
if [ "$API_KEY" != "YOUR_API_KEY_HERE" ]; then
  echo -e "\n${BLUE}Testing through API Gateway:${NC}"
  curl -s -X POST "${API_GATEWAY_URL}/detect/factual_consistency" \
    -H "Content-Type: application/json" \
    -H "x-api-key: ${API_KEY}" \
    -d '{
      "text": "Water is a colorless substance with the chemical formula H2O that is essential for life on Earth. It is found in oceans, lakes, and streams.",
      "context": "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance. It is the main constituent of Earth'\''s streams, lakes, and oceans, and the fluids of most living organisms. Its chemical formula is H2O, meaning that each of its molecules contains one oxygen and two hydrogen atoms."
    }' | jq
else
  echo -e "\n${RED}Skipping API Gateway test. Replace 'YOUR_API_KEY_HERE' with your actual API key to test.${NC}"
fi

echo -e "\n${GREEN}Testing completed!${NC}" 