#!/bin/bash

# BERTweet-Base Service Test Script
# Tests the tone evaluation endpoint with various text samples

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
LOCAL_URL="http://localhost:8080"
PRODUCTION_URL="https://bertweet-base-production-drnc7zg5yq-uc.a.run.app"
API_GATEWAY_URL="https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1/detect/response_tone"

# Default to local if no argument provided
SERVICE_URL=${1:-$LOCAL_URL}

echo -e "${YELLOW}Testing BERTweet-Base Service${NC}"
echo "Service URL: $SERVICE_URL"
echo "=================================="

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local text="$2"
    local expected_sentiment="$3"
    
    echo -e "\n${YELLOW}Test: $name${NC}"
    echo "Text: \"$text\""
    echo "Expected sentiment: $expected_sentiment"
    echo "---"
    
    response=$(curl -s -X POST "$SERVICE_URL/detect/response_tone" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\"}")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Request successful${NC}"
        echo "Response:"
        echo "$response" | jq '.'
        
        # Extract actual value for validation
        actual_value=$(echo "$response" | jq -r '.actual_value // "null"')
        raw_label=$(echo "$response" | jq -r '.others.raw_label // "null"')
        
        if [ "$actual_value" != "null" ] && [ "$raw_label" != "null" ]; then
            echo -e "${GREEN}✓ Valid response structure${NC}"
            echo "Score: $actual_value, Label: $raw_label"
        else
            echo -e "${RED}✗ Invalid response structure${NC}"
        fi
    else
        echo -e "${RED}✗ Request failed${NC}"
    fi
}

# Function to test API Gateway
test_api_gateway() {
    local name="$1"
    local text="$2"
    
    echo -e "\n${YELLOW}API Gateway Test: $name${NC}"
    echo "Text: \"$text\""
    echo "---"
    
    response=$(curl -s -X POST "$API_GATEWAY_URL" \
        -H "Content-Type: application/json" \
        -H "x-api-key: test-key" \
        -d "{\"text\": \"$text\"}")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ API Gateway request successful${NC}"
        echo "Response:"
        echo "$response" | jq '.'
    else
        echo -e "${RED}✗ API Gateway request failed${NC}"
    fi
}

# Test health endpoint
echo -e "\n${YELLOW}Testing Health Endpoint${NC}"
echo "---"
health_response=$(curl -s "$SERVICE_URL/health")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Health check successful${NC}"
    echo "$health_response" | jq '.'
else
    echo -e "${RED}✗ Health check failed${NC}"
fi

# Test cases with different sentiments
test_endpoint "Positive Sentiment" "I love this amazing product! It's fantastic and works perfectly." "Positive"

test_endpoint "Negative Sentiment" "This is terrible and I hate it. Worst experience ever!" "Negative"

test_endpoint "Neutral Sentiment" "The weather is cloudy today. It might rain later." "Neutral"

test_endpoint "Mixed Sentiment" "The product has some good features but also several issues that need fixing." "Mixed"

test_endpoint "Short Positive" "Great job! 👍" "Positive"

test_endpoint "Short Negative" "Not good 😞" "Negative"

test_endpoint "Social Media Style" "OMG this is sooo cool!!! #awesome #love" "Positive"

test_endpoint "Formal Text" "The quarterly report indicates a steady increase in revenue streams." "Neutral"

test_endpoint "Question" "How are you doing today?" "Neutral"

test_endpoint "Exclamation" "What an incredible achievement!" "Positive"

# Test API Gateway if not testing locally
if [ "$SERVICE_URL" != "$LOCAL_URL" ]; then
    echo -e "\n${YELLOW}Testing API Gateway Integration${NC}"
    echo "=================================="
    
    test_api_gateway "Positive Text" "This is absolutely wonderful!"
    test_api_gateway "Negative Text" "This is really disappointing."
fi

# Performance test
echo -e "\n${YELLOW}Performance Test${NC}"
echo "---"
echo "Running 5 consecutive requests..."

start_time=$(date +%s%N)
for i in {1..5}; do
    curl -s -X POST "$SERVICE_URL/detect/response_tone" \
        -H "Content-Type: application/json" \
        -d '{"text": "Performance test message number '$i'"}' > /dev/null
done
end_time=$(date +%s%N)

duration=$(( (end_time - start_time) / 1000000 ))
avg_time=$(( duration / 5 ))

echo -e "${GREEN}✓ Performance test completed${NC}"
echo "Total time: ${duration}ms"
echo "Average time per request: ${avg_time}ms"

echo -e "\n${GREEN}All tests completed!${NC}" 