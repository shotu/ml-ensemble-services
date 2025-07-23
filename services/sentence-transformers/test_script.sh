#!/bin/bash

# Sentence Transformers Service Test Script
# Tests the similarity computation endpoint with various text pairs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
LOCAL_URL="http://localhost:8080"
PRODUCTION_URL="https://sentence-transformers-production-drnc7zg5yq-uc.a.run.app"
API_GATEWAY_URL="https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1/compute/similarity"

# Default to local if no argument provided
SERVICE_URL=${1:-$LOCAL_URL}

echo -e "${YELLOW}Testing Sentence Transformers Service${NC}"
echo "Service URL: $SERVICE_URL"
echo "=================================="

# Function to test endpoint
test_similarity() {
    local name="$1"
    local text1="$2"
    local text2="$3"
    local expected_similarity="$4"
    
    echo -e "\n${YELLOW}Test: $name${NC}"
    echo "Text 1: \"$text1\""
    echo "Text 2: \"$text2\""
    echo "Expected similarity: $expected_similarity"
    echo "---"
    
    response=$(curl -s -X POST "$SERVICE_URL/compute/similarity" \
        -H "Content-Type: application/json" \
        -d "{\"text1\": \"$text1\", \"text2\": \"$text2\"}")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Request successful${NC}"
        echo "Response:"
        echo "$response" | jq '.'
        
        # Extract actual value for validation
        actual_value=$(echo "$response" | jq -r '.actual_value // "null"')
        
        if [ "$actual_value" != "null" ]; then
            echo -e "${GREEN}✓ Valid response structure${NC}"
            echo "Similarity Score: $actual_value"
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
    local text1="$2"
    local text2="$3"
    
    echo -e "\n${YELLOW}API Gateway Test: $name${NC}"
    echo "Text 1: \"$text1\""
    echo "Text 2: \"$text2\""
    echo "---"
    
    response=$(curl -s -X POST "$API_GATEWAY_URL" \
        -H "Content-Type: application/json" \
        -H "x-api-key: test-key" \
        -d "{\"text1\": \"$text1\", \"text2\": \"$text2\"}")
    
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

# Test cases with different similarity levels

# High similarity tests
test_similarity "Identical Texts" \
    "The quick brown fox jumps over the lazy dog" \
    "The quick brown fox jumps over the lazy dog" \
    "High (1.0)"

test_similarity "Paraphrased Sentences" \
    "The cat is sleeping on the couch" \
    "A feline is resting on the sofa" \
    "High (0.8-0.9)"

test_similarity "Synonymous Phrases" \
    "I am happy today" \
    "I feel joyful this day" \
    "High (0.7-0.8)"

# Medium similarity tests
test_similarity "Related Topics" \
    "Machine learning is a subset of artificial intelligence" \
    "Deep learning algorithms require large datasets" \
    "Medium (0.5-0.7)"

test_similarity "Same Domain Different Focus" \
    "Python is a programming language" \
    "JavaScript is used for web development" \
    "Medium (0.4-0.6)"

# Low similarity tests
test_similarity "Different Topics" \
    "I love eating pizza" \
    "The weather is sunny today" \
    "Low (0.2-0.4)"

test_similarity "Opposite Meanings" \
    "This movie is excellent" \
    "This film is terrible" \
    "Low (0.1-0.3)"

test_similarity "Completely Unrelated" \
    "Quantum physics is fascinating" \
    "My grandmother bakes delicious cookies" \
    "Very Low (0.0-0.2)"

# Question-Answer similarity tests
test_similarity "Question-Answer Match" \
    "What is the capital of France?" \
    "The capital of France is Paris" \
    "High (0.7-0.8)"

test_similarity "Question-Wrong Answer" \
    "What is the capital of France?" \
    "The capital of Germany is Berlin" \
    "Medium (0.3-0.5)"

# Technical content tests
test_similarity "Technical Documentation" \
    "Install the package using pip install numpy" \
    "Use pip to install the numpy package" \
    "High (0.8-0.9)"

test_similarity "Code vs Description" \
    "def hello_world(): print('Hello, World!')" \
    "This function prints a greeting message" \
    "Medium (0.4-0.6)"

# Short vs Long text tests
test_similarity "Short vs Long" \
    "AI" \
    "Artificial Intelligence is a field of computer science that aims to create intelligent machines" \
    "Medium (0.5-0.7)"

# Test API Gateway if not testing locally
if [ "$SERVICE_URL" != "$LOCAL_URL" ]; then
    echo -e "\n${YELLOW}Testing API Gateway Integration${NC}"
    echo "=================================="
    
    test_api_gateway "High Similarity" \
        "The dog is running in the park" \
        "A canine is jogging through the garden"
    
    test_api_gateway "Low Similarity" \
        "I enjoy reading books" \
        "The car needs gasoline"
fi

# Performance test
echo -e "\n${YELLOW}Performance Test${NC}"
echo "---"
echo "Running 5 consecutive requests..."

start_time=$(date +%s%N)
for i in {1..5}; do
    curl -s -X POST "$SERVICE_URL/compute/similarity" \
        -H "Content-Type: application/json" \
        -d '{"text1": "Performance test message number '$i'", "text2": "This is test message '$i' for performance evaluation"}' > /dev/null
done
end_time=$(date +%s%N)

duration=$(( (end_time - start_time) / 1000000 ))
avg_time=$(( duration / 5 ))

echo -e "${GREEN}✓ Performance test completed${NC}"
echo "Total time: ${duration}ms"
echo "Average time per request: ${avg_time}ms"

echo -e "\n${GREEN}All tests completed!${NC}" 