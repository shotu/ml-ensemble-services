#!/bin/bash

# DeBERTa v3 Large Service Test Script
# Tests the simplified API with input_text and output_text

set -e

# Configuration
SERVICE_URL_PRODUCTION="https://deberta-v3-large-production-drnc7zg5yq-uc.a.run.app"
SERVICE_URL_STAGING="https://deberta-v3-large-staging-drnc7zg5yq-uc.a.run.app"
API_GATEWAY_URL="https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1/detect/answer_relevance"
API_KEY="AIzaSyCpyPsaWepJQ1MQiq-tq1x0fALqeNmlBL8"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to test endpoint
test_endpoint() {
    local url=$1
    local description=$2
    local input_text=$3
    local output_text=$4
    local headers=$5
    
    print_status "Testing: $description"
    echo "URL: $url"
    echo "Input: $input_text"
    echo "Output: $output_text"
    
    local response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
        $headers \
        -H "Content-Type: application/json" \
        -d "{
            \"input_text\": \"$input_text\",
            \"output_text\": \"$output_text\"
        }")
    
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" = "200" ]; then
        print_success "HTTP $http_code - Success"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "HTTP $http_code - Failed"
        echo "$body"
    fi
    echo "----------------------------------------"
}

# Function to test health endpoint
test_health() {
    local url=$1
    local description=$2
    
    print_status "Testing health: $description"
    echo "URL: $url/health"
    
    local response=$(curl -s -w "\n%{http_code}" "$url/health")
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" = "200" ]; then
        print_success "HTTP $http_code - Health check passed"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        print_error "HTTP $http_code - Health check failed"
        echo "$body"
    fi
    echo "----------------------------------------"
}

echo "=========================================="
echo "DeBERTa v3 Large Service Test Suite"
echo "=========================================="

# Test 1: Health checks
print_status "=== HEALTH CHECKS ==="
test_health "$SERVICE_URL_PRODUCTION" "Production Service"

# Test 2: Direct service tests
print_status "=== DIRECT SERVICE TESTS ==="

# High relevance case
test_endpoint "$SERVICE_URL_PRODUCTION/detect/answer_relevance" \
    "High Relevance - Question and Direct Answer" \
    "What is the capital of France?" \
    "The capital of France is Paris." \
    ""

# Medium relevance case
test_endpoint "$SERVICE_URL_PRODUCTION/detect/answer_relevance" \
    "Medium Relevance - Related but Indirect" \
    "How do I bake a cake?" \
    "Baking requires flour, eggs, and sugar as main ingredients." \
    ""

# Low relevance case
test_endpoint "$SERVICE_URL_PRODUCTION/detect/answer_relevance" \
    "Low Relevance - Unrelated Content" \
    "What is quantum computing?" \
    "The weather today is sunny and warm." \
    ""

# Complex technical case
test_endpoint "$SERVICE_URL_PRODUCTION/detect/answer_relevance" \
    "Technical Content - Programming Question" \
    "How do I implement a binary search algorithm?" \
    "Binary search works by repeatedly dividing the search interval in half and comparing the target value with the middle element." \
    ""

# Test 3: API Gateway tests
print_status "=== API GATEWAY TESTS ==="

# Test through API Gateway with API key
test_endpoint "$API_GATEWAY_URL" \
    "API Gateway - High Relevance" \
    "Explain machine learning" \
    "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed." \
    "-H \"x-api-key: $API_KEY\""

test_endpoint "$API_GATEWAY_URL" \
    "API Gateway - Low Relevance" \
    "What is photosynthesis?" \
    "I like to eat pizza on weekends." \
    "-H \"x-api-key: $API_KEY\""

# Test 4: Edge cases
print_status "=== EDGE CASE TESTS ==="

# Empty input
test_endpoint "$SERVICE_URL_PRODUCTION/detect/answer_relevance" \
    "Edge Case - Empty Input" \
    "" \
    "This is a response to nothing." \
    ""

# Very long text
test_endpoint "$SERVICE_URL_PRODUCTION/detect/answer_relevance" \
    "Edge Case - Long Text" \
    "What are the benefits of renewable energy sources?" \
    "Renewable energy sources such as solar, wind, hydroelectric, and geothermal power offer numerous benefits including environmental sustainability, reduced greenhouse gas emissions, energy independence, job creation in green industries, long-term cost savings, and reduced reliance on fossil fuels which helps combat climate change and air pollution." \
    ""

# Test 5: Performance test
print_status "=== PERFORMANCE TESTS ==="
print_status "Running 5 consecutive requests to test performance..."

for i in {1..5}; do
    print_status "Request $i/5"
    start_time=$(date +%s%N)
    
    response=$(curl -s -w "\n%{http_code}" -X POST "$SERVICE_URL_PRODUCTION/detect/answer_relevance" \
        -H "Content-Type: application/json" \
        -d '{
            "input_text": "What is artificial intelligence?",
            "output_text": "Artificial intelligence is the simulation of human intelligence in machines."
        }')
    
    end_time=$(date +%s%N)
    duration=$(( (end_time - start_time) / 1000000 ))
    
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" = "200" ]; then
        print_success "Request $i completed in ${duration}ms"
    else
        print_error "Request $i failed with HTTP $http_code"
    fi
done

echo "=========================================="
print_success "DeBERTa v3 Large Service test suite completed!"
echo "==========================================" 