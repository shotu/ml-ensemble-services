#!/bin/bash

# Set the service URL (local or deployed)
SERVICE_URL="${1:-http://localhost:8080}"

echo "====================================================================="
echo "  TOKEN CLASSIFICATION TEST"
echo "  Service URL: ${SERVICE_URL}"
echo "  Note: The service returns continuous scores with higher scores"
echo "  indicating more personal information detected"
echo "====================================================================="

# Test with content containing personal data
echo "Testing with content containing personal data..."
curl -X POST "${SERVICE_URL}/detect/data_leakage" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My name is John Smith and my email is john.smith@example.com. You can reach me at 555-123-4567 or visit me at 123 Main St, New York, NY 10001."
  }' | jq .

echo ""

# Test with content without personal data
echo "Testing with content without personal data..."
curl -X POST "${SERVICE_URL}/detect/data_leakage" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The weather is nice today and I think it will be sunny tomorrow as well. The forecast predicts a high of 75 degrees."
  }' | jq .

echo ""

# Test with content containing moderate personal data
echo "Testing with content containing moderate personal data..."
curl -X POST "${SERVICE_URL}/detect/data_leakage" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I live in New York. The weather has been quite pleasant this week with temperatures around 70 degrees."
  }' | jq .

echo ""

# Test health endpoint
echo "Testing health endpoint..."
curl -X GET "${SERVICE_URL}/health" | jq .

echo ""
echo "All tests completed!" 