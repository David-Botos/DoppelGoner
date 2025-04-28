#!/bin/bash
# test_inference_worker.sh
# A collection of simple tests for the inference worker service

# Configuration
API_KEY="your-api-key"
WORKER_HOST="localhost"
WORKER_PORT="3000"
MODEL_ID="bge-small-en-v1.5"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${YELLOW}===== Inference Worker Test Script =====${NC}"
echo "Testing worker at $WORKER_HOST:$WORKER_PORT"
echo ""

# Test 1: Check if worker is healthy
echo -e "${YELLOW}Test 1: Check worker health${NC}"
HEALTH_RESPONSE=$(curl -s "http://$WORKER_HOST:$WORKER_PORT/api/health")
HEALTH_STATUS=$(echo $HEALTH_RESPONSE | jq -r '.status')

if [ "$HEALTH_STATUS" == "healthy" ]; then
    echo -e "${GREEN}✓ Worker is healthy${NC}"
    echo "Details: $(echo $HEALTH_RESPONSE | jq -r '.details')"
else
    echo -e "${RED}✗ Worker is not healthy${NC}"
    echo "Response: $HEALTH_RESPONSE"
    exit 1
fi
echo ""

# Test 2: Check worker status
echo -e "${YELLOW}Test 2: Check worker status${NC}"
STATUS_RESPONSE=$(curl -s -H "X-API-Key: $API_KEY" "http://$WORKER_HOST:$WORKER_PORT/api/status")
STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')

if [ -n "$STATUS" ]; then
    echo -e "${GREEN}✓ Worker status retrieved${NC}"
    echo "Status: $STATUS"
    echo "Worker ID: $(echo $STATUS_RESPONSE | jq -r '.worker_id')"
    echo "Capabilities: $(echo $STATUS_RESPONSE | jq -r '.capabilities')"
else
    echo -e "${RED}✗ Failed to get worker status${NC}"
    echo "Response: $STATUS_RESPONSE"
    exit 1
fi
echo ""

# Test 3: Check GPU metrics
echo -e "${YELLOW}Test 3: Check GPU metrics${NC}"
GPU_RESPONSE=$(curl -s -H "X-API-Key: $API_KEY" "http://$WORKER_HOST:$WORKER_PORT/api/gpu/memory")
TOTAL_MEM=$(echo $GPU_RESPONSE | jq -r '.total_mb')

if [ -n "$TOTAL_MEM" ]; then
    echo -e "${GREEN}✓ GPU metrics retrieved${NC}"
    echo "Total memory: $TOTAL_MEM MB"
    echo "Used memory: $(echo $GPU_RESPONSE | jq -r '.used_mb') MB"
    echo "Free memory: $(echo $GPU_RESPONSE | jq -r '.free_mb') MB"
else
    echo -e "${RED}✗ Failed to get GPU metrics${NC}"
    echo "Response: $GPU_RESPONSE"
fi
echo ""

# Test 4: Process a simple batch
echo -e "${YELLOW}Test 4: Process a simple batch${NC}"
REQUEST_ID=$(uuidgen || python -c "import uuid; print(uuid.uuid4())")
JOB_ID=$(uuidgen || python -c "import uuid; print(uuid.uuid4())")

BATCH_DATA='{
  "documents": [
    {
      "service_id": "test-service-1",
      "tokenized_text": "This is a test service description for embedding generation",
      "token_count": 10,
      "job_id": "'$JOB_ID'"
    }
  ],
  "request_id": "'$REQUEST_ID'",
  "priority": 1,
  "model_id": "'$MODEL_ID'"
}'

echo "Sending batch with request_id: $REQUEST_ID"
BATCH_RESPONSE=$(curl -s -X POST \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$BATCH_DATA" \
    "http://$WORKER_HOST:$WORKER_PORT/api/batches")

ERROR=$(echo $BATCH_RESPONSE | jq -r '.error')

if [ "$ERROR" == "null" ]; then
    echo -e "${GREEN}✓ Batch processed successfully${NC}"
    echo "Processing time: $(echo $BATCH_RESPONSE | jq -r '.processing_time_ms') ms"
    echo "Embedding dimensions: $(echo $BATCH_RESPONSE | jq -r '.results[0].embedding | length')"
else
    echo -e "${RED}✗ Batch processing failed${NC}"
    echo "Error: $ERROR"
fi
echo ""

# Test 5: Check Prometheus metrics endpoint
echo -e "${YELLOW}Test 5: Check Prometheus metrics endpoint${NC}"
METRICS_RESPONSE=$(curl -s "http://$WORKER_HOST:$(($WORKER_PORT + 1))/metrics")

if [[ $METRICS_RESPONSE == *"worker_active_batches"* ]]; then
    echo -e "${GREEN}✓ Prometheus metrics available${NC}"
    echo "Sample metrics:"
    echo "$METRICS_RESPONSE" | grep "worker_" | head -n 5
else
    echo -e "${RED}✗ Failed to get Prometheus metrics${NC}"
    echo "Response: ${METRICS_RESPONSE:0:100}..."
fi
echo ""

# Test 6: Register with orchestrator (optional)
echo -e "${YELLOW}Test 6: Register with orchestrator (skipped)${NC}"
echo "To test registration, uncomment and run the following command:"
echo 'curl -X POST -H "X-API-Key: '$API_KEY'" -H "Content-Type: application/json" -d '\''{"orchestrator_url": "http://orchestrator:3000"}'\'' "http://'$WORKER_HOST':'$WORKER_PORT'/api/register"'
echo ""

# Test 7: Test admin endpoints
echo -e "${YELLOW}Test 7: Test admin pause/resume${NC}"
echo "Pausing worker..."
PAUSE_RESPONSE=$(curl -s -X POST -H "X-API-Key: $API_KEY" "http://$WORKER_HOST:$WORKER_PORT/api/admin/pause")
PAUSE_STATUS=$(echo $PAUSE_RESPONSE | jq -r '.status')

if [ "$PAUSE_STATUS" == "paused" ]; then
    echo -e "${GREEN}✓ Worker paused successfully${NC}"
else
    echo -e "${RED}✗ Failed to pause worker${NC}"
    echo "Response: $PAUSE_RESPONSE"
fi

echo "Resuming worker..."
RESUME_RESPONSE=$(curl -s -X POST -H "X-API-Key: $API_KEY" "http://$WORKER_HOST:$WORKER_PORT/api/admin/resume")
RESUME_STATUS=$(echo $RESUME_RESPONSE | jq -r '.status')

if [ "$RESUME_STATUS" == "resumed" ]; then
    echo -e "${GREEN}✓ Worker resumed successfully${NC}"
else
    echo -e "${RED}✗ Failed to resume worker${NC}"
    echo "Response: $RESUME_RESPONSE"
fi
echo ""

# Summary
echo -e "${YELLOW}===== Test Summary =====${NC}"
echo "All tests completed. Check results above for any failures."
echo "To run more comprehensive tests, consider using the Rust unit and integration tests."
echo ""