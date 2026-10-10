#!/bin/bash
set -e

CONTAINER_NAME="tensorforge-service"
IMAGE_NAME="tensorforge-team-xturbo:v1.2"
API_KEY="${API_KEY:-tf2_842bc2c62f969cfe2b8d9be71606bd1a510ede5fc7ffc8e2}"

echo "=========================================================="
echo " Executing Full Automated Test Suite Inside Container     "
echo "=========================================================="

# If container is not running, start an ephemeral test container
EPHEMERAL=false
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Starting ephemeral test container..."
    EPHEMERAL=true
    CONTAINER_NAME="tf2_test_runner"
    docker rm -f ${CONTAINER_NAME} 2>/dev/null || true
    docker run -d \
        --name ${CONTAINER_NAME} \
        --network none \
        -p 8000:8000 \
        -e API_KEY="${API_KEY}" \
        ${IMAGE_NAME}
    sleep 2
fi

echo "Running test suite..."
docker exec ${CONTAINER_NAME} python3.11 /app/tests/test_api_endpoints.py

if [ "$EPHEMERAL" = true ]; then
    echo "Cleaning up ephemeral container..."
    docker rm -f ${CONTAINER_NAME} >/dev/null
fi

echo "All tests executed successfully."

