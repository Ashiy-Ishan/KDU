#!/bin/bash
set -e

IMAGE_NAME="tensorforge-team-xturbo:v1.2"
CONTAINER_NAME="tensorforge-service"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-tf2_842bc2c62f969cfe2b8d9be71606bd1a510ede5fc7ffc8e2}"

echo "=========================================================="
echo " Starting TensorForge 2.0 MVP Service (Team XTurbo)       "
echo "=========================================================="

# Build if image does not exist
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "Building Docker image ${IMAGE_NAME}..."
    docker build -t ${IMAGE_NAME} .
fi

# Stop existing container if running
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

echo "Running container (offline mode)..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:8000 \
    -e API_KEY="${API_KEY}" \
    ${IMAGE_NAME}

echo "Container started. Waiting for readiness..."
sleep 2

docker logs ${CONTAINER_NAME}

echo ""
echo "Service is live:"
echo "  • Public Health: http://localhost:${PORT}/health"
echo "  • Web Demo:      http://localhost:${PORT}/"
echo "  • Predict:       POST http://localhost:${PORT}/predict"
echo "  • Batch Predict: POST http://localhost:${PORT}/predict/batch"
echo "  • Async Jobs:    POST http://localhost:${PORT}/batch/jobs"

