#!/bin/bash

# Make script executable
chmod +x start.sh

# Create necessary directories if they don't exist
mkdir -p logs data/arxiv_papers models/fine_tuned

# Build and start all services
docker-compose up --build -d

# Show running containers
docker-compose ps

echo "Services are starting up..."
echo "RAG API will be available at: http://localhost:8000"
echo "Prometheus metrics at: http://localhost:8001"
echo "Prometheus UI at: http://localhost:9090"
echo "Grafana UI at: http://localhost:3000" 