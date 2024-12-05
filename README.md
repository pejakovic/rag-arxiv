# RAG System for arXiv Papers

A Retrieval-Augmented Generation (RAG) system that uses arXiv papers as a knowledge base, with FastAPI endpoints and Prometheus/Grafana monitoring.
This project is a simple implementation of a RAG system for arXiv papers for educational and demonstration purposes.

## Features

- RAG implementation using LangChain and HuggingFace models
- FastAPI endpoints for querying the system
- Prometheus metrics monitoring
- Grafana dashboards
- Docker containerization
- Support for both CPU and ROCm (AMD GPU) environments

## Prerequisites

- Docker and Docker Compose
- At least 8GB RAM
- 20GB disk space (for models and papers)
- (Optional) AMD GPU for ROCm support

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Initialize the system (download papers and create database):
```bash
python src/rag_arxiv.py --mode init --search-query "federated learning" --papers-dir "data/papers" --dataset-path "data/dataset.jsonl"
```

3. Train the model:
```bash
python src/rag_arxiv.py --mode train --dataset-path "data/dataset.jsonl" --model_output_dir "models/fine_tuned"
```

4. Run inference:
```bash
python src/rag_arxiv.py --mode inference --query "What is federated learning?"
```

5. Start the services using Docker Compose:
```bash
docker compose up -d
```

## Available Endpoints

- RAG API: http://localhost:8000
  - API Documentation: http://localhost:8000/docs
  - Health Check: http://localhost:8000/health
  - Query Endpoint: http://localhost:8000/query (POST)
- Prometheus Metrics: http://localhost:8001
- Prometheus UI: http://localhost:9090
- Grafana: http://localhost:3000 (default login: admin/admin)

## API Usage

Query the RAG system:

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{"query": "What is federated learning?"}'
```

## Directory Structure

```
├── data/
│ ├── arxiv_papers/ # Downloaded papers
│ └── dataset.jsonl # Processed data
├── models/
│ ├── fine_tuned/ # Fine-tuned models
│ ├── cache/ # Transformers cache
│ └── torch/ # PyTorch models
├── src/
│ ├── rag_arxiv.py # Main implementation
│ ├── args.py # Argument parsing
│ ├── monitoring.py # Prometheus metrics
│ └── ...
├── logs/ # Application logs
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml # Prometheus configuration
└── requirements.txt
```

## Configuration

### Environment Variables

- `PYTHONPATH=/app`
- `TRANSFORMERS_CACHE=/app/models/cache`
- `TORCH_HOME=/app/models/torch`

### Docker Configuration

- CPU version: Uses `ubuntu:22.04` base image
- ROCm version: Uses `rocm/dev-ubuntu-22.04:5.7` base image

To use ROCm version:
```bash
docker-compose build --build-arg RUNTIME=rocm
```


## Monitoring

The system exposes the following metrics:
- Request counts
- Query latency
- Document retrieval duration
- Model inference duration
- Error rates

Access these metrics in Grafana by:
1. Login to Grafana (http://localhost:3000)
2. Add Prometheus as a data source
3. Import dashboards

## Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the system:
```bash
python src/rag_arxiv.py --mode init --search-query "federated learning" --papers-dir "data/papers" --dataset-path "data/dataset.jsonl"
```

## Troubleshooting

- If you encounter issues with the ROCm version, ensure your GPU drivers are correctly installed and compatible with the specified ROCm version.
- Check the logs in `logs/` for any errors or warnings that might help in diagnosing the issue.

```bash
docker-compose logs
```


2. Check individual service logs:
```bash
docker-compose logs rag_app
docker-compose logs prometheus
docker-compose logs grafana
```

3. Restart services:
```bash
docker-compose restart [service_name]
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.