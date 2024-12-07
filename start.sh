#!/bin/bash

# Multimodal PDF Data Extraction Project Startup Script

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project Configuration
PROJECT_NAME="Multimodal PDF RAG System"
PYTHON_VERSION="3.10.12"
VENV_NAME=".venv"

# Directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/${VENV_NAME}"
DATA_DIR="${PROJECT_ROOT}/data"
LOGS_DIR="${PROJECT_ROOT}/logs"
MODELS_DIR="${PROJECT_ROOT}/models"

# Dependency and Environment Setup
setup_environment() {
    echo -e "${GREEN}🚀 Setting up development environment for ${PROJECT_NAME}${NC}"

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found. Please install Python ${PYTHON_VERSION}${NC}"
        exit 1
    fi

    # Create necessary directories
    mkdir -p "${DATA_DIR}/arxiv_papers" "${LOGS_DIR}" "${MODELS_DIR}/fine_tuned"

    # Create virtual environment
    if [ ! -d "${VENV_PATH}" ]; then
        echo -e "${YELLOW}🔧 Creating virtual environment...${NC}"
        python3 -m venv "${VENV_PATH}"
    fi

    # Activate virtual environment
    source "${VENV_PATH}/bin/activate"

    # Upgrade pip and install basic dependencies first
    echo -e "${YELLOW}📦 Installing basic dependencies...${NC}"
    pip install --upgrade pip wheel setuptools
    
    # Install PyTorch CPU version first
    echo -e "${YELLOW}📦 Installing PyTorch...${NC}"
    pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/rocm5.7
    #--index-url https://download.pytorch.org/whl/cpu

    # Install spaCy and its model
    echo -e "${YELLOW}📦 Installing spaCy...${NC}"
    pip install spacy
    python -m spacy download en_core_web_sm

    # Install other dependencies
    echo -e "${YELLOW}📦 Installing remaining dependencies...${NC}"
    pip install -r requirements.txt

    echo -e "${GREEN}✅ Environment setup complete${NC}"
}

# Initialize Project Components
initialize_project() {
    echo -e "${YELLOW}🔍 Initializing project components${NC}"

    # Example initialization steps
    python src/rag_arxiv.py --mode init \
        --search-query "machine learning" \
        --papers-dir "${DATA_DIR}/arxiv_papers" \
        --dataset-path "${DATA_DIR}/dataset.jsonl"

    echo -e "${GREEN}✅ Project initialization complete${NC}"
}

# Train the Model
train_model() {
    echo -e "${YELLOW}🏋️ Training RAG model${NC}"

    python src/rag_arxiv.py --mode train \
        --dataset-path "${DATA_DIR}/dataset.jsonl" \
        --model-output-dir "${MODELS_DIR}/fine_tuned"

    echo -e "${GREEN}✅ Model training complete${NC}"
}

# Start Services
start_services() {
    echo -e "${GREEN}🌐 Starting project services${NC}"

    # Start Prometheus metrics server
    python -m prometheus_client.exposition &

    # Start RAG API
    python src/rag_arxiv.py --mode api &

    echo -e "${YELLOW}Services running:
    - RAG API: http://localhost:8000
    - Metrics: http://localhost:8001${NC}"
}

# Stop Services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping services${NC}"
    pkill -f "python src/rag_arxiv.py"
    pkill -f "prometheus_client.exposition"
}

# Try Multimodal RAG Example
try_multimodal() {
    echo -e "${YELLOW}🔬 Running Multimodal RAG Example${NC}"
    
    # Activate virtual environment
    source "${VENV_PATH}/bin/activate"
    
    # Create necessary directories
    mkdir -p "${DATA_DIR}/arxiv_papers" "${MODELS_DIR}/multimodal_finetuned" "${LOGS_DIR}/multimodal_training"
    
    # Run the example
    python src/examples/try_multimodal_rag.py
}

# Main Execution
main() {
    clear
    
    case "$1" in
        setup)
            setup_environment
            ;;
        init)
            #setup_environment
            initialize_project
            ;;
        train)
            #setup_environment
            train_model
            ;;
        start)
            #setup_environment
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            #setup_environment
            start_services
            ;;
        try)
            #setup_environment
            try_multimodal
            ;;
        *)
            echo -e "${RED}Usage: $0 {setup|init|train|start|stop|restart|try}${NC}"
            exit 1
    esac
}

# Make the script executable
chmod +x "$0"

# Execute main function with arguments
main "$@" 