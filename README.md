# 🚀 Multimodal PDF Data Extraction & RAG System

## Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system for extracting and analyzing multimodal PDF documents. Leveraging advanced machine learning techniques, the system can:

- Extract text, tables, and charts from PDFs
- Create semantic embeddings
- Enable intelligent querying across document collections

## 🌟 Key Features

- 📄 Multimodal PDF Parsing
- 🔍 Semantic Document Retrieval
- 🤖 AI-Powered Question Answering
- 📊 Prometheus Metrics Monitoring
- 🌐 FastAPI Backend

## Prerequisites

- Python 3.10.12 (recommended)
- pip
- Virtual environment support
- Tesseract OCR (optional but recommended)

## Installation

### 1. Set up Python environment

```bash
# Install correct Python version
chmod +x setup_python.sh
./setup_python.sh

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Install PyTorch first
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/rocm5.7

# Install other dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Clone the Repository

```bash
git clone https://github.com/pejakovic/multimodal-rag
```
