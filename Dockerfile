# Use ARG to make base image configurable
ARG RUNTIME=cpu

# CPU base image
FROM ubuntu:22.04 AS cpu-base
# ROCm base image
FROM rocm/dev-ubuntu-22.04:5.7 AS rocm-base

# Select final base image based on RUNTIME arg
FROM ${RUNTIME}-base

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch and torchvision with specific versions
RUN if [ "$RUNTIME" = "cpu" ]; then \
        pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/rocm5.7; \
    fi

# Verify PyTorch installation and CUDA availability
RUN python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Install basic requirements first
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    typing-extensions

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install spacy model
RUN python3 -m spacy download en_core_web_sm

# Copy the application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p logs data/arxiv_papers models/fine_tuned

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/models/cache
ENV TORCH_HOME=/app/models/torch

# Expose ports for the API and Prometheus metrics
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "src/rag_arxiv.py", "--mode", "api"]