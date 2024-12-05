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
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch based on runtime
RUN if [ "$RUNTIME" = "cpu" ]; then \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7; \
    fi

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

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

# Run the application
CMD ["python3", "src/rag_arxiv.py", "--mode", "api"]