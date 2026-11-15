# Production Docker image for BDBV Host Prediction
# Ensures bit-for-bit reproducible environment across systems

FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing and bioinformatics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    curl \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables for reproducibility
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=42 \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=4

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (exact versions pinned)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create directories for data, models, results
RUN mkdir -p /app/data/{raw,processed,external} \
    /app/results/{figures,models,reports} \
    /app/logs

# Set up volumes for data and results
VOLUME /app/data
VOLUME /app/results

# Default command: run training pipeline
ENTRYPOINT ["python"]
CMD ["scripts/run_training.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.pipelines.training_pipeline import TrainingPipeline; print('OK')" || exit 1

# Add labels for metadata
LABEL maintainer="bdbv-prediction@example.com" \
      version="1.0.0" \
      description="BDBV Host Susceptibility Prediction - Production Docker Image" \
      reproducibility="Pinned dependencies for exact reproducibility"
