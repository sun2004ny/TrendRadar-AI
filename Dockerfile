# TrendRadar-AI Dockerfile
# Base image: Python 3.11 slim
FROM python:3.11-slim

# Prevent interactive prompts and set Python runtime performance flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies needed for compiling C-extensions (hdbscan, etc.) and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and configurations
COPY . .

# Expose Streamlit dashboard port
EXPOSE 8501

# Healthcheck to verify Streamlit responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch application via run.py launcher
CMD ["python", "run.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
