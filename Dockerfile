# syntax=docker/dockerfile:1

# Base: slim Python image for smaller footprint
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for OpenCV and media handling
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy application code
COPY . .

# Expose API port
EXPOSE 5000

# Default start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]


