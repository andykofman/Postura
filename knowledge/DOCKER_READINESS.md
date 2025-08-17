# Dockerization (CPU)

This project now includes a production-ready, CPU-only Docker image for the FastAPI service.

## Image design

- Base: `python:3.11-slim`
- System packages: `ffmpeg`, `libsm6`, `libxext6` (OpenCV runtime)
- Python deps: installed via `requirements.txt` with strict `constraints.txt`
- Server: `uvicorn api.main:app --host 0.0.0.0 --port 5000`
- Port: `5000`
- Build context kept small via `.dockerignore` (excludes notebooks, media, tests, caches)

Minimal `Dockerfile` (already in repo root):

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt
COPY . .
EXPOSE 5000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]
```

## Build and run

```bash
docker build -t postura:cpu .
docker images postura
docker run --rm -p 5000:5000 postura:cpu
```

Verify: open `http://localhost:5000/ui`.

## Image size and optimization

- Slim base + no pip cache + `.dockerignore` → small CPU image.
- Multi-stage does not meaningfully reduce size (no native builds). To shrink further:
  - Remove heavy libs not used by the API (e.g., `matplotlib`).
  - Keep large artifacts out of context (already done).

Check size:

```bash
docker images postura
docker image inspect postura:cpu --format '{{.Size}}'
```
