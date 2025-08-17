
# Docker Readiness (No Dockerfile Yet)

This defines the future full-project Dockerization strategy. Do not add a Dockerfile yet.

Assumptions:

- Python 3.10 baseline.
- CPU image first; optional GPU variant.

## Base Image Candidates

- CPU: `python:3.10-slim`
- GPU (optional): `nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04` (requires NVIDIA Container Toolkit)

## Minimal OS Packages (Debian/Ubuntu)

- libgl1
- libglib2.0-0
- ffmpeg
- ca-certificates
- curl
- git (if cloning during build)
- build-essential (optional; only if native builds are needed)

## Environment Variables

- PYTHONDONTWRITEBYTECODE=1
- PYTHONUNBUFFERED=1
- PIP_NO_CACHE_DIR=1
- POSTURA_ENV=docker

## Python Dependency Strategy

- Copy `requirements.txt` and `constraints.txt` first, then `pip install` to leverage layer cache.
- `pip install --upgrade pip && pip install -r requirements.txt -c constraints.txt`
- Consider BuildKit pip cache: `--mount=type=cache,target=/root/.cache/pip`

## Entrypoint Design

- `scripts/entrypoint.sh` (future):
  - MODE=api → start API service (future)
  - MODE=batch → run batch (e.g., `python -m pose.backend` or a CLI)
- Default CMD: MODE=batch
- Proper exit codes and graceful shutdown.

## Healthcheck Concept

- API mode: HTTP `/healthz` every 30s; fail on non-2xx.
- Batch mode: healthcheck disabled or “process alive” check.

## GPU Handling (optional)

- Host: NVIDIA drivers + NVIDIA Container Toolkit.
- Run: `docker run --gpus all ...`
- If we add GPU MediaPipe later, pin versions compatible with the CUDA base.

## Models and Data

- MediaPipe BlazePose models are included in the pip package.
- Keep image small; fetch data at start or mount volumes.

## Ports and Runtime Args

- API mode: expose 8000 (future).
- Batch mode: no ports.
- Batch CLI should support `--input` and `--output`.

## Layer Caching Strategy

1) FROM base
2) Install OS packages
3) Copy pins
4) pip install
5) Copy source
6) Set entrypoint/CMD

## Repro Steps (CPU First)

- `docker build -t postura:cpu .`
- `docker run --rm postura:cpu MODE=batch`
- GPU later: add `--gpus all`.

## Acceptance Criteria for the Future Dockerfile

- Deterministic builds with pinned deps.
- Minimal OS libs only.
- Non-root runtime where feasible.
- API healthcheck available.
- Reproducible outputs across machines and CI.
- Works offline after build (no network at runtime).
