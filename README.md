# Postura

Fast, CPU‑friendly posture analysis using MediaPipe BlazePose and OpenCV, with a simple FastAPI backend and a lightweight web UI.

## Choose your path (quick start)

### A) Colab notebook (recommended)

See `knowledge/COLAB_SETUP.md (Postura_Live_Demo.ipynb)` for a Colab‑first workflow (installs apt packages and pinned Python deps).

### B) Run locally (Python venv)

```bash
python -m venv .venv
. .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -c constraints.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000/docs`.

### C) Run with Docker

Requirements: Docker Desktop. On Windows, enable WSL2.

```bash
docker build -t postura:cpu .
docker run --rm -p 5000:5000 postura:cpu
```

Open the interactive docs at `http://localhost:5000/docs`.

## What’s inside (project layout)

- `api/` — FastAPI service (`api.main:app`) serving analysis endpoints and static UI
- `analysis/` — feature extraction, finite‑state machines for rep counting, report generation
- `pose/` — MediaPipe BlazePose backend, smoothing, drawing
- `web/` — minimal UI: upload, progress, results
- `tests/` — unit and integration tests
- `demo/` — sample media and generated summaries/reports
- `report/` — per‑run artifacts created by the API (JSON, thumbnails, annotated video)

## Core API endpoints

- `POST /analyze` — upload a video for analysis, returns a job id
- `GET /status/{video_id}` — live progress (frames, fps, ETA)
- `GET /result/{video_id}` — merged JSON (summary + thumbnails)
- `GET /report/{video_id}.pdf` — on‑the‑fly PDF
- Static mounts: `/ui` → `web/`, `/reports` → `report/`

## Development and testing

Run the test suite:

```bash
pytest -q
```

Coding guidelines are documented inline; type hints are preferred on public APIs. See `pytest.ini` for test config.

## Dockerization details

CPU‑only image based on `python:3.11-slim`, installs `ffmpeg`, `libsm6`, `libxext6`, and pinned Python dependencies. Exposes `5000` and starts uvicorn for `api.main:app`.

- Dockerfile: repo root (`Dockerfile`)
- Build & run: see quick start above
- More details (Windows/WSL2 tips, optimization): `knowledge/DOCKER_READINESS.md`

## Knowledge base (start here)

- `knowledge/INDEX.md` — table of contents for all docs
- `knowledge/ARCHITECTURE.md` — modules, data flow, and endpoints
- `knowledge/ENVIRONMENT.md` — runtimes (Docker, Colab), OS notes
- `knowledge/DEPENDENCIES.md` — pinned versions and system packages
- `knowledge/FORM_EVALUATION.md` — metrics and interpretation
- `knowledge/testing_pose_backend.md` — backend testing notes

## Demo

Sample videos and generated outputs live in `demo/`. The API writes artifacts under `report/{video_id}/` and serves them via `/reports`.

## Troubleshooting

- MediaPipe / protobuf conflicts: ensure `constraints.txt` is used; avoid `protobuf>=5`.
- Windows + Docker: ensure WSL2 is enabled and a Linux distro (Ubuntu) is installed and running.
