
# Postura

Colab-first posture analysis with MediaPipe BlazePose and OpenCV.

- Colab quickstart: see `COLAB_SETUP.md`
- Dockerization setup: see `DOCKER_READINESS.md`
- Architecture and environment details: see `docs/knowledge-base/`

Assumptions:
- Colab Python 3.10 runtime (default on most Colab instances at the time of writing).
- CPU usage by default; GPU not required for MediaPipe BlazePose CPU variant.

Quick start on Colab:
1) Set runtime (CPU or GPU) in Colab.
2) Run the cells from `COLAB_SETUP.md` to:
   - Clone this repo
   - Install system packages (ffmpeg, libgl1, libglib2.0-0)
   - Install Python deps via `requirements.txt` and `constraints.txt`
   - Run `scripts/colab_bootstrap.py` to verify imports and a minimal inference sanity test

Outputs & persistence:
- Use Google Drive to persist outputs (e.g., annotated videos under `demo/` or `report/`).
- The setup doc shows how to mount Drive and set an output directory.

Note on Docker:
- All legacy MediaPipe-only Docker artifacts were removed.
- When we later add a full-project Dockerfile, follow `DOCKER_READINESS.md`.
*** End Patch