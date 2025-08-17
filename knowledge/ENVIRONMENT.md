# Environment

- Python 3.11 (Docker image). Colab flows may use 3.10.
- OS: Debian-based. Windows dev via Docker Desktop + WSL2 recommended for parity.
- CPU only for current BlazePose usage; GPU optional.

Runtime behavior:

- API saves artifacts under `report/{video_id}/` and serves them via `/reports`.
- PDF is generated on demand at `/report/{video_id}.pdf`.
- Results landing page at `/ui/result.html?video_id={id}` provides one-click access to both.

Colab runtime requirements:

- `libgl1`, `libglib2.0-0`, `ffmpeg` via apt.
- Remove preinstalled Colab packages that can force `protobuf>=5`:
  - `pip uninstall -y ydf grpcio-status`

Docker:

- See `DOCKER_READINESS.md` for the finalized Dockerfile, OS libs, and run instructions.
