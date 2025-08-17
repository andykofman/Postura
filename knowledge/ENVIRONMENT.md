# Environment

- Python 3.10 (Colab default).
- OS: Debian-based (Colab). Windows/macOS dev OK; WSL2 recommended for parity.
- CPU only for current BlazePose usage; GPU optional.

Colab runtime requirements:

- `libgl1`, `libglib2.0-0`, `ffmpeg` via apt.
- Remove preinstalled Colab packages that can force `protobuf>=5`:
  - `pip uninstall -y ydf grpcio-status`

Docker (future):

- See `DOCKER_READINESS.md` for base images and OS libs.
