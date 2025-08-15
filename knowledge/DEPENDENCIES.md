
# Dependencies and Pins

Top-level Python packages:

- `numpy==1.26.4`
  - Stable, wide binary availability for Python 3.10.
- `mediapipe==0.10.14`
  - Matches existing project usage; compatible with Python 3.10.
- `opencv-python==4.10.0.84`
  - OpenCV core; works with `libgl1` and `ffmpeg` for video IO.
- `pytest==8.2.2`
  - For running tests in Colab and CI.

Critical transitive pins in `constraints.txt`:

- `protobuf==4.25.3`
  - MediaPipe requires protobuf <5; this version is stable and widely used.
- `absl-py==2.1.0`
  - Common MediaPipe dependency.
- `attrs==24.2.0`
  - Safe, broadly compatible version.
- `flatbuffers==23.5.26`
  - Used by MediaPipe underneath.

System packages (Debian/Ubuntu):

- `libgl1`, `libglib2.0-0` — required by OpenCV GUI/x functionality even in headless mode.
- `ffmpeg` — needed for OpenCV’s video read/write codecs.

Regenerating constraints later (recommended):

- Use a clean virtualenv and `pip-compile` (pip-tools) to lock all transitive deps:
  - `pip install pip-tools`
  - `pip-compile --generate-hashes --resolver=backtracking --output-file constraints.txt requirements.txt`
- Commit the resulting `constraints.txt`.

Notes:

- Keep Python at 3.10 to avoid wheel compatibility issues with MediaPipe.
