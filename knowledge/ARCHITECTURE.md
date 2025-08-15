*** Begin Patch
*** Add File: docs/knowledge-base/ARCHITECTURE.md
# Architecture (Current)

High-level components:

- `pose/`
  - `backend.py`: `PoseBackend` wrapping MediaPipe BlazePose (CPU), returns 33 normalized keypoints.
  - `draw.py`: Utilities to draw keypoints and connections; video rendering with OpenCV.
  - `smoothing.py`: (reserved) smoothing/filtering (currently empty).
- `analysis/`:
  - `analyzer.py`: dataclasses and result handling for keypoints (currently focused on types).
  - `features.py`, `fsm_pushup.py`, `fsm_squat.py`: stubs/placeholders for future analysis.
- `api/`: scaffolding for a future API (files present but empty).
- `tests/`:
  - Unit tests use fake models to avoid MediaPipe dependency.
  - Integration test uses real MediaPipe + OpenCV on `tests/input.mp4` (or `demo/input.mp4`).

Data flow (typical batch run):
1) Read frames with OpenCV.
2) Inference via `PoseBackend` (BGR→RGB inside).
3) Draw keypoints/links and write annotated video.
4) Optionally compute features/metrics (future).
*** End Patch