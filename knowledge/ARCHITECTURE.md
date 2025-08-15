

# Architecture (Current)

- `pose/`
  - `backend.py`: `PoseBackend` wrapping MediaPipe BlazePose (CPU).
  - `draw.py`: drawing utilities and annotated video rendering with OpenCV.
  - `smoothing.py`: per-joint EMA smoother (`EmaSmoother`) that treats None/low-confidence
    joints as missing and outputs NaN for those, along with a validity mask.
- `analysis/`
  - `analyzer.py`: dataclasses and types for keypoints.
  - `features.py`: geometric features with NaN-safe behavior:
    - `angle(A,B,C)` returns radians at B; NaN on invalid inputs
    - `horiz_offset(knee, ankle)` signed x-offset; NaN on invalid
    - `collinearity_residual(hip, shoulder, ankle)` signed perpendicular distance; NaN on invalid
  - `fsm_pushup.py`, `fsm_squat.py`: placeholders.
- `api/`: scaffolding for a future API.
- `tests/`
  - Unit tests use fake models (no MediaPipe).
  - Integration test uses real MediaPipe + OpenCV.

Flow:

1) Read frames with OpenCV.
2) Inference via `PoseBackend` (BGR→RGB internally).
3) Draw keypoints/links and write video.
4) Compute features/metrics from smoothed keypoints (NaNs propagate to signal invalid segments).
