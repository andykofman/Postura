

# Architecture (Current)

- `pose/`
  - `backend.py`: `PoseBackend` wrapping MediaPipe BlazePose (CPU).
  - `draw.py`: drawing utilities and annotated video rendering with OpenCV.
  - `smoothing.py`: per-joint EMA smoother (`EmaSmoother`) that treats None/low-confidence
    joints as missing and outputs NaN for those, along with a validity mask.
- `analysis/`
  - `analyzer.py`: `analyze_video(frames_iter)` computes per-frame features, feeds FSMs, and
    returns a JSON-ready dict with `video_id` (uuid4), `frame_data`, and per-exercise summaries.
  - `features.py`: geometric features with NaN-safe behavior:
    - `angle(A,B,C)` returns radians at B; NaN on invalid inputs
    - `horiz_offset(knee, ankle)` signed x-offset; NaN on invalid
    - `collinearity_residual(hip, shoulder, ankle)` signed perpendicular distance; NaN on invalid
  - `fsm_pushup.py`, `fsm_squat.py`: hysteresis-based FSMs counting reps.
- `api/`: FastAPI backend and web serving.
  - `api/main.py`: endpoints
    - `POST /analyze` → returns job id and runs background processing
    - `GET /status/{video_id}` → live progress (frames, fps, eta)
    - `GET /result/{video_id}` → merged JSON (summary + thumbnails)
    - `GET /logs/{video_id}` → streaming logs
    - `GET /frame/{video_id}/{frame_index}` → single frame JPEG
    - `GET /report/{video_id}.pdf` → on‑the‑fly PDF generation
  - Static mounts:
    - `/ui` → `web/` (front‑end)
    - `/reports` → `report/` (artifacts per `video_id`)
  - Persistence layout under `report/{video_id}/`:
    - `input.mp4`, `summary.json`, `thumbnails.json`, `thumbs/*.jpg`
- `tests/`
  - Unit tests use fake models (no MediaPipe).
  - Integration test uses real MediaPipe + OpenCV.

Front‑end:

- `web/index.html` – animated hero landing page.
- `web/app.html` – analyzer UI with upload, stages, progress, live logs, green links.
- `web/result.html` – dedicated results page for PDF and JSON, opened via a completion modal.

Flow:

1) Read frames with OpenCV.
2) Inference via `PoseBackend` (BGR→RGB internally).
3) Draw keypoints/links and write video.
4) Compute features/metrics from smoothed keypoints (NaNs propagate to signal invalid segments).
