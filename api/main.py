from __future__ import annotations

import os
import threading
from queue import Queue
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from analysis.analyzer import analyze_video
from api.schemas import AnalyzeResponse
from pose.backend import PoseBackend


# Threading/env tuning to avoid oversubscription on Colab CPUs
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


app = FastAPI(title="Postura API")


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_CONTENT_TYPES = {"video/mp4", "application/octet-stream"}

# Mount static web UI and reports directory for end-to-end testing
WEB_ROOT = Path("web").resolve()
REPORT_ROOT = Path("report").resolve()
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
if WEB_ROOT.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_ROOT), html=True), name="ui")
app.mount("/reports", StaticFiles(directory=str(REPORT_ROOT)), name="reports")


def _configure_opencv_threads() -> None:
    try:
        import cv2  # type: ignore
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(2)
        except Exception:
            pass
    except Exception:
        pass


def _iter_frames_from_video_bytes(data: bytes):
    import cv2  # type: ignore
    _configure_opencv_threads()
    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / "upload.mp4"
    with open(tmp_path, "wb") as f:
        f.write(data)

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open uploaded video")

    # Producer-consumer pipeline: overlap decode (producer) and inference (consumer)
    frame_queue: Queue[Optional[np.ndarray]] = Queue(maxsize=16)

    def producer() -> None:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_queue.put(frame, block=True)
        finally:
            # Signal end of stream
            frame_queue.put(None)

    prod_thread = threading.Thread(target=producer, name="decoder-producer", daemon=True)
    prod_thread.start()

    # Build backend with fallback when mediapipe is unavailable (e.g., unit tests)
    backend: Optional[PoseBackend]
    try:
        backend = PoseBackend()
    except ImportError:
        class _NoModel:
            def process(self, frame_rgb):
                class _R:
                    pose_landmarks = None
                return _R()
        backend = PoseBackend(pose_model=_NoModel())

    try:
        with backend as b:
            while True:
                frame = frame_queue.get(block=True)
                if frame is None:
                    break
                kps = b.infer(frame)
                yield kps
    finally:
        cap.release()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    # Validate content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES and not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=415, detail="Unsupported media type; expected MP4")

    # Size guard
    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large; max 50MB")

    # Decode and analyze
    try:
        frames_iter = _iter_frames_from_video_bytes(body)
        result = analyze_video(frames_iter)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to analyze video: {exc}")

    # Persist report
    video_id = str(result.get("video_id"))
    report_dir = Path("report") / video_id
    report_dir.mkdir(parents=True, exist_ok=True)
    # Write JSON
    import json

    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return JSONResponse(content=result)


