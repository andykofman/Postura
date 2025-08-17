from __future__ import annotations

import os
import threading
from queue import Queue
from pathlib import Path
from typing import Optional, Iterator, Sequence

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from analysis.analyzer import analyze_video
from api.schemas import AnalyzeResponse
from pose.backend import PoseBackend
from pose.backend import Keypoint


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


def _iter_frames_from_video_bytes(
    data: bytes,
    *,
    target_fps: Optional[float] = None,
    target_width: Optional[int] = None,
    model_complexity: int = 2,
    adaptive_motion: bool = False,
    motion_resize_width: int = 0,
    motion_threshold: float = 0.0,
    max_consecutive_skips: int = 0,
) -> Iterator[Sequence[Optional[Keypoint]]]:
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
    # Determine decimation stride if requested
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    stride = 1
    if target_fps and target_fps > 0 and orig_fps > 0:
        stride = max(1, int(round(orig_fps / float(target_fps))))

    def producer() -> None:
        try:
            idx = 0
            last_small = None
            skipped = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                send = True
                # Decimate frames to target_fps if configured
                if stride > 1 and (idx % stride) != 0:
                    send = False
                # Motion-adaptive override (fast mode): if motion is high, force send
                if adaptive_motion:
                    try:
                        import cv2  # type: ignore
                        small_w = motion_resize_width or 0
                        if small_w > 0:
                            h, w = frame.shape[:2]
                            scale = float(small_w) / float(w)
                            new_w = small_w
                            new_h = max(1, int(round(h * scale)))
                            small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        else:
                            small = frame
                        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                        if last_small is None:
                            motion_score = 999.0
                        else:
                            # mean absolute difference
                            diff = cv2.absdiff(gray, last_small)
                            motion_score = float(diff.mean())
                        # Decide to send if motion is strong or we already skipped several frames
                        if not send and (motion_score >= motion_threshold or skipped >= max_consecutive_skips):
                            send = True
                        if send:
                            last_small = gray
                        else:
                            skipped += 1
                    except Exception:
                        # If motion calc fails, fall back to decimation decision
                        pass
                idx += 1
                if not send:
                    continue
                skipped = 0
                # Resize to target_width if configured
                if target_width and target_width > 0:
                    h, w = frame.shape[:2]
                    if w > target_width:
                        scale = float(target_width) / float(w)
                        new_w = int(round(w * scale))
                        new_h = int(round(h * scale))
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame_queue.put(frame, block=True)
        finally:
            # Signal end of stream
            frame_queue.put(None)

    prod_thread = threading.Thread(target=producer, name="decoder-producer", daemon=True)
    prod_thread.start()

    # Build backend with fallback when mediapipe is unavailable (e.g., unit tests)
    backend: Optional[PoseBackend]
    try:
        backend = PoseBackend(model_complexity=model_complexity)
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
async def analyze(
    file: UploadFile = File(...),
    fast: int = Query(0, description="Enable speed optimizations (decimation+resize+lighter model). 0=off,1=on"),
):
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
        if fast:
            # Fast mode with motion-adaptive sampling to retain more key frames
            frames_iter = _iter_frames_from_video_bytes(
                body,
                target_fps=12.0,
                target_width=640,   # keep a bit more resolution for robustness
                model_complexity=1,
                adaptive_motion=True,
                motion_resize_width=256,
                motion_threshold=2.0,
                max_consecutive_skips=3,
            )
        else:
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


