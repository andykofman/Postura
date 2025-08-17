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
from api.schemas import AnalyzeResponse, JobSubmitResponse, JobStatusResponse
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
    burst_preframes: int = 0,
    burst_postframes: int = 0,
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
            burst_forward = 0
            # Keep a small ring buffer of preframes to flush when we decide to send
            from collections import deque
            prebuf = deque(maxlen=max(0, int(burst_preframes)))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            JOB_FRAMES[CURRENT_JOB_ID.get()] = max(1, total)
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
                        if not send and (motion_score >= motion_threshold or skipped >= max_consecutive_skips or burst_forward > 0):
                            send = True
                        if send:
                            last_small = gray
                        else:
                            skipped += 1
                    except Exception:
                        # If motion calc fails, fall back to decimation decision
                        pass
                idx += 1
                # Maintain pre-buffer for potential burst
                if not send:
                    prebuf.append(frame)
                    if burst_forward > 0:
                        send = True
                    else:
                        continue
                # Flush prebuffer when sending this frame
                if prebuf:
                    while prebuf:
                        f = prebuf.popleft()
                        f_out = f
                        if target_width and target_width > 0:
                            h, w = f_out.shape[:2]
                            if w > target_width:
                                scale = float(target_width) / float(w)
                                new_w = int(round(w * scale))
                                new_h = int(round(h * scale))
                                f_out = cv2.resize(f_out, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        frame_queue.put(f_out, block=True)
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
                DONE_FRAMES[CURRENT_JOB_ID.get()] = DONE_FRAMES.get(CURRENT_JOB_ID.get(), 0) + 1
                # Enable forward burst of consecutive frames after a send
                if burst_postframes > 0:
                    burst_forward = burst_postframes
                else:
                    burst_forward = 0
        finally:
            # Signal end of stream
            frame_queue.put(None)

    prod_thread = threading.Thread(target=producer, name="decoder-producer", daemon=True)
    prod_thread.start()

    # Build backend; if mediapipe is unavailable, surface an explicit error
    backend: Optional[PoseBackend]
    try:
        backend = PoseBackend(model_complexity=model_complexity)
    except ImportError as exc:
        raise RuntimeError("mediapipe is not available in the API runtime") from exc

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


# In-memory job registry and progress counters (simple, for Colab). For production use a proper queue.
JOBS: dict[str, dict] = {}
JOB_FRAMES: dict[str, int] = {}
DONE_FRAMES: dict[str, int] = {}

# Track current job id within iterator threads
class _JobCtx:
    _id: Optional[str] = None
    def set(self, vid: str) -> None:
        self._id = vid
    def get(self) -> str:
        return self._id or ""

CURRENT_JOB_ID = _JobCtx()


@app.post("/analyze", response_model=JobSubmitResponse, status_code=202)
async def analyze(
    file: UploadFile = File(...),
):
    # Validate content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES and not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=415, detail="Unsupported media type; expected MP4")

    # Size guard
    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large; max 50MB")

    # Create job
    import uuid as _uuid
    video_id = str(_uuid.uuid4())
    JOBS[video_id] = {"status": "queued", "error": None}

    def worker(vid: str, payload: bytes) -> None:
        JOBS[vid] = {"status": "processing", "error": None}
        try:
            CURRENT_JOB_ID.set(vid)
            JOB_FRAMES[vid] = 0
            DONE_FRAMES[vid] = 0
            frames_iter = _iter_frames_from_video_bytes(payload)
            result = analyze_video(frames_iter)
            # Persist report
            report_dir = Path("report") / str(result.get("video_id", vid))
            report_dir.mkdir(parents=True, exist_ok=True)
            import json
            with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            JOBS[vid] = {"status": "done", "error": None, "video_id": result.get("video_id")}
        except Exception as exc:
            JOBS[vid] = {"status": "error", "error": str(exc)}

    threading.Thread(target=worker, args=(video_id, body), daemon=True).start()
    return JobSubmitResponse(video_id=video_id, status="queued")


@app.get("/status/{video_id}", response_model=JobStatusResponse)
async def status(video_id: str):
    job = JOBS.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown video_id")
    # Progress & ETA if available
    total = JOB_FRAMES.get(video_id)
    done = DONE_FRAMES.get(video_id)
    progress = None
    eta = None
    if total and done is not None:
        progress = max(0.0, min(1.0, float(done) / float(total)))
        # ETA not tracked precisely; leave None or estimate elsewhere
    return JobStatusResponse(
        video_id=video_id,
        status=job.get("status", "unknown"),
        detail=job.get("error"),
        progress=progress,
        eta_seconds=eta,
    )


@app.get("/result/{video_id}")
async def result(video_id: str):
    job = JOBS.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown video_id")
    if job.get("status") != "done":
        raise HTTPException(status_code=425, detail="Result not ready")
    # Serve saved JSON
    actual_id = job.get("video_id") or video_id
    p = Path("report") / str(actual_id) / "summary.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="summary.json missing")
    import json
    return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))


