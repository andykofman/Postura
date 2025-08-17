from __future__ import annotations

import os
import threading
from queue import Queue
from pathlib import Path
from typing import Optional, Iterator, Sequence

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, PlainTextResponse
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


def _iter_frames_from_video_file(
    video_path: Path,
    *,
    target_fps: Optional[float] = None,
    target_width: Optional[int] = None,
    model_complexity: int = 2,
    on_start: Optional[callable] = None,
    on_progress: Optional[callable] = None,
) -> Iterator[Sequence[Optional[Keypoint]]]:
    import cv2  # type: ignore
    _configure_opencv_threads()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open uploaded video")

    # Producer-consumer pipeline: overlap decode (producer) and inference (consumer)
    frame_queue: Queue[Optional[np.ndarray]] = Queue(maxsize=16)
    # Determine decimation stride if requested
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = 1
    if target_fps and target_fps > 0 and orig_fps > 0:
        stride = max(1, int(round(orig_fps / float(target_fps))))

    def producer() -> None:
        try:
            idx = 0
            if callable(on_start):
                try:
                    on_start(total_frames)
                except Exception:
                    pass
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                send = not (stride > 1 and (idx % stride) != 0)
                idx += 1
                if not send:
                    continue
                # Resize to target_width if configured
                if target_width and target_width > 0:
                    h, w = frame.shape[:2]
                    if w > target_width:
                        scale = float(target_width) / float(w)
                        new_w = int(round(w * scale))
                        new_h = int(round(h * scale))
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame_queue.put(frame, block=True)
                if callable(on_progress):
                    try:
                        on_progress(idx)
                    except Exception:
                        pass
        finally:
            # Signal end of stream
            frame_queue.put(None)

    prod_thread = threading.Thread(target=producer, name="decoder-producer", daemon=True)
    prod_thread.start()

    # Build backend
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


JOBS: dict[str, dict] = {}
JOB_LOGS: dict[str, list[str]] = {}
JOB_FRAMES_TOTAL: dict[str, int] = {}
JOB_FRAMES_DONE: dict[str, int] = {}
JOB_START_TS: dict[str, float] = {}


def _log(vid: str, msg: str) -> None:
    JOB_LOGS.setdefault(vid, []).append(msg)
    report_dir = REPORT_ROOT / vid
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "logs.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@app.post("/analyze", response_model=JobSubmitResponse, status_code=202)
async def analyze(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES and not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=415, detail="Unsupported media type; expected MP4")
    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large; max 50MB")

    import uuid as _uuid, time
    vid = str(_uuid.uuid4())
    JOBS[vid] = {"status": "queued"}
    JOB_LOGS[vid] = []
    JOB_FRAMES_TOTAL[vid] = 0
    JOB_FRAMES_DONE[vid] = 0
    JOB_START_TS[vid] = time.time()

    report_dir = REPORT_ROOT / vid
    report_dir.mkdir(parents=True, exist_ok=True)
    video_path = report_dir / "input.mp4"
    with open(video_path, "wb") as f:
        f.write(body)

    def worker(job_id: str, path: Path) -> None:
        import time, json
        JOBS[job_id] = {"status": "processing"}
        _log(job_id, "Job started")

        def on_start(total):
            JOB_FRAMES_TOTAL[job_id] = int(total)
            _log(job_id, f"Frames total: {total}")

        def on_progress(done):
            JOB_FRAMES_DONE[job_id] = int(done)

        try:
            frames_iter = _iter_frames_from_video_file(path, on_start=on_start, on_progress=on_progress)
            t0 = time.time()
            result = analyze_video(frames_iter)
            elapsed = max(1e-6, time.time() - t0)
            fps = JOB_FRAMES_DONE.get(job_id, 0) / elapsed
            _log(job_id, f"Processing done in {elapsed:.2f}s @ {fps:.2f} FPS")
            with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            JOBS[job_id] = {"status": "done", "video_id": result.get("video_id")}
        except Exception as exc:
            _log(job_id, f"Error: {exc}")
            JOBS[job_id] = {"status": "error", "error": str(exc)}

    threading.Thread(target=worker, args=(vid, video_path), daemon=True).start()
    return JobSubmitResponse(video_id=vid, status="queued")


@app.get("/status/{video_id}", response_model=JobStatusResponse)
async def status(video_id: str):
    import time
    job = JOBS.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown video_id")
    total = JOB_FRAMES_TOTAL.get(video_id) or 0
    done = JOB_FRAMES_DONE.get(video_id) or 0
    progress = (done / total) if total else None
    elapsed = time.time() - JOB_START_TS.get(video_id, time.time())
    fps = (done / elapsed) if elapsed > 0 else None
    eta = ((total - done) / fps) if fps and total and done is not None else None
    return JobStatusResponse(
        video_id=video_id,
        status=job.get("status", "unknown"),
        detail=job.get("error"),
        progress=progress,
        eta_seconds=eta if eta and eta > 0 else None,
        frames_done=done,
        frames_total=total,
        fps=fps if fps else None,
    )


@app.get("/result/{video_id}")
async def result(video_id: str):
    p = REPORT_ROOT / video_id / "summary.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="summary.json missing")
    import json
    return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))


@app.get("/logs/{video_id}", response_class=PlainTextResponse)
async def logs_endpoint(video_id: str):
    p = REPORT_ROOT / video_id / "logs.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


@app.get("/frame/{video_id}/{frame_index}")
async def frame(video_id: str, frame_index: int):
    import cv2
    p = REPORT_ROOT / video_id / "input.mp4"
    if not p.exists():
        raise HTTPException(status_code=404, detail="input.mp4 missing")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(status_code=404, detail="Frame not found")
    import cv2 as _cv2
    ok, buf = _cv2.imencode('.jpg', frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Encode error")
    return Response(content=buf.tobytes(), media_type='image/jpeg')


