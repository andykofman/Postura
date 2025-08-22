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
from fastapi.middleware.cors import CORSMiddleware

from analysis.analyzer import analyze_video
from api.schemas import AnalyzeResponse, JobSubmitResponse, JobStatusResponse
from pose.backend import PoseBackend
from pose.backend import Keypoint
from pose.draw import draw_keypoints


# Threading/env tuning - optimized for Render's 8-core environment
# Use environment variables if set, otherwise auto-detect optimal values
import multiprocessing
optimal_threads = min(multiprocessing.cpu_count(), 6)  # Leave 2 cores for system

os.environ.setdefault("OMP_NUM_THREADS", str(optimal_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(optimal_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(optimal_threads))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(optimal_threads))


app = FastAPI(title="Postura API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_CONTENT_TYPES = {"video/mp4", "application/octet-stream"}

# Mount static web UI and reports directory for end-to-end testing
WEB_ROOT = Path("web").resolve()
REPORT_ROOT = Path("report").resolve()
DEMO_ROOT = Path("demo").resolve()
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
if WEB_ROOT.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_ROOT), html=True), name="ui")
app.mount("/reports", StaticFiles(directory=str(REPORT_ROOT)), name="reports")
if DEMO_ROOT.exists():
    app.mount("/demo", StaticFiles(directory=str(DEMO_ROOT)), name="demo")
def _get_env_int(name: str, default: int) -> int:
    try:
        v = int(str(os.getenv(name, str(default))).strip())
        return v
    except Exception:
        return default

def _get_env_float(name: str, default: float) -> float:
    try:
        v = float(str(os.getenv(name, str(default))).strip())
        return v
    except Exception:
        return default

# Performance knobs (tunable via environment) — defaults preserve original accuracy
# These can be overridden via environment variables for Render optimization
POSTURA_TARGET_FPS = _get_env_float("POSTURA_TARGET_FPS", 0.0)  # <= 0 disables decimation
POSTURA_TARGET_WIDTH = _get_env_int("POSTURA_TARGET_WIDTH", 0)  # <= 0 disables resize
POSTURA_MODEL_COMPLEXITY = _get_env_int("POSTURA_MODEL_COMPLEXITY", 2)  # 0/1/2

# Render-specific optimizations - can be tuned via environment
POSTURA_RENDER_OPTIMIZED = _get_env_int("POSTURA_RENDER_OPTIMIZED", 1)  # Enable Render optimizations

# Memory management optimizations
POSTURA_MEMORY_OPTIMIZED = _get_env_int("POSTURA_MEMORY_OPTIMIZED", 1)  # Enable memory optimizations
POSTURA_GC_INTERVAL = _get_env_int("POSTURA_GC_INTERVAL", 50)  # Garbage collection interval (frames) - more aggressive
POSTURA_MEMORY_THRESHOLD = _get_env_int("POSTURA_MEMORY_THRESHOLD", 70)  # Memory usage threshold for forced GC (%)

# Processing mode optimization
POSTURA_STREAMING_MODE = _get_env_int("POSTURA_STREAMING_MODE", 1)  # Enable streaming frame processing (no queuing)



def _configure_opencv_threads() -> None:
    try:
        import cv2  # type: ignore
        cv2.setUseOptimized(True)
        try:
            # Use optimal thread count for Render's 8-core environment
            optimal_cv_threads = min(multiprocessing.cpu_count(), 6)
            cv2.setNumThreads(optimal_cv_threads)
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
    import time
    
    _configure_opencv_threads()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open uploaded video")

    # Back to original simple approach with detailed timing
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = 1
    if target_fps and target_fps > 0 and orig_fps > 0:
        stride = max(1, int(round(orig_fps / float(target_fps))))

    # Build backend
    backend: Optional[PoseBackend]
    try:
        backend = PoseBackend(model_complexity=model_complexity)
    except ImportError as exc:
        raise RuntimeError("mediapipe is not available in the API runtime") from exc

    # Timing diagnostics
    timing_stats = {
        'read_times': [],
        'resize_times': [],
        'infer_times': [],
        'total_frames': 0,
        'processed_frames': 0
    }

    try:
        with backend as b:
            idx = 0
            if callable(on_start):
                try:
                    on_start(total_frames)
                except Exception:
                    pass
                    
            while True:
                # Time frame reading
                read_start = time.time()
                ok, frame = cap.read()
                read_time = time.time() - read_start
                
                if not ok:
                    break
                
                timing_stats['total_frames'] += 1
                timing_stats['read_times'].append(read_time)
                
                send = not (stride > 1 and (idx % stride) != 0)
                idx += 1
                if not send:
                    continue
                
                timing_stats['processed_frames'] += 1
                
                # Time resize if needed
                resize_start = time.time()
                if target_width and target_width > 0:
                    h, w = frame.shape[:2]
                    if w > target_width:
                        scale = float(target_width) / float(w)
                        new_w = int(round(w * scale))
                        new_h = int(round(h * scale))
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                resize_time = time.time() - resize_start
                timing_stats['resize_times'].append(resize_time)
                
                # Time inference
                infer_start = time.time()
                kps = b.infer(frame)
                infer_time = time.time() - infer_start
                timing_stats['infer_times'].append(infer_time)
                
                # Log timing every 50 frames
                if timing_stats['processed_frames'] % 50 == 0:
                    avg_read = sum(timing_stats['read_times'][-50:]) / min(50, len(timing_stats['read_times']))
                    avg_resize = sum(timing_stats['resize_times'][-50:]) / min(50, len(timing_stats['resize_times']))
                    avg_infer = sum(timing_stats['infer_times'][-50:]) / min(50, len(timing_stats['infer_times']))
                    print(f"Frame {timing_stats['processed_frames']}: read={avg_read*1000:.1f}ms, resize={avg_resize*1000:.1f}ms, infer={avg_infer*1000:.1f}ms")
                
                if callable(on_progress):
                    try:
                        on_progress(idx)
                    except Exception:
                        pass
                
                yield kps
    finally:
        cap.release()
        
        # Print final timing stats
        if timing_stats['processed_frames'] > 0:
            avg_read = sum(timing_stats['read_times']) / len(timing_stats['read_times'])
            avg_resize = sum(timing_stats['resize_times']) / len(timing_stats['resize_times'])
            avg_infer = sum(timing_stats['infer_times']) / len(timing_stats['infer_times'])
            print(f"FINAL TIMING STATS:")
            print(f"  Average read time: {avg_read*1000:.1f}ms")
            print(f"  Average resize time: {avg_resize*1000:.1f}ms") 
            print(f"  Average inference time: {avg_infer*1000:.1f}ms")
            print(f"  Total processed frames: {timing_stats['processed_frames']}")
            print(f"  Theoretical max FPS: {1.0/(avg_read+avg_resize+avg_infer):.1f}")


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
            frames_iter = _iter_frames_from_video_file(
                path,
                target_fps=POSTURA_TARGET_FPS if POSTURA_TARGET_FPS > 0 else None,
                target_width=POSTURA_TARGET_WIDTH if POSTURA_TARGET_WIDTH > 0 else None,
                model_complexity=POSTURA_MODEL_COMPLEXITY,
                on_start=on_start,
                on_progress=on_progress,
            )
            t0 = time.time()
            result = analyze_video(frames_iter)
            elapsed = max(1e-6, time.time() - t0)
            fps = JOB_FRAMES_DONE.get(job_id, 0) / elapsed
            _log(job_id, f"Processing done in {elapsed:.2f}s @ {fps:.2f} FPS")
            # Force result video_id to job_id to keep API/UI paths stable
            result["video_id"] = job_id
            # Persist under the job_id directory
            result_dir = REPORT_ROOT / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            # input.mp4 already exists at path
            with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            # Generate annotated thumbnails for reps
            try:
                _generate_thumbnails(video_path=result_dir / "input.mp4", result=result, out_dir=result_dir / "thumbs", report_id=job_id)
                _log(job_id, "Thumbnails generated")
            except Exception as thumb_exc:
                _log(job_id, f"Thumbnail generation failed: {thumb_exc}")
            JOBS[job_id] = {"status": "done", "video_id": job_id}
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
    data = json.loads(p.read_text(encoding="utf-8"))
    # Merge thumbnails info if present
    tjson = REPORT_ROOT / video_id / "thumbnails.json"
    if tjson.exists():
        try:
            thumbs = json.loads(tjson.read_text(encoding="utf-8"))
            if isinstance(thumbs, list):
                data["thumbnails"] = thumbs
        except Exception:
            pass
    return JSONResponse(content=data)


@app.get("/logs/{video_id}", response_class=PlainTextResponse)
async def logs_endpoint(video_id: str):
    p = REPORT_ROOT / video_id / "logs.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.get("/system-info", response_class=PlainTextResponse) 
async def system_info() -> str:
    """Debug endpoint to check system resources on deployment platform"""
    import psutil
    import platform
    
    info = []
    info.append(f"Platform: {platform.platform()}")
    info.append(f"CPU Count: {psutil.cpu_count()}")
    info.append(f"CPU Count (logical): {psutil.cpu_count(logical=True)}")
    info.append(f"Memory Total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    info.append(f"Memory Available: {psutil.virtual_memory().available / (1024**3):.2f} GB") 
    info.append(f"Memory Used: {psutil.virtual_memory().used / (1024**3):.2f} GB")
    info.append(f"Memory Percent: {psutil.virtual_memory().percent:.1f}%")
    info.append(f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    
    # Thread settings
    info.append(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS', 'unset')}")
    
    try:
        import cv2
        info.append(f"OpenCV Threads: {cv2.getNumThreads()}")
    except ImportError:
        info.append("OpenCV Threads: N/A (cv2 not available)")
    
    # Memory optimization status
    info.append(f"POSTURA_RENDER_OPTIMIZED: {POSTURA_RENDER_OPTIMIZED}")
    info.append(f"POSTURA_MEMORY_OPTIMIZED: {POSTURA_MEMORY_OPTIMIZED}")
    info.append(f"POSTURA_GC_INTERVAL: {POSTURA_GC_INTERVAL}")
    info.append(f"POSTURA_MEMORY_THRESHOLD: {POSTURA_MEMORY_THRESHOLD}%")
    info.append(f"Processing Mode: Diagnostic (with detailed timing)")
    info.append(f"Debug Endpoints: /debug-comparison for environment analysis")
    
    return "\n".join(info)


@app.get("/memory-status", response_class=PlainTextResponse)
async def memory_status() -> str:
    """Real-time memory monitoring for debugging performance issues"""
    import psutil
    import gc
    
    info = []
    
    # Current memory status
    memory = psutil.virtual_memory()
    info.append(f"Memory Total: {memory.total / (1024**3):.2f} GB")
    info.append(f"Memory Available: {memory.available / (1024**3):.2f} GB")
    info.append(f"Memory Used: {memory.used / (1024**3):.2f} GB")
    info.append(f"Memory Percent: {memory.percent:.1f}%")
    
    # Memory pressure indicators
    if memory.percent > 80:
        info.append(" HIGH MEMORY PRESSURE - Performance may degrade")
    elif memory.percent > 60:
        info.append(" MODERATE MEMORY PRESSURE - Monitor closely")
    else:
        info.append("Memory usage is healthy")
    
    # Garbage collection stats
    gc_stats = gc.get_stats()
    info.append(f"GC Collections: {gc_stats[0]['collections'] if gc_stats else 'N/A'}")
    
    # System recommendations
    if memory.percent > 70:
        info.append("Consider setting POSTURA_GC_INTERVAL=25 for more aggressive cleanup")
        info.append(" Consider setting POSTURA_MEMORY_THRESHOLD=60 for earlier intervention")
    
    return "\n".join(info)


@app.get("/debug-comparison", response_class=PlainTextResponse)
async def debug_comparison() -> str:
    """Compare Render environment to typical local Docker environment"""
    import psutil
    import platform
    import subprocess
    import os
    
    info = []
    info.append("=== RENDER ENVIRONMENT DEBUG ===")
    info.append(f"Platform: {platform.platform()}")
    info.append(f"Python: {platform.python_version()}")
    info.append(f"CPU Count: {psutil.cpu_count()}")
    info.append(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    # Check OpenCV build info
    try:
        import cv2
        info.append(f"OpenCV Version: {cv2.__version__}")
        build_info = cv2.getBuildInformation()
        # Extract key performance-related info
        for line in build_info.split('\n'):
            if any(keyword in line.lower() for keyword in ['parallel', 'thread', 'cpu', 'simd', 'optimization']):
                info.append(f"  {line.strip()}")
    except Exception as e:
        info.append(f"OpenCV info error: {e}")
    
    # Check MediaPipe info
    try:
        import mediapipe as mp
        info.append(f"MediaPipe Version: {mp.__version__}")
    except Exception as e:
        info.append(f"MediaPipe info error: {e}")
    
    # Environment variables affecting performance
    perf_vars = [
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 
        'NUMEXPR_NUM_THREADS', 'OPENCV_NUM_THREADS'
    ]
    info.append("\n=== PERFORMANCE ENVIRONMENT VARIABLES ===")
    for var in perf_vars:
        info.append(f"{var}: {os.getenv(var, 'unset')}")
    
    # Check if we're in a container
    try:
        with open('/proc/1/cgroup', 'r') as f:
            cgroup = f.read()
            if 'docker' in cgroup or 'container' in cgroup:
                info.append("\n✅ Running in container")
            else:
                info.append("\n⚠️  Not detected as container")
    except:
        info.append("\n❓ Container detection failed")
    
    # CPU features
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            flags_line = [line for line in cpuinfo.split('\n') if line.startswith('flags')]
            if flags_line:
                flags = flags_line[0].split(':')[1].strip()
                important_flags = [flag for flag in flags.split() if flag in ['avx', 'avx2', 'sse4_1', 'sse4_2', 'fma']]
                info.append(f"\nCPU Performance Flags: {', '.join(important_flags)}")
    except:
        info.append("\n❓ CPU info not available")
    
    info.append("\n=== EXPECTED VS ACTUAL PERFORMANCE ===")
    info.append("Expected on local Docker: 6 FPS")
    info.append("Expected on Render (same specs): 5-6 FPS")
    info.append("Current on Render: 1.2-1.5 FPS")
    info.append("Performance Gap: 4-5x slower than expected")
    
    info.append("\n💡 This debug info will help identify the bottleneck!")
    info.append("💡 Check the timing logs after processing a video!")
    
    return "\n".join(info)


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


def _generate_thumbnails(*, video_path: Path, result: dict, out_dir: Path, report_id: str) -> None:
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_data = result.get("frame_data") or []
    # generate for first N reps per exercise to limit time
    max_reps = 12
    thumbs = []
    with PoseBackend() as backend:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        for item in frame_data[:max_reps]:
            idx = int(item.get("frame_index", -1))
            if idx < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            try:
                kps = backend.infer(frame)
                frame = draw_keypoints(frame, kps)
            except Exception:
                pass
            name = f"rep_{item.get('exercise','rep')}_{item.get('rep_id',0)}_f{idx}.jpg"
            path = out_dir / name
            import cv2 as _cv2
            ok, buf = _cv2.imencode('.jpg', frame)
            if ok:
                path.write_bytes(buf.tobytes())
                thumbs.append({
                    "frame_index": idx,
                    "exercise": item.get("exercise"),
                    "rep_id": item.get("rep_id"),
                    "angles": item.get("angles"),
                    "path": f"/reports/{report_id}/thumbs/{name}"
                })
        cap.release()
    # write manifest
    import json
    (video_path.parent / "thumbnails.json").write_text(json.dumps(thumbs, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/report/{video_id}.pdf")
async def pdf_report(video_id: str):
    """Generate a PDF report with summary, charts, and a few annotated frames."""
    import io, json, math
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summ_path = REPORT_ROOT / video_id / "summary.json"
    if not summ_path.exists():
        raise HTTPException(status_code=404, detail="summary.json missing")
    data = json.loads(summ_path.read_text(encoding="utf-8"))

    # Prepare PDF buffer
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, height-2*cm, f"Postura Report – {video_id}")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, height-2.7*cm, "Automated form analysis and rep summary")

    # Summary
    s = data.get("summary", {})
    sq = s.get("squats", {})
    pu = s.get("pushups", {})
    y = height-4*cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, "Summary")
    y -= 0.6*cm
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Squats – total: {sq.get('total_reps',0)}, good-form: {sq.get('good_form_reps',0)}")
    y -= 0.5*cm
    c.drawString(2*cm, y, f"Pushups – total: {pu.get('total_reps',0)}, good-form: {pu.get('good_form_reps',0)}")

    # Bar chart good vs bad
    fig, ax = plt.subplots(figsize=(4,2))
    labels = ['Squats','Pushups']
    good = [sq.get('good_form_reps',0), pu.get('good_form_reps',0)]
    total = [sq.get('total_reps',0), pu.get('total_reps',0)]
    bad = [max(0, total[0]-good[0]), max(0, total[1]-good[1])]
    x = [0,1]
    ax.bar(x, good, color="#22c55e", label="Good")
    ax.bar(x, bad, bottom=good, color="#ef4444", label="Bad")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(1, max(total)))
    ax.legend(loc='upper right')
    ax.set_title('Good vs Bad reps')
    chart_buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(chart_buf, format='png', dpi=150)
    plt.close(fig)
    chart_buf.seek(0)
    c.drawImage(ImageReader(chart_buf), 2*cm, y-6*cm, width=8*cm, height=5*cm)

    # Angle vs frame plot (use knee if available else elbow)
    frames = data.get('frame_data', [])
    if frames:
        xs = [f.get('frame_index',0) for f in frames]
        ys = [(f.get('angles') or {}).get('knee') or (f.get('angles') or {}).get('elbow') or 0 for f in frames]
        fig2, ax2 = plt.subplots(figsize=(4,2))
        ax2.plot(xs, ys, color="#A855F7")
        ax2.set_title('Angle vs Frame')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Angle (deg)')
        b2 = io.BytesIO()
        plt.tight_layout()
        fig2.savefig(b2, format='png', dpi=150)
        plt.close(fig2)
        b2.seek(0)
        c.drawImage(ImageReader(b2), 12*cm, y-6*cm, width=8*cm, height=5*cm)

    c.showPage()

    # Thumbnails page
    tjson = REPORT_ROOT / video_id / "thumbnails.json"
    thumbs = []
    if tjson.exists():
        try:
            thumbs = json.loads(tjson.read_text(encoding="utf-8"))
        except Exception:
            thumbs = []
    if thumbs:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, height-2*cm, "Annotated Snapshots")
        x, y = 2*cm, height-3.5*cm
        w, h = 7*cm, 4*cm
        per_row = 2
        count = 0
        for t in thumbs[:6]:
            img_path = REPORT_ROOT / video_id / "thumbs" / Path(t.get('path','')).name
            if img_path.exists():
                c.drawImage(str(img_path), x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
                c.setFont("Helvetica", 10)
                c.drawString(x, y-0.3*cm, f"Rep {t.get('rep_id')} • {t.get('exercise')} • angle {(t.get('angles') or {}).get('knee') or (t.get('angles') or {}).get('elbow')}")
                count += 1
                if count % per_row == 0:
                    x = 2*cm; y -= h + 1.2*cm
                else:
                    x += w + 1.0*cm
        c.showPage()

    c.save()
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type='application/pdf')


