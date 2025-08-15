from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .backend import Keypoint


# Subset of BlazePose connections (indices for 33-landmark model)
# Focus on stable torso/limbs; avoids relying on MediaPipe import at runtime.
CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27), (27, 29), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32),
)


def _project_to_px(width: int, height: int, kp: Keypoint) -> Tuple[int, int]:
    x = int(round(kp.x * (width - 1)))
    y = int(round(kp.y * (height - 1)))
    x = 0 if np.isnan(x) else max(0, min(width - 1, x))
    y = 0 if np.isnan(y) else max(0, min(height - 1, y))
    return x, y


def draw_keypoints(
    frame_bgr: np.ndarray,
    keypoints: Sequence[Optional[Keypoint]],
    *,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    line_color: Tuple[int, int, int] = (0, 200, 255),
    confidence_threshold: float = 0.3,
    draw_connections: Iterable[Tuple[int, int]] = CONNECTIONS,
) -> np.ndarray:
    """Draw keypoints and skeletal connections on the frame (in place) and return it.

    Does not require OpenCV at import; uses it lazily to avoid hard dependency during tests.
    """
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
        return frame_bgr

    try:
        import cv2  # type: ignore
    except Exception:
        # If cv2 is unavailable, return the original frame
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    radius = max(2, int(round(0.004 * max(width, height))))
    thickness = max(1, int(round(0.003 * max(width, height))))

    # Draw connections first so points render on top
    for a, b in draw_connections:
        if a < len(keypoints) and b < len(keypoints):
            kpa = keypoints[a]
            kpb = keypoints[b]
            if (
                kpa is not None
                and kpb is not None
                and kpa.confidence >= confidence_threshold
                and kpb.confidence >= confidence_threshold
            ):
                pa = _project_to_px(width, height, kpa)
                pb = _project_to_px(width, height, kpb)
                cv2.line(frame_bgr, pa, pb, line_color, thickness)

    # Draw keypoints
    for kp in keypoints:
        if kp is not None and kp.confidence >= confidence_threshold:
            cx, cy = _project_to_px(width, height, kp)
            cv2.circle(frame_bgr, (cx, cy), radius, point_color, -1)

    return frame_bgr


def render_video_with_traces(
    input_path: str,
    output_path: str,
    *,
    backend_factory,
    limit_frames: Optional[int] = None,
) -> str:
    """Render an annotated video with pose traces.

    - backend_factory: a callable that returns a context-managed backend (e.g., PoseBackend)
    - limit_frames: if provided, stops after this many frames (useful for samples/tests)
    Returns the output path on success.
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer: {output_path}")

    processed = 0
    with backend_factory() as backend:
        while True:
            if limit_frames is not None and processed >= limit_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            kps = backend.infer(frame)
            draw_keypoints(frame, kps)
            writer.write(frame)
            processed += 1

    writer.release()
    cap.release()
    return output_path



