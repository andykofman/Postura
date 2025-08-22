from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


class PoseBackend:
    """
    Single-person pose backend using MediaPipe BlazePose (Full/Heavy).

    - Keeps the model warm-loaded after construction
    - Accepts BGR frames (as from OpenCV)
    - Returns normalized keypoints in [0, 1]
    - Returns a fixed-length list (33) with None if no pose is detected
    """

    NUM_LANDMARKS = 33

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pose_model: Optional[object] = None,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True,
    ) -> None:
        """
        If pose_model is provided, it must expose a .process(np.ndarray[R,G,B]) -> result
        where result.pose_landmarks is either None or an object with .landmark list
        of 33 items, each having attributes .x, .y and .visibility in [0, 1].
        """
        self._external_model = pose_model is not None
        if pose_model is not None:
            self._pose = pose_model
        else:
            try:
                import mediapipe as mp  # type: ignore
            except Exception as exc:  # pragma: no cover - exercised only when mediapipe missing
                raise ImportError(
                    "mediapipe is required for PoseBackend. Install with `pip install mediapipe`"
                ) from exc

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                model_complexity=model_complexity,
                enable_segmentation=enable_segmentation,
                smooth_landmarks=smooth_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def close(self) -> None:
        """Release underlying resources."""
        try:
            if hasattr(self, "_pose") and not self._external_model:
                close_fn = getattr(self._pose, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:
            pass

    def __enter__(self) -> "PoseBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup
        self.close()

    def infer(self, frame_bgr: np.ndarray) -> List[Optional[Keypoint]]:
        """Run single-person pose detection on a BGR image frame.

        Returns a list of 33 items (MediaPipe BlazePose full): Keypoint or None.
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr must be a numpy array")

        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim < 2:
            raise ValueError("frame_bgr must be an HxWxC numpy array")

        # Convert BGR (OpenCV) -> RGB without requiring cv2
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] >= 3:
            frame_rgb = frame_bgr[..., ::-1]
        else:
            # Grayscale or unexpected channels, treat as RGB already
            frame_rgb = frame_bgr

        result = self._pose.process(frame_rgb)

        if result is None or getattr(result, "pose_landmarks", None) is None:
            return [None] * self.NUM_LANDMARKS

        landmarks = getattr(result.pose_landmarks, "landmark", None)
        if landmarks is None or len(landmarks) == 0:
            return [None] * self.NUM_LANDMARKS

        keypoints: List[Optional[Keypoint]] = []
        # MediaPipe provides normalized coordinates already
        for idx in range(self.NUM_LANDMARKS):
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = float(getattr(lm, "x", 0.0))
                y = float(getattr(lm, "y", 0.0))
                conf = float(getattr(lm, "visibility", 0.0))
                # Clamp to [0,1]
                x = 0.0 if np.isnan(x) else max(0.0, min(1.0, x))
                y = 0.0 if np.isnan(y) else max(0.0, min(1.0, y))
                conf = 0.0 if np.isnan(conf) else max(0.0, min(1.0, conf))
                keypoints.append(Keypoint(x=x, y=y, confidence=conf))
            else:
                keypoints.append(None)

        return keypoints


if __name__ == "__main__":
    import time
    from pathlib import Path
    import sys
    from .draw import render_video_with_traces

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "OpenCV (cv2) is required for the demo. Install with `pip install opencv-python`"
        ) from exc

    video_path = Path(__file__).resolve().parents[1] / "tests" / "input.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        raise SystemExit(1)

    output_path = str(Path(__file__).resolve().parents[1] / "demo" / "output_annotated.mp4")
    t0 = time.time()
    render_video_with_traces(
        str(video_path),
        output_path,
        backend_factory=PoseBackend,
    )
    t1 = time.time()
    elapsed = max(1e-6, t1 - t0)
    print(f"Wrote annotated video to: {output_path}")
    print(f"Elapsed: {elapsed:.2f}s")
    sys.stdout.flush()


