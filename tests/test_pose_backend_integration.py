from __future__ import annotations

import time
from pathlib import Path

import pytest


mp = pytest.importorskip("mediapipe", reason="mediapipe not installed; integration test skipped")
cv2 = pytest.importorskip("cv2", reason="opencv-python not installed; integration test skipped")


@pytest.mark.integration
@pytest.mark.slow
def test_pose_backend_on_demo_video():
    from pose.backend import PoseBackend, Keypoint

    # Prefer tests/input.mp4 if present, else fallback to demo/input.mp4
    root = Path(__file__).resolve().parents[1]
    candidate_tests = root / "tests" / "input.mp4"
    candidate_demo = root / "demo" / "input.mp4"
    video_path = candidate_tests if candidate_tests.exists() else candidate_demo
    if not video_path.exists():
        pytest.skip(f"no input.mp4 found at {candidate_tests} or {candidate_demo}")

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Failed to open demo video"

    processed = 0
    frames_with_pose = 0
    first_keypoints = None

    t0 = time.time()
    with PoseBackend() as backend:
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            kps = backend.infer(frame)
            processed += 1
            if any(kp is not None for kp in kps):
                frames_with_pose += 1
                if first_keypoints is None:
                    first_keypoints = kps
    t1 = time.time()
    cap.release()

    assert processed > 0, "No frames processed from demo video"
    assert frames_with_pose >= 1, "No pose detected in the first few frames"

    # Validate normalization on a subset of keypoints
    if first_keypoints is not None:
        for kp in first_keypoints[:5]:
            if kp is None:
                continue
            assert 0.0 <= kp.x <= 1.0
            assert 0.0 <= kp.y <= 1.0
            assert 0.0 <= kp.confidence <= 1.0

    # Basic performance sanity (should be positive FPS)
    elapsed = max(1e-6, t1 - t0)
    fps = processed / elapsed
    assert fps > 0


