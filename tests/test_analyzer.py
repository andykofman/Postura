from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from analysis.analyzer import analyze_video
from pose.backend import Keypoint


L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def _blank_frame() -> List[Optional[Keypoint]]:
    return [None] * 33


def _kp(x: float, y: float, c: float = 1.0) -> Keypoint:
    return Keypoint(x=float(x), y=float(y), confidence=float(c))


def test_analyze_video_schema_and_counts():
    frames: List[List[Optional[Keypoint]]] = []

    # Frame 0: squat bottom (knee ~90deg), alignment OK (hip-shoulder-ankle collinear on x-axis)
    f0 = _blank_frame()
    f0[L_HIP] = _kp(0.2, 0.5)
    f0[L_SHOULDER] = _kp(0.4, 0.5)
    f0[L_KNEE] = _kp(0.6, 0.5)  # knee at same y
    f0[L_ANKLE] = _kp(0.8, 0.5)
    # Configure knee angle at ~90 by placing hip and ankle around knee
    f0[L_HIP] = _kp(0.6, 0.3)
    f0[L_ANKLE] = _kp(0.8, 0.5)
    frames.append(f0)

    # Frame 1: squat top (~180deg) -> counts 1 rep
    f1 = _blank_frame()
    f1[L_HIP] = _kp(0.4, 0.5)
    f1[L_KNEE] = _kp(0.6, 0.5)
    f1[L_ANKLE] = _kp(0.8, 0.5)
    f1[L_SHOULDER] = _kp(0.4, 0.5)
    frames.append(f1)

    # Frame 2: pushup bottom (elbow ~90deg), alignment OK using same hip/shoulder/ankle line
    f2 = _blank_frame()
    f2[L_SHOULDER] = _kp(0.5, 0.5)
    f2[L_ELBOW] = _kp(0.5, 0.5)  # pivot
    f2[L_WRIST] = _kp(0.7, 0.5)
    # To get ~90deg at elbow: place shoulder above and wrist to the right
    f2[L_SHOULDER] = _kp(0.5, 0.3)
    # Alignment line and ankle
    f2[L_HIP] = _kp(0.4, 0.5)
    f2[L_ANKLE] = _kp(0.8, 0.5)
    frames.append(f2)

    # Frame 3: pushup top (~180deg) -> counts 1 rep
    f3 = _blank_frame()
    f3[L_SHOULDER] = _kp(0.3, 0.5)
    f3[L_ELBOW] = _kp(0.5, 0.5)
    f3[L_WRIST] = _kp(0.7, 0.5)
    f3[L_HIP] = _kp(0.4, 0.5)
    f3[L_ANKLE] = _kp(0.8, 0.5)
    frames.append(f3)

    out = analyze_video(frames)
    assert isinstance(out, dict)
    assert "video_id" in out and isinstance(out["video_id"], str)
    assert "summary" in out and "frame_data" in out

    summary = out["summary"]
    assert summary["squats"]["total_reps"] >= 1
    assert summary["pushups"]["total_reps"] >= 1

    # Validate frame_data schema
    fd = out["frame_data"]
    assert isinstance(fd, list)
    assert any(item.get("exercise") == "squat" for item in fd)
    assert any(item.get("exercise") == "pushup" for item in fd)
    # Angles must be present and numeric when emitted
    for item in fd:
        assert "frame_index" in item and "rep_id" in item and "is_form_ok" in item
        assert "angles" in item and isinstance(item["angles"], dict)

