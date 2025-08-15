from __future__ import annotations

import numpy as np

from pose.backend import PoseBackend, Keypoint


class _Lm:
    def __init__(self, x: float, y: float, visibility: float) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


class _Landmarks:
    def __init__(self, count: int) -> None:
        self.landmark = [_Lm(x=i / count, y=i / count, visibility=1.0) for i in range(count)]


class _Result:
    def __init__(self, pose_landmarks) -> None:
        self.pose_landmarks = pose_landmarks


class _FakePoseAll:
    def process(self, frame_rgb):
        assert isinstance(frame_rgb, np.ndarray)
        return _Result(_Landmarks(33))


class _FakePoseNone:
    def process(self, frame_rgb):
        return _Result(None)


class _FakePosePartial:
    def __init__(self, n: int) -> None:
        self.n = n

    def process(self, frame_rgb):
        return _Result(_Landmarks(self.n))


def _dummy_frame(h: int = 64, w: int = 64) -> np.ndarray:
    # BGR dummy frame
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_infer_with_valid_landmarks():
    backend = PoseBackend(pose_model=_FakePoseAll())
    kps = backend.infer(_dummy_frame())
    assert len(kps) == 33
    assert all((kp is None or isinstance(kp, Keypoint)) for kp in kps)
    # Check normalization bounds
    for kp in kps:
        assert kp is not None
        assert 0.0 <= kp.x <= 1.0
        assert 0.0 <= kp.y <= 1.0
        assert 0.0 <= kp.confidence <= 1.0


def test_infer_no_pose_returns_nones():
    backend = PoseBackend(pose_model=_FakePoseNone())
    kps = backend.infer(_dummy_frame())
    assert len(kps) == 33
    assert all(kp is None for kp in kps)


def test_infer_partial_fills_with_none():
    backend = PoseBackend(pose_model=_FakePosePartial(10))
    kps = backend.infer(_dummy_frame())
    assert len(kps) == 33
    assert sum(1 for kp in kps if kp is None) == 23
    assert sum(1 for kp in kps if kp is not None) == 10


