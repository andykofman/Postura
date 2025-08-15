from __future__ import annotations

import math

import numpy as np

from pose.backend import Keypoint
from pose.smoothing import EmaSmoother


def _kp(x: float, y: float, c: float = 1.0) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=c)


def test_ema_initialization_and_update():
    sm = EmaSmoother(num_landmarks=3, alpha=0.5, min_confidence=0.5)
    # First frame: only index 1 is valid
    kps = [None, _kp(0.2, 0.4, 0.9), None]
    smoothed, mask = sm.update(kps)
    assert mask == [False, True, False]
    assert math.isnan(smoothed[0].x) and math.isnan(smoothed[0].y)
    assert abs(smoothed[1].x - 0.2) < 1e-12
    assert abs(smoothed[1].y - 0.4) < 1e-12

    # Second frame: a new observation at index 1; EMA with alpha=0.5
    kps2 = [None, _kp(0.4, 0.6, 0.9), None]
    s2, m2 = sm.update(kps2)
    assert m2 == [False, True, False]
    # EMA: 0.5*0.4 + 0.5*0.2 = 0.3
    assert abs(s2[1].x - 0.3) < 1e-12
    assert abs(s2[1].y - 0.5) < 1e-12


def test_ema_low_confidence_propagates_nan():
    sm = EmaSmoother(num_landmarks=1, alpha=0.8, min_confidence=0.5)
    # Initialize with valid
    s1, m1 = sm.update([_kp(0.5, 0.6, 0.9)])
    assert m1 == [True]
    # Next is low confidence -> invalid measurement, outputs NaN (state kept internally)
    s2, m2 = sm.update([_kp(0.9, 0.9, 0.1)])
    assert m2 == [False]
    assert math.isnan(s2[0].x) and math.isnan(s2[0].y)
    # Next valid should resume from previous state (0.5, 0.6)
    s3, m3 = sm.update([_kp(0.7, 0.6, 0.9)])
    assert m3 == [True]
    # EMA: 0.8*0.7 + 0.2*0.5 = 0.66
    assert abs(s3[0].x - 0.66) < 1e-12


