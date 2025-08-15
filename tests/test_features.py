from __future__ import annotations

import math

import numpy as np

from pose.backend import Keypoint
from analysis.features import angle, horiz_offset, collinearity_residual


def _kp(x: float, y: float, c: float = 1.0) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=c)


def test_angle_basic_right_angle():
    # Right angle at B: A(0,0), B(0,1), C(1,1)
    A = _kp(0.0, 0.0)
    B = _kp(0.0, 1.0)
    C = _kp(1.0, 1.0)
    th = angle(A, B, C)
    assert math.isfinite(th)
    assert abs(th - (math.pi / 2)) < 1e-6


def test_angle_nan_on_missing():
    A = _kp(0.0, 0.0)
    B = None
    C = _kp(1.0, 1.0)
    th = angle(A, B, C)
    assert math.isnan(th)


def test_horiz_offset_sign():
    k = _kp(0.4, 0.5)
    a = _kp(0.7, 0.9)
    v = horiz_offset(k, a)
    assert math.isfinite(v)
    assert abs(v - 0.3) < 1e-9


def test_horiz_offset_nan_on_invalid():
    k = None
    a = _kp(0.7, 0.9)
    assert math.isnan(horiz_offset(k, a))


def test_collinearity_residual_zero_when_on_line():
    # hip(0,0), shoulder(1,0) => x-axis; ankle on x-axis => zero residual
    hip = _kp(0.0, 0.0)
    sh = _kp(1.0, 0.0)
    an = _kp(0.5, 0.0)
    r = collinearity_residual(hip, sh, an)
    assert abs(r) < 1e-12


def test_collinearity_residual_signed_distance():
    # Line along x-axis; ankle above the line => positive (since cross = vx*wy - vy*wx, vy=0)
    hip = _kp(0.0, 0.0)
    sh = _kp(2.0, 0.0)
    an = _kp(1.0, 0.5)
    r = collinearity_residual(hip, sh, an)
    # seg_len = 2, cross = 2*0.5 - 0 = 1.0 -> residual = 0.5
    assert abs(r - 0.5) < 1e-12


