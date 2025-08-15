from __future__ import annotations

from math import acos, atan2, isfinite
from typing import Optional, Sequence, Tuple

import numpy as np

from pose.backend import Keypoint


def _vec(a: Keypoint, b: Keypoint) -> Tuple[float, float]:
    return (b.x - a.x, b.y - a.y)


def _dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def _norm(ax: float, ay: float) -> float:
    return float(np.hypot(ax, ay))


def angle(a: Optional[Keypoint], b: Optional[Keypoint], c: Optional[Keypoint]) -> float:
    """
    Returns the angle at point B (in radians) for triangle (A,B,C).

    - If any point is None or has non-finite coordinates, returns np.nan
    - If any vector is near-zero, returns np.nan
    """
    if a is None or b is None or c is None:
        return float("nan")
    if not all(isfinite(v) for v in (a.x, a.y, b.x, b.y, c.x, c.y)):
        return float("nan")

    abx, aby = _vec(b, a)
    cbx, cby = _vec(b, c)
    n1 = _norm(abx, aby)
    n2 = _norm(cbx, cby)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return float("nan")

    cos_theta = _dot(abx, aby, cbx, cby) / (n1 * n2)
    # Clamp due to numerical errors
    cos_theta = max(-1.0, min(1.0, float(cos_theta)))
    return float(acos(cos_theta))


def horiz_offset(knee: Optional[Keypoint], ankle: Optional[Keypoint]) -> float:
    """
    Horizontal offset between knee and ankle (signed, in normalized units): ankle.x - knee.x
    Returns np.nan if inputs invalid.
    """
    if knee is None or ankle is None:
        return float("nan")
    if not all(isfinite(v) for v in (knee.x, ankle.x)):
        return float("nan")
    return float(ankle.x - knee.x)


def collinearity_residual(
    hip: Optional[Keypoint], shoulder: Optional[Keypoint], ankle: Optional[Keypoint]
) -> float:
    """
    Signed area-based residual indicating how far (ankle) deviates from the line through (hip→shoulder).

    - Compute perpendicular distance of ankle to the line (hip->shoulder), normalized by the segment length.
    - Positive/negative sign corresponds to side of the line via cross product sign.
    - Returns np.nan if inputs invalid or hip==shoulder.
    """
    if hip is None or shoulder is None or ankle is None:
        return float("nan")
    if not all(isfinite(v) for v in (hip.x, hip.y, shoulder.x, shoulder.y, ankle.x, ankle.y)):
        return float("nan")

    vx = shoulder.x - hip.x
    vy = shoulder.y - hip.y
    wx = ankle.x - hip.x
    wy = ankle.y - hip.y
    seg_len = float(np.hypot(vx, vy))
    if seg_len <= 1e-12:
        return float("nan")

    # 2D cross product magnitude gives area scale; sign for side
    cross = vx * wy - vy * wx
    # Normalize by segment length to obtain signed perpendicular distance
    return float(cross / seg_len)


