from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .backend import Keypoint


@dataclass(frozen=True)
class SmoothedKeypoint:
    x: float
    y: float
    confidence: float


class EmaSmoother:
    """
    Per-joint exponential moving average (EMA) smoother for 2D keypoints.

    - Maintains independent EMA state for each landmark index
    - Treats None or low-confidence measurements as missing
    - When a measurement is missing, previous EMA state is kept and NaN is returned for that joint
      to allow downstream metrics to propagate NaNs; a validity mask is also returned

    Returns from update():
      - smoothed_keypoints: List[SmoothedKeypoint] with NaN x/y where invalid
      - valid_mask: List[bool] indicating which joints were updated with a valid measurement
    """

    def __init__(
        self,
        num_landmarks: int,
        *,
        alpha: float = 0.7,
        min_confidence: float = 0.5,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if num_landmarks <= 0:
            raise ValueError("num_landmarks must be positive")
        self.num_landmarks = int(num_landmarks)
        self.alpha = float(alpha)
        self.min_confidence = float(min_confidence)

        self._prev_x = np.full(self.num_landmarks, np.nan, dtype=np.float64)
        self._prev_y = np.full(self.num_landmarks, np.nan, dtype=np.float64)

    def reset(self) -> None:
        self._prev_x.fill(np.nan)
        self._prev_y.fill(np.nan)

    def update(
        self, keypoints: Sequence[Optional[Keypoint]]
    ) -> Tuple[List[SmoothedKeypoint], List[bool]]:
        if len(keypoints) != self.num_landmarks:
            raise ValueError("keypoints length does not match num_landmarks")

        smoothed: List[SmoothedKeypoint] = []
        valid_mask: List[bool] = []

        for idx, kp in enumerate(keypoints):
            if kp is None or not np.isfinite(kp.x) or not np.isfinite(kp.y) or kp.confidence < self.min_confidence:
                # Missing or low-confidence: keep previous state, output NaNs
                smoothed.append(SmoothedKeypoint(x=np.nan, y=np.nan, confidence=0.0))
                valid_mask.append(False)
                continue

            x_obs = float(kp.x)
            y_obs = float(kp.y)

            px = self._prev_x[idx]
            py = self._prev_y[idx]

            if np.isnan(px) or np.isnan(py):
                # First valid observation initializes the EMA at the observation
                x_f = x_obs
                y_f = y_obs
            else:
                a = self.alpha
                x_f = a * x_obs + (1.0 - a) * px
                y_f = a * y_obs + (1.0 - a) * py

            # Store filtered state
            self._prev_x[idx] = x_f
            self._prev_y[idx] = y_f

            smoothed.append(SmoothedKeypoint(x=x_f, y=y_f, confidence=kp.confidence))
            valid_mask.append(True)

        return smoothed, valid_mask

