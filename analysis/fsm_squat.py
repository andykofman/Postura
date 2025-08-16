from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .features import angle
from .utils import HysteresisThresholds, SQUAT_KNEE_ANGLE
from pose.backend import Keypoint


@dataclass
class SquatState:
    in_bottom: bool = False
    reps: int = 0


class SquatFSM:
    """
    Simple squat FSM using knee angle with hysteresis.

    - When knee angle <= thresholds.go_down: enter bottom (in_bottom=True)
    - When knee angle >= thresholds.go_up: exit bottom, count a rep (in_bottom False)
    """

    def __init__(self, thresholds: HysteresisThresholds = SQUAT_KNEE_ANGLE) -> None:
        self.thresholds = thresholds
        self.state = SquatState()
        self._frame_idx = -1

    def reset(self) -> None:
        self.state = SquatState()
        self._frame_idx = -1

    def process_frame(
        self,
        features: Dict[str, float],
        *,
        frame_idx: Optional[int] = None,
        joints: Optional[Dict[str, Optional[Keypoint]]] = None,
    ) -> Dict[str, object]:
        """
        features should contain a 'knee_angle' in radians. NaN yields no state change.
        joints may carry original keypoints if needed for debugging (optional).
        Returns an event dict: {frame_idx, exercise, rep_event, angles, flags}
        """
        self._frame_idx = int(frame_idx) if frame_idx is not None else (self._frame_idx + 1)
        knee_angle = float(features.get("knee_angle", np.nan))

        rep_event: Optional[str] = None
        flags: Dict[str, bool] = {"in_bottom": self.state.in_bottom}

        if np.isfinite(knee_angle):
            if not self.state.in_bottom and knee_angle <= self.thresholds.go_down:
                self.state.in_bottom = True
                flags["in_bottom"] = True
            elif self.state.in_bottom and knee_angle >= self.thresholds.go_up:
                self.state.in_bottom = False
                flags["in_bottom"] = False
                self.state.reps += 1
                rep_event = "rep_complete"

        event = {
            "frame_idx": self._frame_idx,
            "exercise": "squat",
            "rep_event": rep_event,
            "angles": {"knee": knee_angle},
            "flags": flags,
        }
        return event


