from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .utils import HysteresisThresholds, PUSHUP_ELBOW_ANGLE


@dataclass
class PushupState:
    in_bottom: bool = False
    reps: int = 0


class PushupFSM:
    """
    Push-up FSM using elbow angle with hysteresis.

    - When elbow angle <= thresholds.go_down: enter bottom
    - When elbow angle >= thresholds.go_up: exit bottom, count a rep
    """

    def __init__(self, thresholds: HysteresisThresholds = PUSHUP_ELBOW_ANGLE) -> None:
        self.thresholds = thresholds
        self.state = PushupState()
        self._frame_idx = -1

    def reset(self) -> None:
        self.state = PushupState()
        self._frame_idx = -1

    def process_frame(
        self,
        features: Dict[str, float],
        *,  # keyword-only arguments
        frame_idx: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        features should contain an 'elbow_angle' in radians. NaN yields no state change.
        Returns an event dict: {frame_idx, exercise, rep_event, angles, flags}
        """
        if frame_idx is not None:
            self._frame_idx = int(frame_idx)
        else:
            self._frame_idx = self._frame_idx + 1    
        
        elbow_angle = float(features.get("elbow_angle", np.nan))

        rep_event: Optional[str] = None
        flags: Dict[str, bool] = {"in_bottom": self.state.in_bottom}

        if np.isfinite(elbow_angle):
            if not self.state.in_bottom and elbow_angle <= self.thresholds.go_down:
                self.state.in_bottom = True
                flags["in_bottom"] = True
            elif self.state.in_bottom and elbow_angle >= self.thresholds.go_up:
                self.state.in_bottom = False
                flags["in_bottom"] = False
                self.state.reps += 1
                rep_event = "rep_complete"

        event = {
            "frame_idx": self._frame_idx,
            "exercise": "pushup",
            "rep_event": rep_event,
            "angles": {"elbow": elbow_angle},
            "flags": flags,
        }
        return event


