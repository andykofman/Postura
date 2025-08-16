from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HysteresisThresholds:
    """Pair of thresholds to avoid chatter: enter condition at go_down, exit at go_up."""
    go_down: float  # radians, tighter/stricter direction
    go_up: float    # radians, looser opposite direction


# Default thresholds (radians) for simple unit tests and demos.
# will be adjusted per dataset later
SQUAT_KNEE_ANGLE = HysteresisThresholds(
    go_down=math.radians(110.0),  # entering squat when knee angle <= 110°
    go_up=math.radians(150.0),    # exiting squat when knee angle >= 150°
)

PUSHUP_ELBOW_ANGLE = HysteresisThresholds(
    go_down=math.radians(90.0),   # entering pushup when elbow angle <= 90°
    go_up=math.radians(150.0),    # exiting pushup when elbow angle >= 150°
)


