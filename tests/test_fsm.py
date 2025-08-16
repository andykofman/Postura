from __future__ import annotations

import math

import numpy as np

from analysis.fsm_squat import SquatFSM
from analysis.fsm_pushup import PushupFSM
from analysis.utils import SQUAT_KNEE_ANGLE, PUSHUP_ELBOW_ANGLE


def test_squat_fsm_deterministic_sequence():
    fsm = SquatFSM(SQUAT_KNEE_ANGLE)
    fsm.reset()

    # Sequence: start high (standing), go below go_down, then above go_up => 1 rep
    angles = [math.radians(160), math.radians(120), math.radians(100), math.radians(160)]
    events = [fsm.process_frame({"knee_angle": a}, frame_idx=i) for i, a in enumerate(angles)]

    assert events[0]["flags"]["in_bottom"] is False
    assert events[1]["flags"]["in_bottom"] is False  # not yet below 110
    assert events[2]["flags"]["in_bottom"] is True   # entered bottom
    assert events[3]["rep_event"] == "rep_complete"   # exited -> counted rep
    assert fsm.state.reps == 1


def test_pushup_fsm_deterministic_sequence():
    fsm = PushupFSM(PUSHUP_ELBOW_ANGLE)
    fsm.reset()

    # Sequence: start high (elbow open), drop below go_down, then rise above go_up => 1 rep
    angles = [math.radians(170), math.radians(120), math.radians(80), math.radians(160)]
    events = [fsm.process_frame({"elbow_angle": a}, frame_idx=i) for i, a in enumerate(angles)]

    assert events[0]["flags"]["in_bottom"] is False
    assert events[1]["flags"]["in_bottom"] is False
    assert events[2]["flags"]["in_bottom"] is True
    assert events[3]["rep_event"] == "rep_complete"
    assert fsm.state.reps == 1


