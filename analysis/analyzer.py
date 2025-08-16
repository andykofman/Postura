from __future__ import annotations

import statistics
import uuid
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pose.backend import Keypoint
from pose.smoothing import EmaSmoother, SmoothedKeypoint
from .features import angle, horiz_offset, collinearity_residual
from .fsm_pushup import PushupFSM
from .fsm_squat import SquatFSM


# MediaPipe BlazePose landmark indices (subset used here)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def _select_valid_triplet(
    triples: List[Tuple[int, int, int]],
    smoothed: Sequence[SmoothedKeypoint],
    valid_mask: Sequence[bool],
) -> Optional[Tuple[Keypoint, Keypoint, Keypoint]]:
    """
    Choose the triplet where all joints are valid; if multiple, pick the one with higher sum confidence.
    return: a tuple of Keypoints (A,B,C) or None if not valid.
    """
    best = None
    best_score = -1.0
    for a, b, c in triples:
        # check if the indices are within the range of the smoothed keypoints
        if a >= len(smoothed) or b >= len(smoothed) or c >= len(smoothed):
            continue
        # check if the keypoints are valid
        if not (valid_mask[a] and valid_mask[b] and valid_mask[c]):
            continue
        ka, kb, kc = smoothed[a], smoothed[b], smoothed[c]
        score = float(ka.confidence + kb.confidence + kc.confidence)
        if score > best_score:
            best_score = score
            best = (
                Keypoint(ka.x, ka.y, ka.confidence),
                Keypoint(kb.x, kb.y, kb.confidence),
                Keypoint(kc.x, kc.y, kc.confidence),
            )
    return best


def _select_valid_pair(
    pairs: List[Tuple[int, int]],
    smoothed: Sequence[SmoothedKeypoint],
    valid_mask: Sequence[bool],
) -> Optional[Tuple[Keypoint, Keypoint]]:
    best = None
    best_score = -1.0
    for a, b in pairs:
        if a >= len(smoothed) or b >= len(smoothed):
            continue
        if not (valid_mask[a] and valid_mask[b]):
            continue
        ka, kb = smoothed[a], smoothed[b]
        score = float(ka.confidence + kb.confidence)
        if score > best_score:
            best_score = score
            best = (
                Keypoint(ka.x, ka.y, ka.confidence),
                Keypoint(kb.x, kb.y, kb.confidence),
            )
    return best


def analyze_video(frames_iter: Iterable[Sequence[Optional[Keypoint]]]) -> Dict[str, object]:
    """
    Analyze a stream of per-frame keypoints and return a JSON-serializable dict.

    Assumptions:
    - frames_iter yields per-frame lists of 33 keypoints (Keypoint or None) from PoseBackend.infer
    - Coordinates are normalized [0,1]

    Output structure (example):
    {
      "video_id": "<uuid4>",
      "summary": {
        "squats": {"total_reps": int, "good_form_reps": int, "common_issues": ["INSUFFICIENT_DEPTH", ...]},
        "pushups": {"total_reps": int, "good_form_reps": int, "common_issues": ["BODY_LINE_BREAK", ...]}
      },
      "frame_data": [
        {"frame_index": 0, "exercise": "squat", "rep_id": 1, "is_form_ok": true, "angles": {"knee": 85.0}},
        {"frame_index": 42, "exercise": "pushup", "rep_id": 1, "is_form_ok": false, "angles": {"elbow": 110.0}}
      ]
    }
    """
    video_id = str(uuid.uuid4())

    # Use alpha=1.0 (no smoothing) to avoid attenuating threshold crossings for FSMs
    smoother = EmaSmoother(num_landmarks=33, alpha=1.0, min_confidence=0.5)
    pushup_fsm = PushupFSM()
    squat_fsm = SquatFSM()

    frame_records: List[Dict[str, object]] = []

    # For summary stats
    elbow_angles: List[float] = []
    knee_angles: List[float] = []
    align_residuals_push: List[float] = []
    align_residuals_squat: List[float] = []

    # Per-exercise rep tracking
    squat_rep_id = 0
    push_rep_id = 0
    squat_good = 0
    push_good = 0

    for frame_idx, keypoints in enumerate(frames_iter):
        smoothed, mask = smoother.update(keypoints)

        # Elbow angle (choose best side)
        elbows = _select_valid_triplet(
            [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)], smoothed, mask
        )
        if elbows is None:
            elbow_theta = float("nan")
        else:
            a, b, c = elbows
            elbow_theta = angle(a, b, c)
        if np.isfinite(elbow_theta):
            elbow_angles.append(float(elbow_theta))

        # Knee angle (choose best side)
        knees = _select_valid_triplet(
            [(L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)], smoothed, mask
        )
        if knees is None:
            knee_theta = float("nan")
        else:
            a, b, c = knees
            knee_theta = angle(a, b, c)
        if np.isfinite(knee_theta):
            knee_angles.append(float(knee_theta))

        # Horizontal offset (knee vs ankle) on the same chosen side as knee angle if available
        if knees is not None:
            _, knee_kp, ankle_kp = knees
            knee_ankle_offset = float(horiz_offset(knee_kp, ankle_kp))
        else:
            knee_ankle_offset = float("nan")

        # Alignment residual using hip-shoulder line vs ankle deviation (squat alignment)
        hips = _select_valid_pair([(L_HIP, L_SHOULDER), (R_HIP, R_SHOULDER)], smoothed, mask)
        ankles = _select_valid_pair([(L_ANKLE, L_ANKLE), (R_ANKLE, R_ANKLE)], smoothed, mask)
        if hips is not None and ankles is not None:
            # Choose ankle corresponding to the chosen hip/shoulder side
            # Note: ankles selector returns a duplicate pair by construction; take first element
            hip_kp, shoulder_kp = hips
            ankle_kp = ankles[0]
            align_resid = float(collinearity_residual(hip_kp, shoulder_kp, ankle_kp))
        else:
            align_resid = float("nan")

        # Also track a pushup alignment residual (same metric works for body alignment)
        # 
        if np.isfinite(align_resid):
            align_residuals_squat.append(abs(align_resid))
            align_residuals_push.append(abs(align_resid))

        features = {
            "elbow_angle": float(elbow_theta),
            "knee_angle": float(knee_theta),
            "knee_ankle_horiz_offset": float(knee_ankle_offset),
            "hip_shoulder_ankle_collinearity": float(align_resid),
        }

        ev_push = pushup_fsm.process_frame(features, frame_idx=frame_idx)
        ev_squat = squat_fsm.process_frame(features, frame_idx=frame_idx)

        # On rep completion, emit a record matching assignment schema
        if ev_squat.get("rep_event") == "rep_complete":
            squat_rep_id += 1
            # Define form OK heuristics at event time
            # Form Assessment:
            # 1. Knee angle is less than 110 degrees
            # 2. Alignment residual is less than 0.05
            knee_degree = float(np.degrees(features["knee_angle"])) if np.isfinite(features["knee_angle"]) else float("nan")
            align_resid = float(features["hip_shoulder_ankle_collinearity"]) if np.isfinite(features["hip_shoulder_ankle_collinearity"]) else float("nan")
            # Form quality heuristics
            is_ok = (np.isfinite(knee_degree) and knee_degree <= 110.0) and (np.isfinite(align_resid) and abs(align_resid) <= 0.05)
            if is_ok:
                squat_good += 1
            frame_records.append(
                {
                    "frame_index": frame_idx,
                    "exercise": "squat",
                    "rep_id": squat_rep_id,
                    "is_form_ok": bool(is_ok),
                    "angles": {"knee": knee_degree},
                }
            )

        if ev_push.get("rep_event") == "rep_complete":
            push_rep_id += 1
            elbow_deg = float(np.degrees(features["elbow_angle"])) if np.isfinite(features["elbow_angle"]) else float("nan")
            align_resid = float(features["hip_shoulder_ankle_collinearity"]) if np.isfinite(features["hip_shoulder_ankle_collinearity"]) else float("nan")
            is_ok = (np.isfinite(elbow_deg) and elbow_deg <= 90.0) and (np.isfinite(align_resid) and abs(align_resid) <= 0.05)
            if is_ok:
                push_good += 1
            frame_records.append(
                {
                    "frame_index": frame_idx,
                    "exercise": "pushup",
                    "rep_id": push_rep_id,
                    "is_form_ok": bool(is_ok),
                    "angles": {"elbow": elbow_deg},
                }
            )

    # Summaries and common issues (heuristics)
    push_summary: Dict[str, object] = {"total_reps": pushup_fsm.state.reps, "good_form_reps": push_good, "common_issues": []}
    squat_summary: Dict[str, object] = {"total_reps": squat_fsm.state.reps, "good_form_reps": squat_good, "common_issues": []}

    if elbow_angles:
        min_elbow = float(min(elbow_angles))
        # If never went below approx 90 deg, report shallow pushups
        if min_elbow > np.deg2rad(95.0):
            push_summary["common_issues"].append("INSUFFICIENT_DEPTH")
    else:
        push_summary["common_issues"].append("NO_ELBOW_SIGNAL")

    if knee_angles:
        min_knee = float(min(knee_angles))
        # If never went below ~110 deg, squat may be shallow
        if min_knee > np.deg2rad(115.0):
            squat_summary["common_issues"].append("INSUFFICIENT_DEPTH")
    else:
        squat_summary["common_issues"].append("NO_KNEE_SIGNAL")

    if align_residuals_squat:
        mean_align = float(statistics.fmean(align_residuals_squat))
        if mean_align > 0.05:
            squat_summary["common_issues"].append("BODY_LINE_BREAK")
    if align_residuals_push:
        mean_align_p = float(statistics.fmean(align_residuals_push))
        if mean_align_p > 0.05:
            push_summary["common_issues"].append("BODY_LINE_BREAK")

    result: Dict[str, object] = {
        "video_id": video_id,
        "summary": {"squats": squat_summary, "pushups": push_summary},
        "frame_data": frame_records,
    }
    return result


