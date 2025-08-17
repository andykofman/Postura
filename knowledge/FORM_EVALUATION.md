## Form evaluation and tuning

This document explains how rep quality is determined and how to tune the thresholds. The logic primarily lives in `analysis/analyzer.py` with configuration in `analysis/utils.py`.

### Overview

- Rep counting remains driven by hysteresis FSMs (`analysis/fsm_squat.py`, `analysis/fsm_pushup.py`).
- Form evaluation happens at rep completion using measurements accumulated during the bottom phase.
- Smoothing uses `EmaSmoother` with `alpha=1.0` to avoid attenuating threshold crossings.

### Squat: per-rep pass criteria

In `analysis/analyzer.py`, when a squat rep completes, we compute:

- Depth: minimum knee angle (degrees) during the bottom phase must be ≤ `SQUAT_KNEE_MAX_DEG`
  - Default: `110.0`
- Alignment: 90th percentile of the absolute hip–shoulder–ankle collinearity residual during the bottom phase must be ≤ the active tolerance
  - Base tolerance: `SQUAT_ALIGNMENT_TOL = 0.12`
  - Adaptive tolerance for deep squats: if knee angle ≤ 100°, allow `SQUAT_ALIGNMENT_TOL_DEEP = 0.16`
- Knee-over-ankle proxy: absolute horizontal knee–ankle offset should be ≤ `SQUAT_KNEE_ANKLE_MAX_OFFSET = 0.20`
  - Used as a secondary guardrail against excessive forward knee travel.

Implementation notes:

- Alignment residual is computed side-consistently: hip/shoulder and ankle are taken from the same side chosen by confidence, avoiding left/right mixing.
- Alignment robustness uses the 90th percentile over the bottom phase instead of a single worst frame to reduce false negatives from jitter/spikes.

### Push-up: per-rep pass criteria (unchanged)

- Elbow depth: minimum elbow angle ≤ `PUSHUP_ELBOW_MAX_DEG = 90.0`
- Alignment: max absolute residual ≤ `PUSHUP_ALIGNMENT_TOL = 0.08`

### Summary issue flags

At the end of analysis, common issues are inferred:

- `INSUFFICIENT_DEPTH` if the global minimum joint angle never crossed a slackened bound.
- `BODY_LINE_BREAK` if the mean absolute alignment residual exceeds the corresponding tolerance (`SQUAT_ALIGNMENT_TOL`/`PUSHUP_ALIGNMENT_TOL`).

### Tuning guide (moderate strictness)

- Squat alignment tolerance: `SQUAT_ALIGNMENT_TOL` in the range 0.12–0.14.
- Deep-squat tolerance: `SQUAT_ALIGNMENT_TOL_DEEP` in the range 0.16–0.18.
- Deep cutoff: knee angle threshold for using deep tolerance, 100–105°.
- Knee–ankle offset: `SQUAT_KNEE_ANKLE_MAX_OFFSET` in the range 0.18–0.22.

All tunables live in `analysis/utils.py`.

### File references

- `analysis/analyzer.py`: form evaluation, side-consistent alignment, robust percentile metric.
- `analysis/utils.py`: `SQUAT_ALIGNMENT_TOL`, `SQUAT_ALIGNMENT_TOL_DEEP`, `SQUAT_KNEE_MAX_DEG`, `SQUAT_KNEE_ANKLE_MAX_OFFSET`, push-up counterparts.
- `analysis/fsm_squat.py`, `analysis/fsm_pushup.py`: FSM rep detection thresholds.


