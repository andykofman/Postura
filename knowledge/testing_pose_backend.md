## Testing: Pose Backend

### Unit tests (fast, deterministic)
- File: `tests/test_pose_backend.py`
- Strategy: Mocked model with controlled landmark outputs
- Guarantees:
  - Always returns 33 entries
  - Values are normalized [0, 1]
  - No-pose → all `None`
  - Partial output → remainder `None`

Run:
```bash
python -m pytest -m "not integration"
```

### Integration test (real model + video)
- File: `tests/test_pose_backend_integration.py`
- Requires: `mediapipe`, `opencv-python`, and `tests/input.mp4` (falls back to `demo/input.mp4`)
- Purpose: sanity check real inference + minimal performance signal

Run:
```bash
python -m pytest -m integration
```



