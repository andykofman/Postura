from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def _load_tiny_mp4() -> bytes:
    # Use provided tests/input.mp4 as a tiny fixture
    p = Path(__file__).resolve().parents[1] / "tests" / "input.mp4"
    if not p.exists():
        pytest.skip("tests/input.mp4 not found for API integration test")
    data = p.read_bytes()
    # Ensure it's small enough
    assert len(data) < 50 * 1024 * 1024
    return data


def test_analyze_endpoint_happy_path():
    data = _load_tiny_mp4()
    resp = client.post(
        "/analyze",
        files={"file": ("input.mp4", io.BytesIO(data), "video/mp4")},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "video_id" in payload and "summary" in payload and "frame_data" in payload
    # basic shape checks
    assert "squats" in payload["summary"] and "pushups" in payload["summary"]


def test_analyze_endpoint_rejects_large_file(monkeypatch):
    # 51MB dummy
    big = b"0" * (51 * 1024 * 1024)
    resp = client.post(
        "/analyze",
        files={"file": ("big.mp4", io.BytesIO(big), "video/mp4")},
    )
    assert resp.status_code == 413


def test_analyze_endpoint_unsupported_type():
    resp = client.post(
        "/analyze",
        files={"file": ("file.txt", io.BytesIO(b"not a video"), "text/plain")},
    )
    assert resp.status_code == 415


