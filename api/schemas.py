from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FrameAngles(BaseModel):
    knee: Optional[float] = None
    elbow: Optional[float] = None


class FrameRecord(BaseModel):
    frame_index: int
    exercise: str
    rep_id: int
    is_form_ok: bool
    angles: Dict[str, float]


class ExerciseSummary(BaseModel):
    total_reps: int = Field(0, ge=0)
    good_form_reps: int = Field(0, ge=0)
    common_issues: List[str] = Field(default_factory=list)


class Summary(BaseModel):
    squats: ExerciseSummary
    pushups: ExerciseSummary


class AnalyzeResponse(BaseModel):
    video_id: str
    summary: Summary
    frame_data: List[FrameRecord]


class JobSubmitResponse(BaseModel):
    video_id: str
    status: str = Field(description="queued | processing | done | error")


class JobStatusResponse(BaseModel):
    video_id: str
    status: str
    detail: Optional[str] = None
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    eta_seconds: Optional[float] = Field(default=None, ge=0.0)


