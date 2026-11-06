"""
Pydantic schemas for API request and response validation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    active_profile: str
    cpu_cores: int
    has_cuda: bool
    free_ram_gb: float
    free_disk_gb: float
    timestamp: float = Field(default_factory=time.time)


class JobCreateRequest(BaseModel):
    video_path: str
    morphology: str = "human"
    target_fps: int = 15
    export_formats: List[str] = Field(default_factory=lambda: ["gltf", "bvh", "blender"])
    allow_cloud_compute: bool = False
    config_overrides: Dict[str, Any] = Field(default_factory=dict)


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    current_stage: str
    message: str
    execution_time_sec: Optional[float] = None
    output_assets: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None


class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    created_at: float = Field(default_factory=time.time)
