"""
Abstract compute provider interface and job schemas for hybrid execution.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProviderCapabilities(BaseModel):
    provider_name: str
    device_type: str  # "cpu", "cuda", "cloud"
    is_available: bool = True
    cpu_cores: int = 1
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    estimated_inference_fps: float = 15.0
    notes: str = ""


class JobSpecification(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    video_path: str
    morphology: str = "human"
    target_fps: int = 15
    export_formats: List[str] = Field(default_factory=lambda: ["gltf", "bvh", "blender"])
    output_dir: Optional[str] = None
    config_overrides: Dict[str, Any] = Field(default_factory=dict)


class JobStatus(BaseModel):
    job_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    current_stage: str = ""
    message: str = ""
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    success: bool
    status: ExecutionStatus
    execution_time_sec: float
    output_assets: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class ComputeProvider(ABC):
    """Abstract interface defining uniform execution across Local CPU, Local GPU, and Cloud."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def detect_capabilities(self) -> ProviderCapabilities:
        """Inspects and returns current device/provider execution capabilities."""
        pass

    @abstractmethod
    def submit_job(self, job_spec: JobSpecification) -> str:
        """Enqueues or initiates an asynchronous reconstruction job. Returns job_id."""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Polls current status and progress of the specified job."""
        pass

    @abstractmethod
    def retrieve_result(self, job_id: str) -> JobResult:
        """Retrieves finalized job artifacts, metrics, and asset paths."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Requests cancellation of an ongoing job."""
        pass
