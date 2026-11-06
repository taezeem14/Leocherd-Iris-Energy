"""
Reconstruction job lifecycle endpoints.
"""

from __future__ import annotations

import uuid
from typing import Dict
from fastapi import APIRouter, HTTPException

from iris.api.schemas.job_schemas import (
    JobCreateRequest,
    JobCreateResponse,
    JobStatusResponse,
    ProjectCreateRequest,
    ProjectResponse,
)
from iris.compute.base import JobSpecification
from iris.compute.local_cpu import LocalCPUProvider

router = APIRouter(tags=["Jobs & Projects"])
compute_provider = LocalCPUProvider()
_projects: Dict[str, ProjectResponse] = {}


@router.post("/projects", response_model=ProjectResponse)
def create_project(req: ProjectCreateRequest) -> ProjectResponse:
    pid = str(uuid.uuid4())[:8]
    resp = ProjectResponse(project_id=pid, name=req.name)
    _projects[pid] = resp
    return resp


@router.post("/jobs", response_model=JobCreateResponse)
def create_job(req: JobCreateRequest) -> JobCreateResponse:
    job_id = str(uuid.uuid4())[:8]
    spec = JobSpecification(
        job_id=job_id,
        video_path=req.video_path,
        morphology=req.morphology,
        target_fps=req.target_fps,
        export_formats=req.export_formats,
        config_overrides={"allow_cloud_compute": req.allow_cloud_compute, **req.config_overrides},
    )
    compute_provider.submit_job(spec)
    return JobCreateResponse(job_id=job_id, status="PENDING", message="Job submitted successfully.")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    status = compute_provider.get_job_status(job_id)
    result = compute_provider.retrieve_result(job_id) if status.status.value in ("COMPLETED", "FAILED") else None

    return JobStatusResponse(
        job_id=job_id,
        status=status.status.value,
        progress=round(status.progress, 2),
        current_stage=status.current_stage,
        message=status.message,
        execution_time_sec=result.execution_time_sec if result else None,
        output_assets=result.output_assets if result else {},
        error=status.error,
    )


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    success = compute_provider.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job; job may not be active.")
    return {"job_id": job_id, "status": "CANCELLED"}
