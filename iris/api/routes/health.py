"""
Health check and environment status endpoints.
"""

from __future__ import annotations

import os
import shutil
from fastapi import APIRouter

from iris.api.schemas.job_schemas import HealthResponse
from iris.core.config import get_settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
def get_health() -> HealthResponse:
    settings = get_settings()
    usage = shutil.disk_usage(str(settings.project_root))
    free_disk = round(usage.free / (1024**3), 2)

    has_cuda = False
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    free_ram = 1.8
    try:
        import psutil

        free_ram = round(psutil.virtual_memory().available / (1024**3), 2)
    except ImportError:
        pass

    return HealthResponse(
        status="ok",
        version="0.1.0",
        active_profile=settings.active_profile,
        cpu_cores=os.cpu_count() or 1,
        has_cuda=has_cuda,
        free_ram_gb=free_ram,
        free_disk_gb=free_disk,
    )
