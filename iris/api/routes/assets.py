"""
Rig, animation, and exported asset retrieval endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from iris.core.config import get_settings

router = APIRouter(tags=["Assets & Downloads"])


@router.get("/rigs/{rig_id}")
def get_rig(rig_id: str):
    settings = get_settings()
    rig_file = settings.outputs_dir / rig_id / f"reconstruction_{rig_id}_rig_data.json"
    if not rig_file.exists():
        # Look for any json in folder
        candidates = list((settings.outputs_dir / rig_id).glob("*rig_data.json"))
        if candidates:
            rig_file = candidates[0]
        else:
            raise HTTPException(status_code=404, detail="Rig asset not found.")

    with open(rig_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/animations/{animation_id}")
def get_animation(animation_id: str):
    settings = get_settings()
    anim_file = settings.outputs_dir / animation_id / f"reconstruction_{animation_id}_anim_data.json"
    if not anim_file.exists():
        candidates = list((settings.outputs_dir / animation_id).glob("*anim_data.json"))
        if candidates:
            anim_file = candidates[0]
        else:
            raise HTTPException(status_code=404, detail="Animation asset not found.")

    with open(anim_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/exports/{job_id}/download/{fmt}")
def download_asset(job_id: str, fmt: str):
    settings = get_settings()
    out_dir = settings.outputs_dir / job_id
    fmt_clean = fmt.lower().lstrip(".")

    target_files = list(out_dir.glob(f"*.{fmt_clean}"))
    if not target_files:
        raise HTTPException(status_code=404, detail=f"No asset found with format '{fmt_clean}' for job {job_id}")

    file_to_send = target_files[0]
    return FileResponse(
        path=str(file_to_send),
        filename=file_to_send.name,
        media_type="application/octet-stream",
    )
