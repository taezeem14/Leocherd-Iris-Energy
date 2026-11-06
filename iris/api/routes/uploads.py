"""
Video upload endpoints with container validation and safe file storage.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile

from iris.core.config import get_settings
from iris.core.logging import get_logger
from iris.video.validation import ALLOWED_EXTENSIONS, VideoValidator

logger = get_logger("iris.api.uploads")
router = APIRouter(prefix="/uploads", tags=["Uploads"])


@router.post("")
async def upload_video(file: UploadFile = File(...)):
    filename = Path(file.filename).name
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    settings = get_settings()
    upload_id = str(uuid.uuid4())[:8]
    safe_filename = f"{upload_id}_{filename}"
    target_path = settings.uploads_dir / safe_filename

    try:
        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Failed to write uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded video to disk.")

    # Validate file integrity and size
    valid, issues = VideoValidator.validate_file(target_path)
    if not valid:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid video container: {'; '.join(issues)}")

    return {
        "upload_id": upload_id,
        "filename": filename,
        "path": str(target_path),
        "size_bytes": target_path.stat().st_size,
    }
