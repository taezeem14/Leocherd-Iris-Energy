"""
Video file and container format validator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from iris.core.exceptions import VideoProcessingError

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

# Common container magic bytes
MAGIC_SIGNATURES = {
    b"ftyp": "ISO Media (MP4 / MOV)",
    b"RIFF": "Resource Interchange File Format (AVI)",
    b"\x1a\x45\xdf\xa3": "Matroska / WebM",
}


class VideoValidator:
    """Performs integrity, magic byte, and extension checks on video inputs."""

    @staticmethod
    def validate_file(file_path: str | Path, max_size_mb: float = 500.0) -> Tuple[bool, List[str]]:
        path = Path(file_path)
        issues: List[str] = []

        if not path.exists():
            raise VideoProcessingError(f"Video file does not exist: {file_path}")

        if not path.is_file():
            raise VideoProcessingError(f"Path is not a regular file: {file_path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise VideoProcessingError(f"Video file is empty (0 bytes): {file_path}")

        size_mb = file_size / (1024 * 1024)
        if size_mb > max_size_mb:
            issues.append(f"File size ({size_mb:.1f} MB) exceeds maximum threshold of {max_size_mb} MB.")

        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            issues.append(
                f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # Check binary magic bytes
        try:
            with open(path, "rb") as f:
                header = f.read(64)
                # MP4/MOV usually has 'ftyp' at offset 4
                is_mp4_mov = b"ftyp" in header[:16]
                is_riff = header.startswith(b"RIFF")
                is_webm = header.startswith(b"\x1a\x45\xdf\xa3")

                if not (is_mp4_mov or is_riff or is_webm or ext in ALLOWED_EXTENSIONS):
                    issues.append("Binary header does not match known video container signatures.")
        except Exception as e:
            issues.append(f"Failed to read file header: {e}")

        if issues:
            return False, issues
        return True, []
