"""
Safe model downloading with SHA-256 verification and Drive D storage isolation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import requests

from iris.core.config import get_settings
from iris.core.exceptions import ModelLoadError
from iris.core.logging import get_logger
from iris.models.registry import MODEL_REGISTRY, ModelMetadata

logger = get_logger("iris.models.downloads")


class ModelDownloader:
    """Manages downloading model checkpoints securely to Drive D cache."""

    def __init__(self, target_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.target_dir = target_dir or settings.models_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_key: str) -> Path:
        metadata = MODEL_REGISTRY.get(model_key)
        if not metadata:
            raise ModelLoadError(f"Unknown model key '{model_key}'. Registered: {list(MODEL_REGISTRY.keys())}")
        if metadata.source_url == "builtin":
            return Path("builtin")

        filename = Path(metadata.source_url.split("?")[0]).name
        return self.target_dir / filename

    def is_downloaded(self, model_key: str) -> bool:
        path = self.get_model_path(model_key)
        return path == Path("builtin") or (path.exists() and path.stat().st_size > 1000)

    def ensure_model(self, model_key: str, force_download: bool = False) -> Path:
        metadata = MODEL_REGISTRY.get(model_key)
        if not metadata:
            raise ModelLoadError(f"Model '{model_key}' not found in registry.")

        if metadata.source_url == "builtin":
            return Path("builtin")

        dest_path = self.get_model_path(model_key)
        if dest_path.exists() and not force_download and dest_path.stat().st_size > 1000:
            logger.info(f"Model '{model_key}' already present at {dest_path}")
            return dest_path

        logger.info(f"Downloading model '{model_key}' ({metadata.size_mb:.1f} MB) from {metadata.source_url}...")
        try:
            response = requests.get(metadata.source_url, stream=True, timeout=30)
            response.raise_for_status()

            temp_path = dest_path.with_suffix(".tmp")
            hasher = hashlib.sha256() if metadata.sha256 else None

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        if hasher:
                            hasher.update(chunk)

            if metadata.sha256 and hasher:
                computed_hash = hasher.hexdigest()
                if computed_hash != metadata.sha256:
                    temp_path.unlink(missing_ok=True)
                    raise ModelLoadError(
                        f"SHA-256 checksum mismatch for {model_key}. Expected: {metadata.sha256}, Got: {computed_hash}"
                    )

            temp_path.replace(dest_path)
            logger.info(f"Successfully saved {model_key} to {dest_path}")
            return dest_path

        except Exception as e:
            logger.warning(f"Download of {model_key} failed ({e}). Falling back to built-in tracker.")
            return Path("builtin")
