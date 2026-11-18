"""
Central configuration and hardware-aware settings for Leocherd Iris Energy.
"""

from __future__ import annotations

import os
import platform
import shutil
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionProfile(str, Enum):
    AUTO = "AUTO"
    CPU_LIGHTWEIGHT = "CPU_LIGHTWEIGHT"
    CPU_STANDARD = "CPU_STANDARD"
    CUDA_GPU = "CUDA_GPU"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IRIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    cache_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / ".cache")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "models")
    uploads_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "uploads")
    outputs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "outputs")
    samples_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "samples")

    # Execution & Hardware
    hardware_profile: ExecutionProfile = ExecutionProfile.AUTO
    active_profile: str = "CPU_LIGHTWEIGHT"

    # Ingestion constraints
    max_video_duration_sec: float = 180.0
    default_sample_fps: int = 15
    max_frame_dimension: int = 1280
    min_confidence_threshold: float = 0.35

    # Tool paths
    blender_executable: Optional[str] = None
    ffmpeg_executable: Optional[str] = None

    # Server Settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_debug: bool = False

    def model_post_init(self, __context) -> None:
        # Create required directories on disk
        for d in [self.cache_dir, self.models_dir, self.uploads_dir, self.outputs_dir, self.samples_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Resolve auto hardware profile
        if self.hardware_profile == ExecutionProfile.AUTO:
            self.active_profile = self._detect_profile()
        else:
            self.active_profile = self.hardware_profile.value

        # Locate ffmpeg if not set
        if not self.ffmpeg_executable:
            found = shutil.which("ffmpeg")
            if not found and platform.system() == "Windows":
                for candidate in [
                    r"C:\Program Files\Softdeluxe\Free Download Manager\ffmpeg.exe",
                    r"C:\ffmpeg\bin\ffmpeg.exe",
                    r"D:\ffmpeg\bin\ffmpeg.exe",
                ]:
                    if os.path.exists(candidate):
                        found = candidate
                        break
            self.ffmpeg_executable = found

        # Locate blender if not set
        if not self.blender_executable:
            found_b = shutil.which("blender")
            if not found_b and platform.system() == "Windows":
                for b_cand in [
                    r"D:\blender\blender-4.2.3-windows-x64\blender.exe",
                    r"D:\blender\blender.exe",
                    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
                    r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
                    r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
                    r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
                    r"D:\Program Files\Blender Foundation\Blender\blender.exe",
                ]:
                    if os.path.exists(b_cand):
                        found_b = b_cand
                        break
            self.blender_executable = found_b

    def _detect_profile(self) -> str:
        # Check CUDA
        try:
            import torch

            if torch.cuda.is_available():
                return ExecutionProfile.CUDA_GPU.value
        except ImportError:
            pass

        cores = os.cpu_count() or 1
        if cores <= 2:
            return ExecutionProfile.CPU_LIGHTWEIGHT.value
        return ExecutionProfile.CPU_STANDARD.value


@lru_cache()
def get_settings() -> Settings:
    return Settings()
