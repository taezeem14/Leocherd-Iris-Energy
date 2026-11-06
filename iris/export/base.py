"""
Abstract base class for 3D and animation asset exporters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from iris.retargeting.animation import SkeletalAnimation
from iris.rigging.skeleton import Skeleton


class BaseAssetExporter(ABC):
    """Abstract interface for all file exporters."""

    def __init__(self, name: str, extension: str) -> None:
        self.name = name
        self.extension = extension

    @abstractmethod
    def export(
        self,
        skeleton: Skeleton,
        animation: SkeletalAnimation,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Exports the skeleton and animation to disk."""
        pass
