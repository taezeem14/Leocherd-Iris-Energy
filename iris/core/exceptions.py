"""
Domain-specific exceptions for Leocherd Iris Energy.
"""

from typing import Any, Dict, Optional


class IrisError(Exception):
    """Base exception for all Iris Energy errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class VideoProcessingError(IrisError):
    """Raised when video reading, codec decoding, or frame extraction fails."""


class ModelLoadError(IrisError):
    """Raised when loading weights, ONNX sessions, or inference runtimes fails."""


class PoseEstimationError(IrisError):
    """Raised when pose detection or landmark extraction encounters an unrecoverable fault."""


class MotionError(IrisError):
    """Raised during temporal motion analysis, velocity derivation, or filtering."""


class MorphologyError(IrisError):
    """Raised when joint topology, bone hierarchy, or morphology validation fails."""


class SkeletonError(IrisError):
    """Raised when reconstructing 3D skeleton graph, quaternions, or bone chains fails."""


class ExportError(IrisError):
    """Raised when asset serialization (glTF, BVH, Blender script) fails."""


class HardwareConstraintError(IrisError):
    """Raised when the execution environment cannot fulfill memory or compute constraints."""
