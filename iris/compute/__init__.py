"""
Compute provider abstraction subsystem for local and cloud execution.
"""

from iris.compute.base import (
    ComputeProvider,
    ExecutionStatus,
    JobResult,
    JobSpecification,
    JobStatus,
    ProviderCapabilities,
)
from iris.compute.kaggle import KaggleExperimentProvider
from iris.compute.local_cpu import LocalCPUProvider
from iris.compute.local_gpu import LocalGPUProvider

__all__ = [
    "ComputeProvider",
    "ExecutionStatus",
    "JobSpecification",
    "JobStatus",
    "JobResult",
    "ProviderCapabilities",
    "LocalCPUProvider",
    "LocalGPUProvider",
    "KaggleExperimentProvider",
]
