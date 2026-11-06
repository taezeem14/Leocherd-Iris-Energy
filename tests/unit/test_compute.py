"""
Unit tests for hybrid compute provider abstraction.
"""

import pytest
from iris.compute.base import JobSpecification
from iris.compute.kaggle import KaggleExperimentProvider
from iris.compute.local_cpu import LocalCPUProvider
from iris.compute.local_gpu import LocalGPUProvider
from iris.core.exceptions import HardwareConstraintError


def test_local_cpu_capabilities():
    provider = LocalCPUProvider()
    caps = provider.detect_capabilities()
    assert caps.is_available is True
    assert caps.device_type == "cpu"


def test_local_gpu_capabilities():
    provider = LocalGPUProvider()
    caps = provider.detect_capabilities()
    assert caps.device_type in ("cuda", "none")


def test_kaggle_provider_privacy_protection():
    provider = KaggleExperimentProvider()
    # Attempting to submit without allow_cloud_compute must raise HardwareConstraintError
    spec = JobSpecification(
        video_path="sample.mp4",
        config_overrides={"allow_cloud_compute": False},
    )
    with pytest.raises(HardwareConstraintError):
        provider.submit_job(spec)
