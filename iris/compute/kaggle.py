"""
Kaggle cloud experiment provider for offloading GPU training, benchmarking, and inference.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris.compute.base import (
    ComputeProvider,
    ExecutionStatus,
    JobResult,
    JobSpecification,
    JobStatus,
    ProviderCapabilities,
)
from iris.core.config import get_settings
from iris.core.exceptions import HardwareConstraintError
from iris.core.logging import get_logger

logger = get_logger("iris.compute.kaggle")


class KaggleExperimentProvider(ComputeProvider):
    """
    Manages packaging, preparing, and tracking remote GPU experiments on Kaggle Notebooks.
    Adheres strictly to zero-unauthorized-upload privacy policy.
    """

    def __init__(self, experiment_dir: Optional[Path] = None) -> None:
        super().__init__(name="KaggleExperimentProvider")
        settings = get_settings()
        self.experiment_dir = experiment_dir or (settings.project_root / "experiments" / "kaggle")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobStatus] = {}

    def _check_kaggle_auth(self) -> bool:
        """Checks for official Kaggle credentials without leaking secrets."""
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        has_file = kaggle_json.exists()
        has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        return has_file or has_env

    def detect_capabilities(self) -> ProviderCapabilities:
        has_auth = self._check_kaggle_auth()
        return ProviderCapabilities(
            provider_name=self.name,
            device_type="cloud",
            is_available=has_auth,
            gpu_name="NVIDIA Tesla T4 / P100 (Cloud GPU)",
            vram_gb=16.0,
            estimated_inference_fps=60.0,
            notes=(
                "Kaggle Cloud GPU Provider available. Credentials detected."
                if has_auth
                else "Kaggle credentials (~/.kaggle/kaggle.json) not detected. Standalone notebook generation active."
            ),
        )

    def prepare_experiment_bundle(
        self,
        experiment_name: str,
        notebook_filename: str,
        config: Dict[str, Any],
    ) -> Path:
        """
        Creates a self-contained, reproducible experiment bundle containing
        the notebook, configuration JSON, and required dependencies.
        """
        bundle_dir = self.experiment_dir / "bundles" / experiment_name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        config_path = bundle_dir / "experiment_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        source_nb = self.experiment_dir / notebook_filename
        if source_nb.exists():
            shutil.copy2(source_nb, bundle_dir / notebook_filename)

        logger.info(f"Prepared Kaggle experiment bundle at {bundle_dir}")
        return bundle_dir

    def submit_job(self, job_spec: JobSpecification) -> str:
        # Enforce Privacy Safeguard
        allow_cloud = job_spec.config_overrides.get("allow_cloud_compute", False)
        if not allow_cloud:
            raise HardwareConstraintError(
                "Cloud offloading rejected: 'allow_cloud_compute' is False. "
                "Explicit interactive user authorization is mandatory before transmitting data to cloud providers."
            )

        job_id = f"kaggle_{job_spec.job_id}"
        logger.info(f"Preparing Kaggle job {job_id} for video: {job_spec.video_path}")

        # Prepare experiment bundle
        bundle_path = self.prepare_experiment_bundle(
            experiment_name=job_id,
            notebook_filename="03_3d_pose_experiments.ipynb",
            config=job_spec.model_dump(),
        )

        status = JobStatus(
            job_id=job_id,
            status=ExecutionStatus.PENDING,
            progress=0.1,
            message=f"Kaggle experiment bundle generated at {bundle_path}. Ready for notebook run.",
        )
        self._jobs[job_id] = status
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        return self._jobs.get(
            job_id,
            JobStatus(job_id=job_id, status=ExecutionStatus.FAILED, message="Job not found."),
        )

    def retrieve_result(self, job_id: str) -> JobResult:
        settings = get_settings()
        artifact_dir = settings.outputs_dir / "experiments" / job_id

        if not artifact_dir.exists():
            return JobResult(
                job_id=job_id,
                success=False,
                status=ExecutionStatus.PENDING,
                execution_time_sec=0.0,
                error_message=f"Kaggle experiment artifacts not found at {artifact_dir}.",
            )

        # Collect exported files
        assets = {f.name: str(f) for f in artifact_dir.glob("*.*")}
        return JobResult(
            job_id=job_id,
            success=True,
            status=ExecutionStatus.COMPLETED,
            execution_time_sec=0.0,
            output_assets=assets,
        )

    def cancel_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].status = ExecutionStatus.CANCELLED
            return True
        return False
