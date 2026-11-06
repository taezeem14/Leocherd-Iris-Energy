"""
Local CPU compute provider using multi-threaded ONNX Runtime and bounded memory execution.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from iris.compute.base import (
    ComputeProvider,
    ExecutionStatus,
    JobResult,
    JobSpecification,
    JobStatus,
    ProviderCapabilities,
)
from iris.core.config import get_settings
from iris.core.events import EventType, PipelineEvent
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, Pipeline
from iris.motion.reconstruction import MotionProcessingStage
from iris.pose.estimator import PoseEstimationStage
from iris.retargeting.animation import AnimationGenerationStage
from iris.rigging.topology import SkeletonReconstructionStage
from iris.video.ingestion import VideoIngestionStage

logger = get_logger("iris.compute.local_cpu")


class LocalCPUProvider(ComputeProvider):
    """Executes reconstruction pipeline locally on host CPU with resource caps."""

    def __init__(self) -> None:
        super().__init__(name="LocalCPUProvider")
        self._jobs: Dict[str, JobStatus] = {}
        self._results: Dict[str, JobResult] = {}
        self._cancel_flags: Dict[str, bool] = {}

    def detect_capabilities(self) -> ProviderCapabilities:
        cores = os.cpu_count() or 1
        return ProviderCapabilities(
            provider_name=self.name,
            device_type="cpu",
            is_available=True,
            cpu_cores=cores,
            gpu_name=None,
            vram_gb=None,
            estimated_inference_fps=15.0 if cores <= 2 else 30.0,
            notes="Active host profile: CPU_LIGHTWEIGHT (ONNX Runtime CPU, 2 threads).",
        )

    def submit_job(self, job_spec: JobSpecification) -> str:
        job_id = job_spec.job_id
        status = JobStatus(
            job_id=job_id,
            status=ExecutionStatus.PENDING,
            progress=0.0,
            message="Job queued for local CPU execution.",
        )
        self._jobs[job_id] = status
        self._cancel_flags[job_id] = False

        # Launch execution worker in background daemon thread
        worker = threading.Thread(target=self._run_job_worker, args=(job_spec,), daemon=True)
        worker.start()
        return job_id

    def _run_job_worker(self, job_spec: JobSpecification) -> None:
        job_id = job_spec.job_id
        status = self._jobs[job_id]
        status.status = ExecutionStatus.RUNNING
        start_time = time.time()

        settings = get_settings()
        out_dir = job_spec.output_dir or str(settings.outputs_dir / job_id)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        context = ExecutionContext(
            job_id=job_id,
            input_path=job_spec.video_path,
            output_dir=out_dir,
            morphology_name=job_spec.morphology,
            target_fps=job_spec.target_fps,
            config_overrides=job_spec.config_overrides,
        )

        # Assemble pipeline stages
        pipeline = Pipeline(name=f"LocalPipeline_{job_id}")
        pipeline.add_stage(VideoIngestionStage())
        pipeline.add_stage(PoseEstimationStage())
        pipeline.add_stage(MotionProcessingStage())
        pipeline.add_stage(SkeletonReconstructionStage())
        pipeline.add_stage(AnimationGenerationStage())

        # Progress event listener
        def on_event(event: PipelineEvent) -> None:
            if self._cancel_flags.get(job_id, False):
                raise InterruptedError("Job cancelled by user.")
            if event.stage_name:
                status.current_stage = event.stage_name
            status.progress = max(status.progress, event.progress)
            status.message = event.message

        pipeline.event_bus.subscribe_all(on_event)

        try:
            context = pipeline.execute(context)
            exec_time = time.time() - start_time
            status.status = ExecutionStatus.COMPLETED
            status.progress = 1.0
            status.end_time = time.time()
            status.message = f"Reconstruction completed successfully in {exec_time:.2f}s."

            self._results[job_id] = JobResult(
                job_id=job_id,
                success=True,
                status=ExecutionStatus.COMPLETED,
                execution_time_sec=round(exec_time, 2),
                output_assets=context.exported_assets,
                metrics={
                    "frames_processed": len(context.frame_sequence) if context.frame_sequence else 0,
                    "joints_reconstructed": len(context.skeleton.joints) if context.skeleton else 0,
                },
            )
        except Exception as e:
            exec_time = time.time() - start_time
            is_cancel = self._cancel_flags.get(job_id, False)
            status.status = ExecutionStatus.CANCELLED if is_cancel else ExecutionStatus.FAILED
            status.end_time = time.time()
            status.error = str(e)
            status.message = f"Job aborted: {e}"

            self._results[job_id] = JobResult(
                job_id=job_id,
                success=False,
                status=status.status,
                execution_time_sec=round(exec_time, 2),
                error_message=str(e),
            )

    def get_job_status(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            return JobStatus(job_id=job_id, status=ExecutionStatus.FAILED, message="Job not found.")
        return self._jobs[job_id]

    def retrieve_result(self, job_id: str) -> JobResult:
        if job_id not in self._results:
            status = self.get_job_status(job_id)
            return JobResult(
                job_id=job_id,
                success=False,
                status=status.status,
                execution_time_sec=0.0,
                error_message="Job result not ready or job not found.",
            )
        return self._results[job_id]

    def cancel_job(self, job_id: str) -> bool:
        if job_id in self._jobs and self._jobs[job_id].status == ExecutionStatus.RUNNING:
            self._cancel_flags[job_id] = True
            logger.info(f"Cancellation requested for job {job_id}")
            return True
        return False
