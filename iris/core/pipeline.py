"""
Central pipeline orchestrator and stage lifecycle manager.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import IrisError
from iris.core.logging import get_logger

logger = get_logger("iris.pipeline")


@dataclass
class StageMetrics:
    stage_name: str
    duration_sec: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Shared state passed between pipeline stages."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_path: Optional[str] = None
    output_dir: Optional[str] = None
    morphology_name: str = "human"
    target_fps: int = 15
    skinning_method: str = "heat_diffusion"
    retarget_to: Optional[str] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Intermediate stage artifacts
    video_metadata: Optional[Any] = None
    frame_sequence: Optional[Any] = None
    detections: Optional[Any] = None
    pose_sequence: Optional[Any] = None
    motion_sequence: Optional[Any] = None
    morphology: Optional[Any] = None
    morphology_prediction: Optional[Any] = None
    skeleton: Optional[Any] = None
    discovered_skeleton: Optional[Any] = None
    animation: Optional[Any] = None
    exported_assets: Dict[str, str] = field(default_factory=dict)

    # Runtime Telemetry
    stage_metrics: List[StageMetrics] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


class PipelineStage(ABC):
    """Abstract base class for all discrete pipeline stages."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, context: ExecutionContext, event_bus: Optional[EventBus] = None) -> ExecutionContext:
        """Executes the stage logic."""
        pass


class Pipeline:
    """Configurable pipeline runner coordinating stages and event dispatch."""

    def __init__(self, name: str = "IrisReconstructionPipeline") -> None:
        self.name = name
        self.stages: List[PipelineStage] = []
        self.event_bus = EventBus()

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        self.stages.append(stage)
        return self

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        logger.info(f"[{context.job_id}] Initiating pipeline '{self.name}' with {len(self.stages)} stages.")
        self.event_bus.emit(
            PipelineEvent(
                event_type=EventType.PIPELINE_STARTED,
                message=f"Starting pipeline {self.name}",
                payload={"job_id": context.job_id, "stages": [s.name for s in self.stages]},
            )
        )

        for idx, stage in enumerate(self.stages):
            progress_ratio = idx / len(self.stages)
            logger.info(f"[{context.job_id}] Executing stage {idx+1}/{len(self.stages)}: {stage.name}")
            self.event_bus.emit(
                PipelineEvent(
                    event_type=EventType.STAGE_STARTED,
                    stage_name=stage.name,
                    progress=progress_ratio,
                    message=f"Starting {stage.name}",
                )
            )

            stage_start = time.perf_counter()
            try:
                context = stage.run(context, self.event_bus)
                duration = time.perf_counter() - stage_start
                context.stage_metrics.append(
                    StageMetrics(stage_name=stage.name, duration_sec=duration, success=True)
                )
                self.event_bus.emit(
                    PipelineEvent(
                        event_type=EventType.STAGE_COMPLETED,
                        stage_name=stage.name,
                        progress=(idx + 1) / len(self.stages),
                        message=f"Completed {stage.name} in {duration:.3f}s",
                        payload={"duration_sec": duration},
                    )
                )
            except Exception as e:
                duration = time.perf_counter() - stage_start
                error_msg = str(e)
                context.stage_metrics.append(
                    StageMetrics(
                        stage_name=stage.name,
                        duration_sec=duration,
                        success=False,
                        error_message=error_msg,
                    )
                )
                logger.error(f"[{context.job_id}] Stage '{stage.name}' failed: {error_msg}")
                self.event_bus.emit(
                    PipelineEvent(
                        event_type=EventType.STAGE_FAILED,
                        stage_name=stage.name,
                        message=f"Stage {stage.name} failed: {error_msg}",
                        payload={"error": error_msg},
                    )
                )
                self.event_bus.emit(
                    PipelineEvent(
                        event_type=EventType.PIPELINE_FAILED,
                        message=f"Pipeline aborted at {stage.name}",
                        payload={"stage": stage.name, "error": error_msg},
                    )
                )
                raise IrisError(f"Pipeline failed at stage '{stage.name}': {error_msg}") from e

        context.end_time = time.time()
        total_time = context.end_time - context.start_time
        logger.info(f"[{context.job_id}] Pipeline completed successfully in {total_time:.2f}s.")
        self.event_bus.emit(
            PipelineEvent(
                event_type=EventType.PIPELINE_COMPLETED,
                progress=1.0,
                message=f"Pipeline completed in {total_time:.2f}s",
                payload={"total_duration_sec": total_time},
            )
        )
        return context
