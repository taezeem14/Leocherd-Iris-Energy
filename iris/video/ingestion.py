"""
Video ingestion coordinator and pipeline stage.
"""

from __future__ import annotations

import cv2
from pathlib import Path
from typing import List, Optional

from iris.core.config import get_settings
from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import VideoProcessingError
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, PipelineStage
from iris.video.extraction import ExtractedFrame, FrameExtractor
from iris.video.metadata import VideoMetadata
from iris.video.sampling import FrameSampler
from iris.video.validation import VideoValidator

logger = get_logger("iris.video.ingestion")


class VideoIngestionService:
    """High-level service managing video validation, metadata probing, and extraction."""

    def __init__(self, max_dimension: Optional[int] = None) -> None:
        settings = get_settings()
        self.max_dimension = max_dimension or settings.max_frame_dimension
        self.extractor = FrameExtractor(max_dimension=self.max_dimension)

    def probe(self, video_path: str) -> VideoMetadata:
        valid, issues = VideoValidator.validate_file(video_path)
        if not valid:
            raise VideoProcessingError(f"Video validation failed: {'; '.join(issues)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Failed to open video container: {video_path}")

        try:
            metadata = VideoMetadata.from_opencv(video_path, cap)
            return metadata
        finally:
            cap.release()

    def process(
        self,
        video_path: str,
        target_fps: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> tuple[VideoMetadata, List[ExtractedFrame]]:
        metadata = self.probe(video_path)
        settings = get_settings()
        desired_fps = target_fps or settings.default_sample_fps

        schedule = FrameSampler.calculate_sampling_indices(
            total_frames=metadata.frame_count,
            source_fps=metadata.fps,
            target_fps=desired_fps,
        )

        frames = self.extractor.extract_all(
            video_path=video_path,
            sample_schedule=schedule,
            max_frames=max_frames,
        )
        return metadata, frames


class VideoIngestionStage(PipelineStage):
    """Pipeline stage executing video ingestion and populating context."""

    def __init__(self, name: str = "VideoIngestionStage") -> None:
        super().__init__(name)
        self.service = VideoIngestionService()

    def run(self, context: ExecutionContext, event_bus: Optional[EventBus] = None) -> ExecutionContext:
        if not context.input_path:
            raise VideoProcessingError("No input_path provided in ExecutionContext.")

        logger.info(f"Ingesting video from: {context.input_path}")
        metadata, frames = self.service.process(
            video_path=context.input_path,
            target_fps=context.target_fps,
            max_frames=context.config_overrides.get("max_frames"),
        )

        context.video_metadata = metadata
        context.frame_sequence = frames

        logger.info(f"Successfully extracted {len(frames)} frames ({metadata.summary()})")
        if event_bus:
            event_bus.emit(
                PipelineEvent(
                    event_type=EventType.STAGE_PROGRESS,
                    stage_name=self.name,
                    progress=1.0,
                    message=f"Extracted {len(frames)} frames",
                    payload={"frames_count": len(frames), "metadata": metadata.model_dump()},
                )
            )

        return context
