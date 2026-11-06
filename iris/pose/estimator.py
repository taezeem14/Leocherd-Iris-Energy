"""
High-level pose estimation orchestrator and pipeline stage.
"""

from __future__ import annotations

from typing import List, Optional

from iris.core.config import get_settings
from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import PoseEstimationError
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, PipelineStage
from iris.models.adapters.kinematic_tracker import KinematicPoseEstimator
from iris.models.adapters.onnx_blazepose import ONNXBlazePoseEstimator
from iris.pose.base import BasePoseEstimator, PoseFrame
from iris.pose.confidence import ConfidenceFilter
from iris.pose.human import HumanPoseSynthesizer
from iris.vision.detection import SubjectDetector
from iris.vision.tracking import IoUTracker

logger = get_logger("iris.pose.estimator")


class PoseEstimatorFactory:
    """Instantiates the appropriate pose estimator depending on profile and availability."""

    @staticmethod
    def create(prefer_onnx: bool = True) -> BasePoseEstimator:
        if prefer_onnx:
            try:
                return ONNXBlazePoseEstimator()
            except Exception as e:
                logger.warning(f"Could not initialize ONNX BlazePose ({e}). Falling back to KinematicPoseEstimator.")
        return KinematicPoseEstimator()


class PoseEstimationStage(PipelineStage):
    """Pipeline stage executing subject tracking and 3D pose reconstruction across frames."""

    def __init__(self, name: str = "PoseEstimationStage", prefer_onnx: bool = True) -> None:
        super().__init__(name)
        self.estimator = PoseEstimatorFactory.create(prefer_onnx=prefer_onnx)
        self.detector = SubjectDetector()
        self.tracker = IoUTracker()
        self.filter = ConfidenceFilter()

    def run(self, context: ExecutionContext, event_bus: Optional[EventBus] = None) -> ExecutionContext:
        if not context.frame_sequence:
            raise PoseEstimationError("No extracted frames available in ExecutionContext.")

        frames = context.frame_sequence
        total_frames = len(frames)
        logger.info(f"Starting 3D pose estimation on {total_frames} frames using {self.estimator.name}...")

        pose_sequence: List[PoseFrame] = []

        for idx, frame in enumerate(frames):
            # 1. Detect & Track Subject
            detections = self.detector.detect(frame.rgb_image)
            active_tracks = self.tracker.update(detections)

            primary_track = active_tracks[0] if active_tracks else (1, detections[0] if detections else None)
            subject_id = primary_track[0]
            bbox = primary_track[1]
            bbox_tuple = (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax) if bbox else None

            # 2. Run Estimator
            raw_pose = self.estimator.estimate(
                rgb_frame=frame.rgb_image,
                frame_index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                subject_id=subject_id,
                bbox=bbox_tuple,
            )

            # 3. Synthesize Standard Rig Joints (if human/biped)
            if context.morphology_name in ("human", "biped"):
                rig_joints = HumanPoseSynthesizer.synthesize_rig_joints(raw_pose.joints)
                raw_pose.joints.update(rig_joints)

            pose_sequence.append(raw_pose)

            if event_bus and (idx % 10 == 0 or idx == total_frames - 1):
                event_bus.emit(
                    PipelineEvent(
                        event_type=EventType.STAGE_PROGRESS,
                        stage_name=self.name,
                        progress=(idx + 1) / total_frames,
                        message=f"Processed pose frame {idx+1}/{total_frames}",
                        payload={"frame_index": frame.index},
                    )
                )

        # 4. Filter and tag missing frames
        context.pose_sequence = self.filter.filter_sequence(pose_sequence)
        logger.info(f"Pose estimation complete. Reconstructed {len(context.pose_sequence)} pose frames.")
        return context
