"""
Skeleton reconstruction coordinator and pipeline stage.
"""

from __future__ import annotations

from typing import Optional

from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import SkeletonError
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, PipelineStage
from iris.morphology import MorphologyRegistry
from iris.rigging.skeleton import Skeleton

logger = get_logger("iris.rigging.topology")


class SkeletonReconstructionStage(PipelineStage):
    """Pipeline stage building the 3D articulated skeleton from motion trajectories."""

    def __init__(self, name: str = "SkeletonReconstructionStage") -> None:
        super().__init__(name)

    def run(self, context: ExecutionContext, event_bus: Optional[EventBus] = None) -> ExecutionContext:
        if not context.pose_sequence:
            raise SkeletonError("No pose_sequence available in ExecutionContext.")

        morphology_name = (context.morphology_name or "human").lower()

        if morphology_name in ("discover", "unsupervised"):
            logger.info("Executing unsupervised automatic skeleton discovery...")
            from iris.rigging.skeleton_discovery import discover_from_pose_sequence
            skeleton = discover_from_pose_sequence(context.pose_sequence, creature_name="discovered_creature")
            context.discovered_skeleton = skeleton
        elif morphology_name == "auto":
            if context.morphology_prediction:
                inferred = context.morphology_prediction.predicted_type.value
                logger.info(f"Auto-selected morphology from prediction: '{inferred}'")
                morphology = MorphologyRegistry.get(inferred)
                context.morphology = morphology
                skeleton = Skeleton.from_morphology(morphology)
            else:
                from iris.morphology.classifier import MorphologyClassifier
                pred = MorphologyClassifier().classify_from_poses(context.pose_sequence)
                context.morphology_prediction = pred
                inferred = pred.predicted_type.value
                logger.info(f"Classified morphology as '{inferred}' with confidence {pred.confidence:.2f}")
                morphology = MorphologyRegistry.get(inferred)
                context.morphology = morphology
                skeleton = Skeleton.from_morphology(morphology)
        else:
            logger.info(f"Instantiating morphology plugin '{morphology_name}'...")
            morphology = MorphologyRegistry.get(morphology_name)
            context.morphology = morphology
            skeleton = Skeleton.from_morphology(morphology)

        context.skeleton = skeleton
        logger.info(
            f"Constructed skeleton '{skeleton.name}' with {len(skeleton.joints)} joints and {len(skeleton.bones)} bones."
        )

        if event_bus:
            event_bus.emit(
                PipelineEvent(
                    event_type=EventType.STAGE_COMPLETED,
                    stage_name=self.name,
                    progress=1.0,
                    message=f"Reconstructed skeleton hierarchy with {len(skeleton.joints)} joints.",
                    payload={"joints_count": len(skeleton.joints), "bones_count": len(skeleton.bones)},
                )
            )

        return context
