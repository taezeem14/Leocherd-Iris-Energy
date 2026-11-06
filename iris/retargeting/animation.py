"""
Animation timeseries representation and keyframe generation stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import MotionError
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, PipelineStage
from iris.rigging.skeleton import Skeleton

logger = get_logger("iris.retargeting.animation")


@dataclass
class SkeletalAnimation:
    """Stores continuous skeletal animation keyframes (rotations & root motion)."""

    name: str
    fps: float
    timestamps: np.ndarray  # Shape: (T,)
    root_joint_name: str
    root_translations: np.ndarray  # Shape: (T, 3)
    # Bone local rotations as unit quaternions [w, x, y, z] for each bone/joint
    rotations: Dict[str, np.ndarray] = field(default_factory=dict)  # (T, 4)

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)

    @property
    def duration_sec(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


class AnimationGenerationStage(PipelineStage):
    """Pipeline stage evaluating skeletal local rotations and root motion across frames."""

    def __init__(self, name: str = "AnimationGenerationStage") -> None:
        super().__init__(name)

    def run(self, context: ExecutionContext, event_bus: Optional[EventBus] = None) -> ExecutionContext:
        if not context.skeleton:
            raise MotionError("No skeleton available in ExecutionContext.")
        if not context.pose_sequence:
            raise MotionError("No pose_sequence available in ExecutionContext.")

        skel: Skeleton = context.skeleton
        poses = context.pose_sequence
        T = len(poses)
        timestamps = np.array([p.timestamp_sec for p in poses], dtype=np.float32)

        fps = 1.0 / max(1e-4, (timestamps[-1] - timestamps[0]) / max(1, T - 1)) if T > 1 else 30.0
        logger.info(f"Generating skeletal animation for {T} frames @ {fps:.1f} FPS...")

        root_translations = np.zeros((T, 3), dtype=np.float32)
        joint_rotations: Dict[str, np.ndarray] = {
            j_name: np.zeros((T, 4), dtype=np.float32) for j_name in skel.joints.keys()
        }

        for t, pose in enumerate(poses):
            # Update skeleton pose with kinematic length preservation
            skel.update_from_pose(pose, enforce_bone_lengths=True)

            # Record root translation
            root_node = skel.joints[skel.root_name]
            root_translations[t] = root_node.world_pos

            # Record local rotations for all joints
            for j_name, node in skel.joints.items():
                joint_rotations[j_name][t] = node.local_rotation

        # Reset skeleton to canonical rest pose for downstream asset exporters
        skel.reset_to_rest()

        anim = SkeletalAnimation(
            name=f"Anim_{skel.name}",
            fps=fps,
            timestamps=timestamps,
            root_joint_name=skel.root_name,
            root_translations=root_translations,
            rotations=joint_rotations,
        )

        context.animation = anim
        logger.info(f"Animation generation complete: {anim.frame_count} frames, {anim.duration_sec:.2f}s.")

        if event_bus:
            event_bus.emit(
                PipelineEvent(
                    event_type=EventType.STAGE_COMPLETED,
                    stage_name=self.name,
                    progress=1.0,
                    message=f"Generated {anim.frame_count} animation keyframes.",
                    payload={"duration_sec": anim.duration_sec, "fps": fps},
                )
            )

        return context
