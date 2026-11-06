"""
Temporal motion reconstruction orchestrator and pipeline stage.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import MotionError
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, PipelineStage
from iris.motion.interpolation import TrajectoryInterpolator
from iris.motion.representation import JointTrajectory, MotionSequence
from iris.motion.smoothing import TemporalSmoother
from iris.pose.base import Joint3D, PoseFrame

logger = get_logger("iris.motion.reconstruction")


class MotionProcessingStage(PipelineStage):
    """Pipeline stage executing temporal smoothing, interpolation, and velocity calculations."""

    def __init__(self, name: str = "MotionProcessingStage", smoothing_method: str = "one_euro") -> None:
        super().__init__(name)
        self.smoothing_method = smoothing_method

    def run(self, context: ExecutionContext, event_bus: Optional[EventBus] = None) -> ExecutionContext:
        if not context.pose_sequence:
            raise MotionError("No pose_sequence available in ExecutionContext.")

        poses = context.pose_sequence
        T = len(poses)
        if T == 0:
            raise MotionError("Pose sequence is empty.")

        timestamps = np.array([p.timestamp_sec for p in poses], dtype=np.float64)
        all_joint_names = sorted(list(set().union(*(p.joints.keys() for p in poses))))

        logger.info(f"Reconstructing motion across {T} frames for {len(all_joint_names)} joints...")

        trajectories: Dict[str, JointTrajectory] = {}

        for j_name in all_joint_names:
            raw_pos = np.zeros((T, 3), dtype=np.float32)
            confs = np.zeros(T, dtype=np.float32)

            for t, p in enumerate(poses):
                if j_name in p.joints:
                    j = p.joints[j_name]
                    raw_pos[t] = j.to_numpy()
                    confs[t] = j.confidence

            # 1. Impute missing / occluded keypoints
            imputed = TrajectoryInterpolator.impute_missing(raw_pos, confs, timestamps)

            # 2. Smooth trajectory
            smoothed = TemporalSmoother.smooth_trajectory(imputed, timestamps, method=self.smoothing_method)

            # 3. Create trajectory and calculate derivatives
            traj = JointTrajectory(
                name=j_name,
                timestamps=timestamps,
                positions=smoothed,
                confidences=confs,
            )
            traj.compute_derivatives()
            trajectories[j_name] = traj

        # 4. Construct updated smoothed PoseFrame sequence
        fps = 1.0 / max(1e-4, (timestamps[-1] - timestamps[0]) / max(1, T - 1)) if T > 1 else 30.0
        motion_seq = MotionSequence(timestamps=timestamps, trajectories=trajectories, fps=fps)

        updated_poses: List[PoseFrame] = []
        for t in range(T):
            orig_pose = poses[t]
            new_joints: Dict[str, Joint3D] = {}
            for j_name, traj in trajectories.items():
                pos = traj.positions[t]
                conf = float(traj.confidences[t])
                new_joints[j_name] = Joint3D(
                    name=j_name,
                    x=round(float(pos[0]), 4),
                    y=round(float(pos[1]), 4),
                    z=round(float(pos[2]), 4),
                    confidence=round(conf, 3),
                    visibility=1.0,
                )
            updated_poses.append(
                PoseFrame(
                    frame_index=orig_pose.frame_index,
                    timestamp_sec=orig_pose.timestamp_sec,
                    subject_id=orig_pose.subject_id,
                    morphology=orig_pose.morphology,
                    joints=new_joints,
                    bounding_box=orig_pose.bounding_box,
                    is_interpolated=orig_pose.is_interpolated,
                )
            )

        context.motion_sequence = motion_seq
        context.pose_sequence = updated_poses

        logger.info(f"Temporal motion processing complete. Calculated velocities for {len(trajectories)} trajectories.")
        if event_bus:
            event_bus.emit(
                PipelineEvent(
                    event_type=EventType.STAGE_COMPLETED,
                    stage_name=self.name,
                    progress=1.0,
                    message="Motion smoothing and trajectory derivatives completed.",
                )
            )

        return context
