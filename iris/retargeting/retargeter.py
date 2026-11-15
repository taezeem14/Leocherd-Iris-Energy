"""
Cross-morphology retargeting engine, spatial scale normalization, and end-effector IK.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import MotionError
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, PipelineStage
from iris.morphology import MorphologyRegistry
from iris.motion.smoothing import FootLockingStabilizer, QuaternionSmoother
from iris.retargeting.animation import SkeletalAnimation
from iris.retargeting.mapping import RigMapping, compute_rest_pose_offsets
from iris.rigging.ik import FABRIKSolver, TwoBoneIKSolver
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton

logger = get_logger("iris.retargeting.retargeter")


class RetargetingEngine:
    """
    Cross-morphology retargeting engine transferring motion between arbitrary creatures.

    Features:
    - Proportion and spatial scale normalization based on creature characteristic height/length.
    - Root displacement trajectory scaling preserving gait cadence and ground clearance.
    - Rest-pose rotational offset alignment avoiding limb twist and inverted axes.
    - Closed-form TwoBoneIK and FABRIK constraint resolution for end-effectors.
    - Manifold S^3 Quaternion temporal smoothing and stance foot-locking integration.
    """

    @staticmethod
    def normalize_spatial_scale(source_skel: Skeleton, target_skel: Skeleton) -> float:
        """
        Calculates the global spatial scaling factor s = H_target / H_source.

        Uses vertical bounding height for upright/limbed morphologies, falling back to
        total spinal chain length or maximal bounding dimension for flat/horizontal creatures.
        """
        src_ys = [float(j.rest_world_pos[1]) for j in source_skel.joints.values()]
        tgt_ys = [float(j.rest_world_pos[1]) for j in target_skel.joints.values()]

        src_h = max(src_ys) - min(src_ys) if len(src_ys) > 0 else 1.0
        tgt_h = max(tgt_ys) - min(tgt_ys) if len(tgt_ys) > 0 else 1.0

        # Fallback to maximal span if vertical height is near zero (e.g. Serpentine resting on floor)
        if src_h < 0.15:
            src_pts = np.array([j.rest_world_pos for j in source_skel.joints.values()])
            src_spans = np.ptp(src_pts, axis=0) if len(src_pts) > 0 else np.array([1.0, 1.0, 1.0])
            src_h = float(np.max(src_spans))

        if tgt_h < 0.15:
            tgt_pts = np.array([j.rest_world_pos for j in target_skel.joints.values()])
            tgt_spans = np.ptp(tgt_pts, axis=0) if len(tgt_pts) > 0 else np.array([1.0, 1.0, 1.0])
            tgt_h = float(np.max(tgt_spans))

        src_h = max(1e-4, src_h)
        tgt_h = max(1e-4, tgt_h)

        scale = tgt_h / src_h
        return float(np.clip(scale, 0.05, 20.0))

    @staticmethod
    def scale_root_translation(
        source_trans: np.ndarray,
        source_skel: Skeleton,
        target_skel: Skeleton,
        scale: float,
    ) -> np.ndarray:
        """
        Scales root trajectory by spatial ratio while anchoring to target rest root elevation.

        Target translation preserves relative stride displacement and vertical bounce:
            X_tgt(t) = X_tgt,rest + s * (X_src(t) - X_src(0))
            Y_tgt(t) = Y_tgt,rest + s * (Y_src(t) - Y_src,rest)
            Z_tgt(t) = Z_tgt,rest + s * (Z_src(t) - Z_src(0))
        """
        T = source_trans.shape[0]
        if T == 0:
            return np.zeros((0, 3), dtype=np.float32)

        tgt_root = target_skel.joints[target_skel.root_name].rest_world_pos
        src_root = source_skel.joints[source_skel.root_name].rest_world_pos

        target_trans = np.zeros_like(source_trans, dtype=np.float32)

        # Horizontal displacements relative to starting frame
        dx = (source_trans[:, 0] - source_trans[0, 0]) * scale
        dz = (source_trans[:, 2] - source_trans[0, 2]) * scale

        # Vertical displacement relative to source rest root
        dy = (source_trans[:, 1] - src_root[1]) * scale

        target_trans[:, 0] = tgt_root[0] + dx
        target_trans[:, 1] = tgt_root[1] + dy
        target_trans[:, 2] = tgt_root[2] + dz

        return target_trans

    @staticmethod
    def retarget_rotations(
        source_anim: SkeletalAnimation,
        source_skel: Skeleton,
        target_skel: Skeleton,
        mapping: RigMapping,
    ) -> Dict[str, np.ndarray]:
        """
        Transfers bone rotations from source animation to target skeleton via rest-pose alignment.
        """
        T = source_anim.frame_count
        target_rotations: Dict[str, np.ndarray] = {
            j_name: np.tile(QuaternionMath.identity(), (T, 1)).astype(np.float32)
            for j_name in target_skel.joints.keys()
        }

        for src_name, tgt_name in mapping.joint_map.items():
            if src_name not in source_anim.rotations or tgt_name not in target_skel.joints:
                continue

            q_off = mapping.get_offset(src_name)
            src_rots = source_anim.rotations[src_name]

            for t in range(T):
                q_src = src_rots[t]
                # Combine rest-pose offset with animated local rotation
                q_aligned = QuaternionMath.multiply(q_off, q_src)
                target_rotations[tgt_name][t] = QuaternionMath.normalize_quat(q_aligned)

        return target_rotations

    @staticmethod
    def apply_limb_ik(
        target_skel: Skeleton,
        target_rotations: Dict[str, np.ndarray],
        target_root_trans: np.ndarray,
        source_anim: SkeletalAnimation,
        source_skel: Skeleton,
        mapping: RigMapping,
        scale: float,
    ) -> None:
        """
        Adjusts limb joint rotations using TwoBoneIKSolver to maintain end-effector reach
        and prevent foot penetration or over-extension.
        """
        T = source_anim.frame_count
        if T == 0:
            return

        # Candidate 2-bone limb chains across morphologies: (Root, Mid, End, Pole Vector)
        limb_chains = [
            # Human / Avian
            ("LeftUpLeg", "LeftLeg", "LeftFoot", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("RightUpLeg", "RightLeg", "RightFoot", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("LeftArm", "LeftForeArm", "LeftHand", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
            ("RightArm", "RightForeArm", "RightHand", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
            # Avian
            ("LeftThigh", "LeftKnee", "LeftFoot", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("RightThigh", "RightKnee", "RightFoot", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("LeftWing_Base", "LeftWing_Mid", "LeftWing_Tip", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ("RightWing_Base", "RightWing_Mid", "RightWing_Tip", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            # Quadruped Hind Limbs
            ("BackLeft_Hip", "BackLeft_Knee", "BackLeft_Paw", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
            ("BackRight_Hip", "BackRight_Knee", "BackRight_Paw", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
            # Quadruped Front Limbs
            ("FrontLeft_Scapula", "FrontLeft_Knee", "FrontLeft_Paw", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("FrontRight_Scapula", "FrontRight_Knee", "FrontRight_Paw", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("FrontLeft_Knee", "FrontLeft_Ankle", "FrontLeft_Paw", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            ("FrontRight_Knee", "FrontRight_Ankle", "FrontRight_Paw", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ]

        active_chains = [
            chain
            for chain in limb_chains
            if chain[0] in target_skel.joints and chain[1] in target_skel.joints and chain[2] in target_skel.joints
        ]

        if not active_chains:
            return

        solver = TwoBoneIKSolver()

        for j_root, j_mid, j_end, pole_vec in active_chains:
            r_root = target_skel.joints[j_root].rest_world_pos
            r_mid = target_skel.joints[j_mid].rest_world_pos
            r_end = target_skel.joints[j_end].rest_world_pos

            l1 = float(np.linalg.norm(r_mid - r_root))
            l2 = float(np.linalg.norm(r_end - r_mid))
            if l1 <= 1e-4 or l2 <= 1e-4:
                continue

            for t in range(T):
                # World translation displacement of root
                d_root = target_root_trans[t] - target_root_trans[0]
                p_root_t = r_root + d_root

                # Target end-effector position
                p_target_t = r_end + d_root
                p_pole_t = p_root_t + pole_vec

                try:
                    res = solver.solve(
                        p_root=p_root_t,
                        p_target=p_target_t,
                        p_pole=p_pole_t,
                        l1=l1,
                        l2=l2,
                        rest_positions=(r_root, r_mid, r_end),
                        pole_is_vector=False,
                    )
                    # Blend IK rotation with existing forward kinematics
                    target_rotations[j_root][t] = res.rotations[0]
                    target_rotations[j_mid][t] = res.rotations[1]
                except Exception:
                    continue

    @classmethod
    def retarget(
        cls,
        source_anim: SkeletalAnimation,
        source_skel: Skeleton,
        target_skel: Skeleton,
        mapping: Optional[RigMapping] = None,
        enable_ik: bool = True,
        enable_smoothing: bool = True,
        enable_foot_locking: bool = True,
    ) -> SkeletalAnimation:
        """
        Executes end-to-end cross-morphology retargeting from source to target skeleton.
        """
        if mapping is None:
            mapping = RigMapping.build_default_mapping(source_skel.name, target_skel.name)

        if not mapping.rotational_offsets:
            mapping.rotational_offsets = compute_rest_pose_offsets(
                source_skel, target_skel, mapping.joint_map
            )

        logger.info(
            f"Retargeting animation '{source_anim.name}' ({source_skel.name}) -> {target_skel.name}..."
        )

        # 1. Spatial scale normalization
        scale = cls.normalize_spatial_scale(source_skel, target_skel)
        logger.info(f"Computed spatial scale factor: {scale:.4f}")

        # 2. Root translation trajectory scaling
        target_root_trans = cls.scale_root_translation(
            source_anim.root_translations, source_skel, target_skel, scale
        )

        # 3. Rest-pose rotational offset alignment
        target_rotations = cls.retarget_rotations(
            source_anim, source_skel, target_skel, mapping
        )

        # 4. End-effector limb IK
        if enable_ik:
            cls.apply_limb_ik(
                target_skel,
                target_rotations,
                target_root_trans,
                source_anim,
                source_skel,
                mapping,
                scale,
            )

        # 5. Quaternion manifold S^3 smoothing
        if enable_smoothing:
            smoother = QuaternionSmoother(cutoff_hz=2.0)
            for j_name, rots in target_rotations.items():
                target_rotations[j_name] = smoother.smooth(rots, source_anim.timestamps)

        # 6. Foot-locking & stance stabilization
        if enable_foot_locking and target_root_trans.shape[0] > 3:
            stabilizer = FootLockingStabilizer()
            # Stance velocity check on root motion to suppress drift
            contacts = stabilizer.detect_contacts(
                target_root_trans, source_anim.timestamps, height_thresh=2.0, vel_thresh=0.05
            )
            # Apply subtle clamp if stationary
            if np.any(contacts):
                target_root_trans = stabilizer.stabilize_trajectory(
                    target_root_trans, source_anim.timestamps, contacts=contacts
                )

        retargeted_anim = SkeletalAnimation(
            name=f"Retargeted_{target_skel.name}",
            fps=source_anim.fps,
            timestamps=source_anim.timestamps.copy(),
            root_joint_name=target_skel.root_name,
            root_translations=target_root_trans,
            rotations=target_rotations,
        )

        logger.info(
            f"Retargeting successful: {retargeted_anim.frame_count} frames, {len(target_rotations)} target joints."
        )
        return retargeted_anim


class RetargetingStage(PipelineStage):
    """
    Pipeline stage applying cross-morphology retargeting when ExecutionContext.retarget_to is set.
    """

    def __init__(self, name: str = "RetargetingStage") -> None:
        super().__init__(name)

    def run(self, context: ExecutionContext, event_bus: Optional[EventBus] = None) -> ExecutionContext:
        if not context.retarget_to:
            logger.info("RetargetingStage: No target morphology specified (retarget_to is None). Skipping.")
            return context

        if not context.skeleton:
            raise MotionError("No skeleton available in ExecutionContext for retargeting.")
        if not context.animation:
            raise MotionError("No animation available in ExecutionContext for retargeting.")

        target_morph_name = str(context.retarget_to).lower()
        logger.info(f"RetargetingStage: Retargeting to target morphology '{target_morph_name}'...")

        target_morphology = MorphologyRegistry.get(target_morph_name)
        target_skeleton = Skeleton.from_morphology(target_morphology)

        retargeted_anim = RetargetingEngine.retarget(
            source_anim=context.animation,
            source_skel=context.skeleton,
            target_skel=target_skeleton,
        )

        # Update context artifacts with retargeted skeleton and animation
        context.skeleton = target_skeleton
        context.animation = retargeted_anim
        context.morphology = target_morphology
        context.morphology_name = target_morph_name

        if event_bus:
            event_bus.emit(
                PipelineEvent(
                    event_type=EventType.STAGE_COMPLETED,
                    stage_name=self.name,
                    progress=1.0,
                    message=f"Successfully retargeted animation to '{target_morph_name}'.",
                    payload={
                        "target_morphology": target_morph_name,
                        "joint_count": len(target_skeleton.joints),
                        "frame_count": retargeted_anim.frame_count,
                    },
                )
            )

        return context
