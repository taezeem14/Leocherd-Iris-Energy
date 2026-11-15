"""
Unit tests for Milestone 3: Cross-Morphology Retargeting, Spatial Normalization,
Quaternion Slerp Smoothing, and Foot-Locking Stabilization.
"""

import argparse
import math
from typing import Tuple
import numpy as np
import pytest

from iris.core.pipeline import ExecutionContext
from iris.morphology.biped import BipedMorphology
from iris.morphology.human import HumanMorphology
from iris.morphology.quadruped import QuadrupedMorphology
from iris.morphology.serpentine import SerpentineMorphology
from iris.morphology.tentacular import TentacularMorphology
from iris.motion.smoothing import (
    FootLockingStabilizer,
    QuaternionSmoother,
    quaternion_slerp,
)
from iris.retargeting.animation import SkeletalAnimation
from iris.retargeting.mapping import (
    RigMapping,
    build_default_mapping,
    compute_rest_pose_offsets,
    create_human_to_avian_mapping,
    create_human_to_quadruped_mapping,
    create_human_to_serpentine_mapping,
    create_human_to_tentacular_mapping,
    create_quadruped_to_human_mapping,
)
from iris.retargeting.retargeter import RetargetingEngine, RetargetingStage
from iris.rigging.ik import TwoBoneIKSolver
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton
from main import build_parser


# ==============================================================================
# Helper Functions
# ==============================================================================

def make_dummy_human_animation(num_frames: int = 15, fps: float = 30.0) -> Tuple[Skeleton, SkeletalAnimation]:
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)

    timestamps = np.linspace(0.0, (num_frames - 1) / fps, num_frames, dtype=np.float32)
    # Human walking forward along Z axis
    root_trans = np.zeros((num_frames, 3), dtype=np.float32)
    root_trans[:, 0] = 0.0
    root_trans[:, 1] = 1.0 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, num_frames))
    root_trans[:, 2] = np.linspace(0.0, 1.5, num_frames)

    rotations = {}
    for j_name in skel.joints.keys():
        rots = np.zeros((num_frames, 4), dtype=np.float32)
        rots[:, 0] = 1.0  # Identity default
        rotations[j_name] = rots

    # Add rhythmic pitch rotation on thighs and arms
    for t in range(num_frames):
        angle = 0.3 * math.sin(2.0 * math.pi * t / num_frames)
        q_swing = np.array([math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0], dtype=np.float32)
        rotations["LeftUpLeg"][t] = q_swing
        rotations["RightUpLeg"][t] = np.array([math.cos(-angle / 2.0), math.sin(-angle / 2.0), 0.0, 0.0], dtype=np.float32)
        rotations["LeftArm"][t] = np.array([math.cos(-angle / 2.0), 0.0, 0.0, math.sin(-angle / 2.0)], dtype=np.float32)
        rotations["RightArm"][t] = np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)], dtype=np.float32)

    anim = SkeletalAnimation(
        name="TestHumanWalk",
        fps=fps,
        timestamps=timestamps,
        root_joint_name="Hips",
        root_translations=root_trans,
        rotations=rotations,
    )
    return skel, anim


# ==============================================================================
# 1. Quaternion Slerp Mathematics
# ==============================================================================

def test_quaternion_slerp_interpolation():
    """Verify Slerp interpolates between two orthogonal unit quaternions on S^3."""
    q_start = QuaternionMath.identity()  # [1, 0, 0, 0]
    # 90 degrees around Z axis: [cos(pi/4), 0, 0, sin(pi/4)]
    half_angle = math.pi / 4.0
    q_end = np.array([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)], dtype=np.float32)

    q_0 = quaternion_slerp(q_start, q_end, 0.0)
    assert np.allclose(q_0, q_start, atol=1e-5)
    assert abs(np.linalg.norm(q_0) - 1.0) < 1e-6

    q_1 = quaternion_slerp(q_start, q_end, 1.0)
    assert np.allclose(q_1, q_end, atol=1e-5)
    assert abs(np.linalg.norm(q_1) - 1.0) < 1e-6

    # Midpoint t=0.5 must be 45 degrees around Z: [cos(pi/8), 0, 0, sin(pi/8)]
    q_half = quaternion_slerp(q_start, q_end, 0.5)
    expected_half = np.array([math.cos(math.pi / 8.0), 0.0, 0.0, math.sin(math.pi / 8.0)], dtype=np.float32)
    assert np.allclose(q_half, expected_half, atol=1e-5)
    assert abs(np.linalg.norm(q_half) - 1.0) < 1e-6


def test_quaternion_slerp_anti_flip():
    """Verify Slerp pivots antipodal quaternions (q and -q) to follow the shortest geodesic arc."""
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # -q1 represents the identical physical orientation
    q2 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Without antipodal flip, this would interpolate through 0; with pivot it must stay identity
    q_mid = quaternion_slerp(q1, q2, 0.5)
    assert abs(np.linalg.norm(q_mid) - 1.0) < 1e-5
    # The physical rotation must remain identity
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_rot = QuaternionMath.rotate_vector(q_mid, v)
    assert np.allclose(v_rot, v, atol=1e-5)


def test_quaternion_slerp_small_angle():
    """Verify Slerp small-angle Taylor expansion handles near-identical rotations without NaN/div-by-zero."""
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q2 = np.array([0.999999, 0.001, 0.0, 0.0], dtype=np.float32)
    q2 /= np.linalg.norm(q2)

    q_interp = quaternion_slerp(q1, q2, 0.5)
    assert not np.isnan(q_interp).any()
    assert abs(np.linalg.norm(q_interp) - 1.0) < 1e-6


# ==============================================================================
# 2. Quaternion Temporal Smoother
# ==============================================================================

def test_quaternion_smoother_noise_reduction():
    """Verify QuaternionSmoother significantly attenuates high-frequency angular jitter."""
    T = 60
    timestamps = np.linspace(0.0, 2.0, T, dtype=np.float32)
    smooth_angles = np.linspace(0.0, math.pi, T)

    np.random.seed(42)
    noisy_rotations = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        angle = smooth_angles[t] + np.random.normal(0.0, 0.15)
        q = np.array([math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0], dtype=np.float32)
        noisy_rotations[t] = q / np.linalg.norm(q)

    smoother = QuaternionSmoother(cutoff_hz=2.0)
    filtered = smoother.smooth(noisy_rotations, timestamps)

    # Compute angular acceleration (jerk proxy)
    noisy_diffs = np.linalg.norm(noisy_rotations[1:] - noisy_rotations[:-1], axis=1)
    filtered_diffs = np.linalg.norm(filtered[1:] - filtered[:-1], axis=1)

    mean_noisy_jerk = float(np.mean(np.abs(noisy_diffs[1:] - noisy_diffs[:-1])))
    mean_filtered_jerk = float(np.mean(np.abs(filtered_diffs[1:] - filtered_diffs[:-1])))

    # Smoothed signal should reduce high-frequency variation by at least 40%
    assert mean_filtered_jerk < mean_noisy_jerk * 0.60


def test_quaternion_smoother_unit_norm_preservation():
    """Verify that every smoothed quaternion strictly satisfies unit norm ||q|| = 1.0."""
    T = 30
    timestamps = np.linspace(0.0, 1.0, T, dtype=np.float32)
    rots = np.random.normal(size=(T, 4)).astype(np.float32)
    rots /= np.linalg.norm(rots, axis=1, keepdims=True)

    smoother = QuaternionSmoother(cutoff_hz=3.0)
    smoothed = smoother.smooth(rots, timestamps)

    norms = np.linalg.norm(smoothed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_quaternion_smoother_short_trajectory():
    """Verify boundary handling for short trajectories (T=1, T=2)."""
    smoother = QuaternionSmoother()
    rots_1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    time_1 = np.array([0.0], dtype=np.float32)
    res_1 = smoother.smooth(rots_1, time_1)
    assert res_1.shape == (1, 4)
    assert np.allclose(res_1[0], [1.0, 0.0, 0.0, 0.0])

    rots_2 = np.array([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]], dtype=np.float32)
    time_2 = np.array([0.0, 0.033], dtype=np.float32)
    res_2 = smoother.smooth(rots_2, time_2)
    assert res_2.shape == (2, 4)
    assert np.allclose(np.linalg.norm(res_2, axis=1), 1.0)


# ==============================================================================
# 3. Foot-Locking & Ground Contact Stabilization
# ==============================================================================

def test_foot_locking_stance_detection():
    """Verify FootLockingStabilizer accurately detects planted stance vs airborne swing."""
    T = 30
    timestamps = np.linspace(0.0, 1.0, T, dtype=np.float32)
    positions = np.zeros((T, 3), dtype=np.float32)

    # Frames 0-9: Swing phase (high Y, moving along Z)
    positions[0:10, 1] = 0.25
    positions[0:10, 2] = np.linspace(0.0, 0.5, 10)

    # Frames 10-20: Stance phase (at ground Y=0.02, low horizontal velocity)
    positions[10:21, 1] = 0.02
    positions[10:21, 2] = 0.55 + np.random.normal(0, 0.002, 11)  # tiny jitter

    # Frames 21-29: Swing phase (lift off)
    positions[21:, 1] = 0.30
    positions[21:, 2] = np.linspace(0.6, 1.0, 9)

    stabilizer = FootLockingStabilizer(height_thresh=0.08, vel_thresh=0.20, ground_height=0.0)
    contacts = stabilizer.detect_contacts(positions, timestamps)

    # Frames 10-20 must be detected as contact
    assert np.all(contacts[11:20])
    # Swing frames must be detected as non-contact
    assert not np.any(contacts[0:9])
    assert not np.any(contacts[22:])


def test_foot_locking_zero_velocity_clamping():
    """Verify stabilize_trajectory clamps stance positions to an exact zero-velocity anchor."""
    T = 30
    timestamps = np.linspace(0.0, 1.0, T, dtype=np.float32)
    positions = np.zeros((T, 3), dtype=np.float32)

    # Stance between frames 10 and 20
    positions[10:21, 1] = 0.03
    positions[10:21, 0] = 0.15 + np.linspace(-0.01, 0.01, 11)  # skating drift
    positions[10:21, 2] = 0.40 + np.linspace(-0.01, 0.01, 11)

    stabilizer = FootLockingStabilizer(height_thresh=0.08, vel_thresh=0.20, ground_height=0.0, blend_frames=2)
    stabilized = stabilizer.stabilize_trajectory(positions, timestamps)

    # In core stance, positions must be identical (zero velocity)
    stance_positions = stabilized[10:21]
    dx = np.ptp(stance_positions[:, 0])
    dz = np.ptp(stance_positions[:, 2])
    assert dx < 1e-5, f"Horizontal X drift in stance: {dx}"
    assert dz < 1e-5, f"Horizontal Z drift in stance: {dz}"
    # Ground height must be clamped to ground elevation
    assert np.allclose(stance_positions[:, 1], 0.0, atol=1e-5)


def test_foot_locking_cubic_smoothstep_transition():
    """Verify smooth continuous cubic transition without discontinuous position jumps at boundary."""
    T = 25
    timestamps = np.linspace(0.0, 1.0, T, dtype=np.float32)
    positions = np.zeros((T, 3), dtype=np.float32)
    positions[:, 1] = np.linspace(0.2, 0.0, T)
    positions[15:, 1] = 0.01

    stabilizer = FootLockingStabilizer(blend_frames=3, ground_height=0.0)
    stabilized = stabilizer.stabilize_trajectory(positions, timestamps)

    # Frame-to-frame displacement across the transition should be continuous and bounded
    displacements = np.linalg.norm(stabilized[1:] - stabilized[:-1], axis=1)
    max_disp = float(np.max(displacements))
    assert max_disp < 0.15, f"Discontinuous jump detected: {max_disp}"


# ==============================================================================
# 4. Topology Mappings & Rest-Pose Alignment
# ==============================================================================

def test_mapping_factories():
    """Verify all pre-configured morphology mappings construct valid correspondences."""
    m_h_q = create_human_to_quadruped_mapping()
    assert m_h_q.source_rig_name == "human"
    assert m_h_q.target_rig_name == "quadruped"
    assert m_h_q.map_joint("Hips") == "Pelvis"
    assert m_h_q.map_joint("LeftArm") == "FrontLeft_Knee"
    assert m_h_q.map_joint("LeftUpLeg") == "BackLeft_Hip"

    m_q_h = create_quadruped_to_human_mapping()
    assert m_q_h.map_joint("Pelvis") == "Hips"
    assert m_q_h.map_joint("FrontLeft_Knee") == "LeftArm"

    m_h_s = create_human_to_serpentine_mapping()
    assert m_h_s.map_joint("Hips") == "Vertebra_0"
    assert m_h_s.map_joint("Head") == "Vertebra_11"

    m_h_t = create_human_to_tentacular_mapping()
    assert m_h_t.map_joint("Hips") == "CentralBody"
    assert "Tentacle_0_Seg_0" in m_h_t.joint_map.values()

    m_h_a = create_human_to_avian_mapping()
    assert m_h_a.map_joint("LeftArm") == "LeftWing_Base"
    assert m_h_a.map_joint("LeftUpLeg") == "LeftThigh"


def test_build_default_mapping():
    """Verify build_default_mapping correctly routes arbitrary morphology names."""
    m1 = build_default_mapping("human", "quadruped")
    assert m1.target_rig_name == "quadruped"

    m2 = build_default_mapping("human", "serpentine")
    assert m2.target_rig_name == "serpentine"

    m3 = build_default_mapping("human", "avian")
    assert m3.target_rig_name == "avian"

    m_self = build_default_mapping("human", "human")
    assert m_self.source_rig_name == "human"


def test_rotational_rest_pose_offset_alignment():
    """Verify compute_rest_pose_offsets calculates valid rotation alignment between rest poses."""
    skel_h = Skeleton.from_morphology(HumanMorphology())
    skel_q = Skeleton.from_morphology(QuadrupedMorphology())
    mapping = create_human_to_quadruped_mapping()

    offsets = compute_rest_pose_offsets(skel_h, skel_q, mapping.joint_map)

    assert "Hips" in offsets
    assert "LeftArm" in offsets

    for j_name, q_off in offsets.items():
        assert q_off.shape == (4,)
        assert abs(np.linalg.norm(q_off) - 1.0) < 1e-5


# ==============================================================================
# 5. Spatial Scale Normalization & Trajectory Scaling
# ==============================================================================

def test_spatial_scale_normalization():
    """Verify RetargetingEngine correctly computes relative height/length scale factor."""
    skel_h = Skeleton.from_morphology(HumanMorphology())
    skel_q = Skeleton.from_morphology(QuadrupedMorphology())
    skel_s = Skeleton.from_morphology(SerpentineMorphology())

    scale_hq = RetargetingEngine.normalize_spatial_scale(skel_h, skel_q)
    # Human height ~1.72m, Quadruped height ~0.95m -> ratio ~0.55
    assert 0.45 < scale_hq < 0.70

    # Serpentine is flat on floor, should use horizontal span
    scale_hs = RetargetingEngine.normalize_spatial_scale(skel_h, skel_s)
    assert scale_hs > 0.05 and scale_hs < 5.0


def test_root_translation_scaling():
    """Verify scale_root_translation scales horizontal displacement while anchoring to target rest elevation."""
    skel_h = Skeleton.from_morphology(HumanMorphology())
    skel_q = Skeleton.from_morphology(QuadrupedMorphology())

    source_trans = np.zeros((10, 3), dtype=np.float32)
    source_trans[:, 1] = 1.0  # Human hips rest elevation
    source_trans[:, 2] = np.linspace(0.0, 2.0, 10)  # Walk 2 meters forward

    scale = 0.5
    target_trans = RetargetingEngine.scale_root_translation(source_trans, skel_h, skel_q, scale)

    # Quadruped pelvis rest elevation is 0.70m
    expected_pelvis_y = skel_q.joints["Pelvis"].rest_world_pos[1]
    assert np.allclose(target_trans[:, 1], expected_pelvis_y, atol=1e-5)

    # Total horizontal forward walk should be 2.0m * 0.5 = 1.0m
    total_z_disp = target_trans[-1, 2] - target_trans[0, 2]
    assert abs(total_z_disp - 1.0) < 1e-4


# ==============================================================================
# 6. End-to-End Cross-Morphology Retargeting
# ==============================================================================

def test_retarget_human_to_quadruped():
    """Verify full retargeting from Human animation to Quadruped skeleton."""
    skel_h, anim_h = make_dummy_human_animation(num_frames=12)
    skel_q = Skeleton.from_morphology(QuadrupedMorphology())

    retargeted = RetargetingEngine.retarget(anim_h, skel_h, skel_q)

    assert retargeted.root_joint_name == "Pelvis"
    assert retargeted.frame_count == 12
    assert len(retargeted.rotations) == len(skel_q.joints)

    # Check unit quaternion preservation on all quadruped joints
    for j_name, rots in retargeted.rotations.items():
        assert rots.shape == (12, 4)
        norms = np.linalg.norm(rots, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"Joint {j_name} not unit norm"


def test_retarget_quadruped_to_human():
    """Verify inverse retargeting from Quadruped animation to Human skeleton."""
    skel_q = Skeleton.from_morphology(QuadrupedMorphology())
    skel_h = Skeleton.from_morphology(HumanMorphology())

    T = 10
    anim_q = SkeletalAnimation(
        name="QuadTrot",
        fps=30.0,
        timestamps=np.linspace(0, 0.3, T, dtype=np.float32),
        root_joint_name="Pelvis",
        root_translations=np.zeros((T, 3), dtype=np.float32),
        rotations={j: np.tile(QuaternionMath.identity(), (T, 1)) for j in skel_q.joints.keys()},
    )

    retargeted = RetargetingEngine.retarget(anim_q, skel_q, skel_h)

    assert retargeted.root_joint_name == "Hips"
    assert retargeted.frame_count == T
    assert "Spine" in retargeted.rotations
    assert "LeftArm" in retargeted.rotations


def test_retarget_human_to_serpentine():
    """Verify retargeting from Humanoid to 12-vertebra Serpentine spine chain."""
    skel_h, anim_h = make_dummy_human_animation(num_frames=8)
    skel_s = Skeleton.from_morphology(SerpentineMorphology())

    retargeted = RetargetingEngine.retarget(anim_h, skel_h, skel_s)

    assert retargeted.root_joint_name == "Vertebra_0"
    assert retargeted.frame_count == 8
    assert "Vertebra_11" in retargeted.rotations


def test_retarget_human_to_tentacular():
    """Verify retargeting from Humanoid to radial Tentacular creature."""
    skel_h, anim_h = make_dummy_human_animation(num_frames=8)
    skel_t = Skeleton.from_morphology(TentacularMorphology())

    retargeted = RetargetingEngine.retarget(anim_h, skel_h, skel_t)

    assert retargeted.root_joint_name == "CentralBody"
    assert retargeted.frame_count == 8
    assert "Tentacle_0_Seg_0" in retargeted.rotations


def test_retarget_human_to_avian():
    """Verify retargeting from Humanoid to Avian/Biped creature."""
    skel_h, anim_h = make_dummy_human_animation(num_frames=8)
    skel_a = Skeleton.from_morphology(BipedMorphology(with_tail=True))

    retargeted = RetargetingEngine.retarget(anim_h, skel_h, skel_a)

    assert retargeted.frame_count == 8
    assert "LeftWing_Base" in retargeted.rotations
    assert "LeftThigh" in retargeted.rotations


# ==============================================================================
# 7. End-Effector IK Solver Integration
# ==============================================================================

def test_two_bone_ik_target_reach_and_length_preservation():
    """Verify TwoBoneIKSolver reaches target within 1mm without bone length distortion."""
    solver = TwoBoneIKSolver()
    p_root = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    p_target = np.array([0.0, 0.2, 0.2], dtype=np.float32)
    p_pole = np.array([0.0, 0.6, 1.0], dtype=np.float32)

    l1, l2 = 0.45, 0.45
    res = solver.solve(p_root, p_target, p_pole, l1=l1, l2=l2)

    p0, p1, p2 = res.positions
    # End position reaches target
    reach_error = float(np.linalg.norm(p2 - p_target))
    assert reach_error < 1e-3, f"IK reach error exceeded 1mm: {reach_error}"

    # Segment lengths strictly preserved
    seg1_len = float(np.linalg.norm(p1 - p0))
    seg2_len = float(np.linalg.norm(p2 - p1))
    assert abs(seg1_len - l1) < 1e-4
    assert abs(seg2_len - l2) < 1e-4


# ==============================================================================
# 8. Pipeline RetargetingStage Integration
# ==============================================================================

def test_retargeting_stage_pipeline_execution():
    """Verify RetargetingStage successfully transforms ExecutionContext when retarget_to is set."""
    skel_h, anim_h = make_dummy_human_animation(num_frames=5)
    context = ExecutionContext(
        morphology_name="human",
        retarget_to="quadruped",
    )
    context.skeleton = skel_h
    context.animation = anim_h

    stage = RetargetingStage()
    out_ctx = stage.run(context)

    assert out_ctx.morphology_name == "quadruped"
    assert out_ctx.skeleton.name == "quadruped"
    assert out_ctx.animation.root_joint_name == "Pelvis"
    assert len(out_ctx.animation.rotations) == len(out_ctx.skeleton.joints)


def test_retargeting_stage_noop_when_none():
    """Verify RetargetingStage acts as transparent no-op when retarget_to is None."""
    skel_h, anim_h = make_dummy_human_animation(num_frames=5)
    context = ExecutionContext(
        morphology_name="human",
        retarget_to=None,
    )
    context.skeleton = skel_h
    context.animation = anim_h

    stage = RetargetingStage()
    out_ctx = stage.run(context)

    assert out_ctx.morphology_name == "human"
    assert out_ctx.skeleton is skel_h
    assert out_ctx.animation is anim_h


# ==============================================================================
# 9. CLI Argument Parsing Verification
# ==============================================================================

def test_cli_retargeting_and_skinning_arguments():
    """Verify main.py CLI correctly parses --skinning-method and --retarget-to flags."""
    parser = build_parser()

    cmd = [
        "reconstruct",
        "dummy_video.mp4",
        "--morphology", "human",
        "--skinning-method", "heat_diffusion",
        "--retarget-to", "quadruped",
    ]
    args = parser.parse_args(cmd)

    assert args.skinning_method == "heat_diffusion"
    assert args.retarget_to == "quadruped"
    assert args.morphology == "human"
