"""
Unit tests for Production Inverse Kinematics (IK) Engine.

Covers:
- TwoBoneIKSolver: analytical Law of Cosines, exact bone length preservation,
  millimeter target reach precision, pole vector bend-plane control, and reach clamping.
- FABRIKSolver: multi-joint reaching, fast convergence within 5 iterations,
  exact bone length preservation, angular cone constraint limits, and reach clamping.
- IKChain: Skeleton-integrated kinematic chain resolution for limbs and spines.
"""

import math
import numpy as np
import pytest

from iris.core.exceptions import SkeletonError
from iris.morphology.human import HumanMorphology
from iris.rigging.ik import (
    FABRIKResult,
    FABRIKSolver,
    IKChain,
    TwoBoneIKResult,
    TwoBoneIKSolver,
)
from iris.rigging.skeleton import Skeleton


# ==============================================================================
# TwoBoneIKSolver Unit Tests
# ==============================================================================


def test_two_bone_ik_arm_length_preservation_and_target_reach():
    """
    Verifies TwoBoneIKSolver on a synthetic humanoid arm (Shoulder -> ForeArm -> Hand):
    1. Exact length preservation of both upper arm and forearm bones (< 1e-5 m).
    2. End-effector reaches target within 1mm (< 1e-3 m).
    3. Output positions and rotations conform to expected interfaces.
    """
    l1 = 0.35  # Upper arm length
    l2 = 0.30  # Forearm length
    solver = TwoBoneIKSolver(l1=l1, l2=l2)

    p_root = np.array([0.0, 1.50, 0.0], dtype=np.float32)
    # Reachable target: distance is sqrt(0.35^2 + (-0.15)^2 + 0.15^2) = ~0.41 m < 0.65 m
    p_target = np.array([0.35, 1.35, 0.15], dtype=np.float32)
    # Pole vector pointing backward (elbow bends back in -Z)
    p_pole = np.array([0.20, 1.50, -0.60], dtype=np.float32)

    result = solver.solve(p_root=p_root, p_target=p_target, p_pole=p_pole)

    # Verify tuple unpacking and attribute access
    assert isinstance(result, TwoBoneIKResult)
    positions, rotations = result
    p_root_out, p_mid, p_end = positions
    q_root, q_mid, q_end = rotations

    # 1. Root position must be identical to input
    assert np.allclose(p_root_out, p_root, atol=1e-6)

    # 2. Exact length preservation of both bones
    len_bone1 = float(np.linalg.norm(p_mid - p_root_out))
    len_bone2 = float(np.linalg.norm(p_end - p_mid))
    assert math.isclose(len_bone1, l1, abs_tol=1e-5), f"Bone 1 length {len_bone1} != {l1}"
    assert math.isclose(len_bone2, l2, abs_tol=1e-5), f"Bone 2 length {len_bone2} != {l2}"

    # 3. End-effector reaches target within 1mm (0.001 m)
    reach_error = float(np.linalg.norm(p_end - p_target))
    assert reach_error < 1e-3, f"Reach error {reach_error} m exceeds 1mm tolerance"

    # 4. Quaternions are valid unit quaternions
    for q in (q_root, q_mid, q_end):
        assert math.isclose(float(np.linalg.norm(q)), 1.0, abs_tol=1e-5)


def test_two_bone_ik_leg_pole_vector_directs_bend_plane():
    """
    Verifies TwoBoneIKSolver on a synthetic leg (Hip -> Knee -> Foot):
    1. Exact length preservation of thigh and shin bones.
    2. End-effector reaches foot target within 1mm.
    3. Pole vector directs the bend plane:
       - Pole forward (+Z) -> knee bends strictly forward (+Z).
       - Pole backward (-Z) -> knee bends strictly backward (-Z).
       - Pole lateral (+X) -> knee bends strictly lateral (+X).
    """
    l1 = 0.45  # Thigh length
    l2 = 0.45  # Shin length
    solver = TwoBoneIKSolver(l1=l1, l2=l2)

    p_root = np.array([0.0, 1.00, 0.0], dtype=np.float32)
    p_target = np.array([0.0, 0.20, 0.0], dtype=np.float32)  # Distance = 0.80 m < 0.90 m

    # Case A: Pole placed forward (+Z) -> knee must bend forward (+Z)
    pole_fwd = np.array([0.0, 0.60, 1.00], dtype=np.float32)
    res_fwd = solver.solve(p_root, p_target, pole_fwd)
    assert res_fwd.p_mid[2] > 0.10, "Knee did not bend forward toward +Z pole"
    assert math.isclose(float(np.linalg.norm(res_fwd.p_mid - p_root)), l1, abs_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(res_fwd.p_end - res_fwd.p_mid)), l2, abs_tol=1e-5)
    assert float(np.linalg.norm(res_fwd.p_end - p_target)) < 1e-3

    # Case B: Pole placed backward (-Z) -> knee must bend backward (-Z)
    pole_back = np.array([0.0, 0.60, -1.00], dtype=np.float32)
    res_back = solver.solve(p_root, p_target, pole_back)
    assert res_back.p_mid[2] < -0.10, "Knee did not bend backward toward -Z pole"
    assert math.isclose(float(np.linalg.norm(res_back.p_mid - p_root)), l1, abs_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(res_back.p_end - res_back.p_mid)), l2, abs_tol=1e-5)
    assert float(np.linalg.norm(res_back.p_end - p_target)) < 1e-3

    # Case C: Pole placed laterally (+X) -> knee must bend laterally (+X)
    pole_side = np.array([1.00, 0.60, 0.00], dtype=np.float32)
    res_side = solver.solve(p_root, p_target, pole_side)
    assert res_side.p_mid[0] > 0.10, "Knee did not bend laterally toward +X pole"
    assert math.isclose(float(np.linalg.norm(res_side.p_mid - p_root)), l1, abs_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(res_side.p_end - res_side.p_mid)), l2, abs_tol=1e-5)
    assert float(np.linalg.norm(res_side.p_end - p_target)) < 1e-3


def test_two_bone_ik_reach_clamping():
    """
    Verifies TwoBoneIKSolver gracefully handles over-extension and under-extension:
    1. Over-extended target is clamped to maximum reach (l1 + l2).
    2. Both bone lengths remain strictly preserved.
    3. Limb points straight along target ray without NaN or numerical instability.
    """
    l1 = 0.45
    l2 = 0.45
    max_reach = l1 + l2  # 0.90 m
    solver = TwoBoneIKSolver()

    p_root = np.array([0.0, 1.00, 0.0], dtype=np.float32)
    # Target 1.50 m away (> 0.90 m reach limit)
    p_target_unreachable = np.array([0.0, -0.50, 0.0], dtype=np.float32)
    p_pole = np.array([0.0, 0.50, 1.00], dtype=np.float32)

    res = solver.solve(p_root, p_target_unreachable, p_pole, l1=l1, l2=l2)

    # 1. Bone lengths strictly preserved
    assert math.isclose(float(np.linalg.norm(res.p_mid - res.p_root)), l1, abs_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(res.p_end - res.p_mid)), l2, abs_tol=1e-5)

    # 2. Total extension equals max_reach
    total_dist = float(np.linalg.norm(res.p_end - res.p_root))
    assert math.isclose(total_dist, max_reach, abs_tol=1e-4)

    # 3. End-effector points straight along ray (0, 1 - 0.90, 0) = (0, 0.10, 0)
    expected_end = np.array([0.0, 0.10, 0.0], dtype=np.float32)
    assert np.allclose(res.p_end, expected_end, atol=1e-4)


# ==============================================================================
# FABRIKSolver Unit Tests
# ==============================================================================


def test_fabrik_5_bone_tentacle_convergence_and_lengths():
    """
    Verifies FABRIKSolver on a 5-bone tentacle/spine kinematic chain (6 joints):
    1. Fast convergence to target within 5 iterations.
    2. End-effector reaches target within 1mm.
    3. Exact length preservation across all 5 segments (< 1e-5 m).
    4. Root joint position is firmly anchored.
    """
    # 6 joints initialized along +Y axis, segment lengths = 0.2 m each, total length = 1.0 m
    joints = np.array(
        [
            [0.0, 0.00, 0.0],
            [0.0, 0.20, 0.0],
            [0.0, 0.40, 0.0],
            [0.0, 0.60, 0.0],
            [0.0, 0.80, 0.0],
            [0.0, 1.00, 0.0],
        ],
        dtype=np.float32,
    )
    expected_segment_length = 0.20

    # Reachable 3D target: distance = sqrt(0.3^2 + 0.7^2 + 0.2^2) = ~0.787 m < 1.0 m
    target = np.array([0.30, 0.70, 0.20], dtype=np.float32)

    solver = FABRIKSolver(tol=1e-3, max_iterations=15)
    result = solver.solve(joints, target)

    # 1. Output type and convergence within 5 iterations
    assert isinstance(result, FABRIKResult)
    assert isinstance(result, np.ndarray)
    assert result.converged is True
    assert result.iterations <= 5, f"FABRIK required {result.iterations} iterations (expected <= 5)"

    # 2. End-effector reaches target within 1mm (0.001 m)
    reach_error = float(np.linalg.norm(result[-1] - target))
    assert reach_error < 1e-3, f"End-effector error {reach_error} m exceeds 1mm tolerance"

    # 3. Exact bone length preservation across all 5 segments
    for i in range(5):
        seg_len = float(np.linalg.norm(result[i + 1] - result[i]))
        assert math.isclose(
            seg_len,
            expected_segment_length,
            abs_tol=1e-5,
        ), f"Segment {i} length {seg_len} != {expected_segment_length}"

    # 4. Root position remains anchored
    assert np.allclose(result[0], joints[0], atol=1e-6)


def test_fabrik_angular_limits_prevent_sharp_folding():
    """
    Verifies FABRIKSolver per-joint angular constraint cones:
    1. Bending angle at every intermediate joint is strictly clamped to theta_max.
    2. Segment lengths remain preserved down to 1e-5 precision.
    3. Confirms that unconstrained FABRIK on the same acute target folds much sharper.
    """
    joints = np.array(
        [
            [0.0, 0.00, 0.0],
            [0.0, 0.20, 0.0],
            [0.0, 0.40, 0.0],
            [0.0, 0.60, 0.0],
            [0.0, 0.80, 0.0],
            [0.0, 1.00, 0.0],
        ],
        dtype=np.float32,
    )
    # Acute folding target close to root and offset sideways
    target = np.array([0.15, 0.35, 0.0], dtype=np.float32)

    theta_max = math.radians(30.0)  # 30 degrees maximum bend per joint (~0.5236 rad)
    solver = FABRIKSolver(tol=1e-3, max_iterations=25)

    # Solve with angular limits
    res_constrained = solver.solve(joints, target, angular_limits=theta_max)

    # 1. Exact segment lengths preserved
    for i in range(5):
        seg_len = float(np.linalg.norm(res_constrained[i + 1] - res_constrained[i]))
        assert math.isclose(seg_len, 0.20, abs_tol=1e-5), f"Segment {i} deformed to {seg_len}"

    # 2. Maximum bend angle at any joint must not exceed theta_max
    for i in range(1, 5):
        u_parent = (res_constrained[i] - res_constrained[i - 1]) / 0.20
        u_child = (res_constrained[i + 1] - res_constrained[i]) / 0.20
        dot = float(np.clip(np.dot(u_parent, u_child), -1.0, 1.0))
        angle = math.acos(dot)
        assert angle <= theta_max + 1e-4, f"Joint {i} bent {math.degrees(angle):.1f} deg > 30 deg"

    # 3. Compare against unconstrained solve
    res_unconstrained = solver.solve(joints, target, angular_limits=None)
    max_angle_unconstrained = 0.0
    for i in range(1, 5):
        u_parent = (res_unconstrained[i] - res_unconstrained[i - 1]) / 0.20
        u_child = (res_unconstrained[i + 1] - res_unconstrained[i]) / 0.20
        dot = float(np.clip(np.dot(u_parent, u_child), -1.0, 1.0))
        max_angle_unconstrained = max(max_angle_unconstrained, math.acos(dot))

    assert max_angle_unconstrained > theta_max, (
        f"Unconstrained solve bent at {math.degrees(max_angle_unconstrained):.1f} deg; "
        "angular limit test condition is valid."
    )


def test_fabrik_reach_clamping():
    """Verifies FABRIKSolver over-extension gracefully stretches straight to total chain length."""
    joints = np.array(
        [
            [0.0, 0.00, 0.0],
            [0.0, 0.20, 0.0],
            [0.0, 0.40, 0.0],
            [0.0, 0.60, 0.0],
            [0.0, 0.80, 0.0],
            [0.0, 1.00, 0.0],
        ],
        dtype=np.float32,
    )
    target_unreachable = np.array([0.0, 2.50, 0.0], dtype=np.float32)  # dist = 2.5 > 1.0

    solver = FABRIKSolver()
    res = solver.solve(joints, target_unreachable)

    assert res.converged is False
    assert math.isclose(float(np.linalg.norm(res[-1] - res[0])), 1.00, abs_tol=1e-5)
    for i in range(5):
        seg_len = float(np.linalg.norm(res[i + 1] - res[i]))
        assert math.isclose(seg_len, 0.20, abs_tol=1e-5)


# ==============================================================================
# IKChain Integration Tests
# ==============================================================================


def test_ik_chain_human_arm_two_bone():
    """Verifies IKChain binding to a real HumanMorphology skeleton using TwoBoneIKSolver."""
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)

    chain = IKChain(
        skeleton=skel,
        joint_names=["LeftArm", "LeftForeArm", "LeftHand"],
    )
    assert chain.solver_type == "two_bone"
    assert len(chain.bone_lengths) == 2

    # Rest position of LeftHand is [-0.75, 1.45, 0.0]
    initial_hand = skel.joints["LeftHand"].world_pos.copy()
    target = initial_hand + np.array([0.06, -0.08, 0.05], dtype=np.float32)
    pole = np.array([-0.40, 1.45, -0.50], dtype=np.float32)

    positions = chain.solve(target_pos=target, pole_pos=pole)

    # 1. Returned dictionary contains all chain joints
    assert set(positions.keys()) == {"LeftArm", "LeftForeArm", "LeftHand"}

    # 2. Skeleton joints are updated and within 1mm of target
    final_hand = skel.joints["LeftHand"].world_pos
    assert np.linalg.norm(final_hand - target) < 1e-3

    # 3. Bone lengths in skeleton match rest lengths
    arm_len = np.linalg.norm(skel.joints["LeftForeArm"].world_pos - skel.joints["LeftArm"].world_pos)
    forearm_len = np.linalg.norm(skel.joints["LeftHand"].world_pos - skel.joints["LeftForeArm"].world_pos)
    assert math.isclose(arm_len, chain.bone_lengths[0], abs_tol=1e-5)
    assert math.isclose(forearm_len, chain.bone_lengths[1], abs_tol=1e-5)

    # 4. Quaternions updated
    assert math.isclose(float(np.linalg.norm(skel.joints["LeftArm"].world_rotation)), 1.0, abs_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(skel.joints["LeftForeArm"].world_rotation)), 1.0, abs_tol=1e-5)


def test_ik_chain_human_spine_fabrik():
    """Verifies IKChain binding to a multi-joint spine chain using FABRIKSolver."""
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)

    spine_joints = ["Hips", "Spine", "Chest", "Neck", "Head"]
    chain = IKChain(
        skeleton=skel,
        joint_names=spine_joints,
        solver_type="fabrik",
    )
    assert chain.solver_type == "fabrik"
    assert len(chain.bone_lengths) == 4

    head_rest = skel.joints["Head"].world_pos.copy()
    target = head_rest + np.array([0.05, -0.04, 0.06], dtype=np.float32)

    positions = chain.solve(target_pos=target)

    # 1. Returned dictionary contains all chain joints
    assert set(positions.keys()) == set(spine_joints)

    # 2. Head position updated
    final_head = skel.joints["Head"].world_pos
    assert np.allclose(final_head, positions["Head"])

    # 3. Segment lengths match skeleton rest lengths
    for i in range(len(spine_joints) - 1):
        p1 = skel.joints[spine_joints[i]].world_pos
        p2 = skel.joints[spine_joints[i + 1]].world_pos
        seg_len = float(np.linalg.norm(p2 - p1))
        assert math.isclose(seg_len, chain.bone_lengths[i], abs_tol=1e-5)


def test_ik_chain_auto_discovery_and_errors():
    """Verifies IKChain automatic hierarchy discovery and error handling."""
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)

    # Auto-discover Left Leg hierarchy: LeftUpLeg -> LeftLeg -> LeftFoot
    chain = IKChain(skeleton=skel, root_joint="LeftUpLeg", end_effector="LeftFoot")
    assert chain.joint_names == ["LeftUpLeg", "LeftLeg", "LeftFoot"]
    assert chain.solver_type == "two_bone"

    # Invalid joint name
    with pytest.raises(SkeletonError, match="Joint 'InvalidJoint' does not exist"):
        IKChain(skeleton=skel, joint_names=["Hips", "InvalidJoint"])

    # No kinematic path between disconnected branches
    with pytest.raises(SkeletonError, match="No kinematic path"):
        IKChain(skeleton=skel, root_joint="LeftHand", end_effector="RightFoot")
