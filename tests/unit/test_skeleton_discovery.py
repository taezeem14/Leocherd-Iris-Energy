"""
Unit tests for the Unsupervised Automatic Skeleton & Kinematic Hierarchy Discovery Engine.
"""

import math
from typing import Dict, List
import numpy as np
import pytest

from iris.core.exceptions import SkeletonError
from iris.pose.base import Joint3D, PoseFrame
from iris.rigging.joints import BoneEdge, JointNode
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton
from iris.rigging.skeleton_discovery import (
    AutomaticSkeletonDiscovery,
    discover_from_pose_sequence,
)


def _generate_3link_pendulum(num_frames: int = 80) -> Dict[str, np.ndarray]:
    """
    Generates synthetic motion trajectory for a 3-link articulated arm / pendulum:
    Base (anchor) -> Elbow (L1=1.0) -> Wrist (L2=0.8) -> Tip (L3=0.6)
    """
    t = np.linspace(0.0, 4.0 * np.pi, num_frames, dtype=np.float32)
    # Joint angles over time (azimuth theta and elevation phi)
    theta1 = 0.6 * np.sin(1.2 * t)
    theta2 = 0.9 * np.cos(1.8 * t)
    theta3 = 0.5 * np.sin(2.4 * t + 0.4)

    phi1 = 0.3 * np.sin(0.8 * t)
    phi2 = 0.25 * np.cos(1.1 * t)
    phi3 = 0.2 * np.sin(1.5 * t)

    l1, l2, l3 = 1.0, 0.8, 0.6

    base = np.zeros((num_frames, 3), dtype=np.float32)

    dir1 = np.column_stack([
        np.cos(phi1) * np.sin(theta1),
        -np.cos(phi1) * np.cos(theta1),
        np.sin(phi1),
    ])
    elbow = base + l1 * dir1

    dir2 = np.column_stack([
        np.cos(phi2) * np.sin(theta1 + theta2),
        -np.cos(phi2) * np.cos(theta1 + theta2),
        np.sin(phi2),
    ])
    wrist = elbow + l2 * dir2

    dir3 = np.column_stack([
        np.cos(phi3) * np.sin(theta1 + theta2 + theta3),
        -np.cos(phi3) * np.cos(theta1 + theta2 + theta3),
        np.sin(phi3),
    ])
    tip = wrist + l3 * dir3

    return {
        "Base": base,
        "Elbow": elbow,
        "Wrist": wrist,
        "Tip": tip,
    }


def _generate_branching_tree(num_frames: int = 80) -> Dict[str, np.ndarray]:
    """
    Generates synthetic motion trajectory for a branching skeletal structure:
               Spine (Root)
             /      |     \
     LeftShoulder  Neck  RightShoulder
          |         |         |
      LeftHand    Head    RightHand
    """
    t = np.linspace(0.0, 4.0 * np.pi, num_frames, dtype=np.float32)
    spine = np.zeros((num_frames, 3), dtype=np.float32)
    # Spine drifts slightly
    spine[:, 0] = 0.05 * np.sin(0.5 * t)
    spine[:, 1] = 0.05 * np.cos(0.5 * t)

    # Neck and Head
    neck_len, head_len = 0.35, 0.25
    neck_pitch = 0.2 * np.sin(1.5 * t)
    neck = spine + np.column_stack([
        np.zeros(num_frames),
        neck_len * np.cos(neck_pitch),
        neck_len * np.sin(neck_pitch),
    ])
    head_yaw = 0.3 * np.cos(1.2 * t)
    head = neck + np.column_stack([
        head_len * np.sin(head_yaw),
        head_len * np.cos(head_yaw),
        np.zeros(num_frames),
    ])

    # Left Arm
    l_sh_len, l_hand_len = 0.45, 0.55
    l_arm_angle = 0.7 * np.sin(2.0 * t)
    l_shoulder = spine + np.column_stack([
        -l_sh_len * np.ones(num_frames),
        0.1 * np.sin(t),
        np.zeros(num_frames),
    ])
    l_hand = l_shoulder + np.column_stack([
        -l_hand_len * np.cos(l_arm_angle),
        -l_hand_len * np.sin(l_arm_angle),
        0.1 * np.cos(t),
    ])

    # Right Arm (different frequency/phase)
    r_sh_len, r_hand_len = 0.45, 0.55
    r_arm_angle = 0.8 * np.cos(1.7 * t + 0.5)
    r_shoulder = spine + np.column_stack([
        r_sh_len * np.ones(num_frames),
        0.1 * np.cos(t),
        np.zeros(num_frames),
    ])
    r_hand = r_shoulder + np.column_stack([
        r_hand_len * np.cos(r_arm_angle),
        -r_hand_len * np.sin(r_arm_angle),
        -0.1 * np.sin(t),
    ])

    return {
        "Spine": spine,
        "Neck": neck,
        "Head": head,
        "LeftShoulder": l_shoulder,
        "LeftHand": l_hand,
        "RightShoulder": r_shoulder,
        "RightHand": r_hand,
    }


def _generate_quadruped_motion(num_frames: int = 100) -> Dict[str, np.ndarray]:
    """
    Generates synthetic motion trajectory for a quadruped with 11 joints:
    Pelvis, Chest, Head, and 4 legs (Hip -> Foot).
    """
    t = np.linspace(0.0, 6.0 * np.pi, num_frames, dtype=np.float32)

    # Pelvis forward walk
    pelvis = np.zeros((num_frames, 3), dtype=np.float32)
    pelvis[:, 2] = 0.1 * t  # Walking forward +Z
    pelvis[:, 1] = 0.6 + 0.02 * np.sin(2.0 * t)  # Vertical bounce

    # Chest connected to pelvis
    spine_len = 0.5
    chest = pelvis.copy()
    chest[:, 2] += spine_len
    chest[:, 1] += 0.03 * np.cos(2.0 * t)

    # Head connected to chest
    neck_len = 0.3
    head = chest.copy()
    head[:, 2] += neck_len * 0.8
    head[:, 1] += neck_len * 0.6 + 0.02 * np.sin(1.5 * t)

    # Legs: FL, FR, BL, BR
    # Trotting gait: diagonal pairs (FL + BR) and (FR + BL)
    leg_len = 0.45

    def make_leg(anchor_pt: np.ndarray, phase: float) -> np.ndarray:
        swing = 0.3 * np.sin(2.0 * t + phase)
        foot = anchor_pt.copy()
        foot[:, 1] -= leg_len * np.cos(swing)
        foot[:, 2] += leg_len * np.sin(swing)
        return foot

    hip_fl = chest + np.array([-0.25, -0.1, 0.0], dtype=np.float32)
    foot_fl = make_leg(hip_fl, 0.0)

    hip_fr = chest + np.array([0.25, -0.1, 0.0], dtype=np.float32)
    foot_fr = make_leg(hip_fr, np.pi)

    hip_bl = pelvis + np.array([-0.22, -0.1, 0.0], dtype=np.float32)
    foot_bl = make_leg(hip_bl, np.pi)

    hip_br = pelvis + np.array([0.22, -0.1, 0.0], dtype=np.float32)
    foot_br = make_leg(hip_br, 0.0)

    return {
        "Pelvis": pelvis,
        "Chest": chest,
        "Head": head,
        "Hip_FL": hip_fl,
        "Foot_FL": foot_fl,
        "Hip_FR": hip_fr,
        "Foot_FR": foot_fr,
        "Hip_BL": hip_bl,
        "Foot_BL": foot_bl,
        "Hip_BR": hip_br,
        "Foot_BR": foot_br,
    }


def test_distance_variance_rigid_segments():
    """Verify that distance variance correctly identifies rigid bone segments vs articulating pairs."""
    data = _generate_3link_pendulum(num_frames=60)
    engine = AutomaticSkeletonDiscovery()
    coords, names = engine._parse_trajectory(data)
    mean_dist, var_dist = engine.compute_pairwise_distance_variance(coords)

    idx_base = names.index("Base")
    idx_elbow = names.index("Elbow")
    idx_wrist = names.index("Wrist")
    idx_tip = names.index("Tip")

    # Rigid connections must have near-zero distance variance
    assert var_dist[idx_base, idx_elbow] < 1e-5
    assert var_dist[idx_elbow, idx_wrist] < 1e-5
    assert var_dist[idx_wrist, idx_tip] < 1e-5

    # Articulating non-adjacent pairs must have significantly higher distance variance
    assert var_dist[idx_base, idx_wrist] > 1e-4
    assert var_dist[idx_elbow, idx_tip] > 1e-4
    assert var_dist[idx_base, idx_tip] > 1e-4


def test_3link_arm_kinematic_hierarchy_recovery():
    """Verify that the MST accurately recovers the 3-link arm hierarchy without templates."""
    data = _generate_3link_pendulum(num_frames=60)
    engine = AutomaticSkeletonDiscovery(lambda_param=100.0)
    skel = engine.discover(data, creature_name="3link_arm", root_name="Base")

    assert skel.name == "3link_arm"
    assert skel.root_name == "Base"
    assert len(skel.joints) == 4
    assert len(skel.bones) == 3

    # Check hierarchy connections
    assert skel.joints["Base"].parent_name is None
    assert "Elbow" in skel.joints["Base"].children_names

    assert skel.joints["Elbow"].parent_name == "Base"
    assert "Wrist" in skel.joints["Elbow"].children_names

    assert skel.joints["Wrist"].parent_name == "Elbow"
    assert "Tip" in skel.joints["Wrist"].children_names

    assert skel.joints["Tip"].parent_name == "Wrist"
    assert len(skel.joints["Tip"].children_names) == 0

    # Bone rest lengths should closely match original link lengths: 1.0, 0.8, 0.6
    assert math.isclose(skel.bones["Bone_Base_Elbow"].rest_length, 1.0, abs_tol=1e-3)
    assert math.isclose(skel.bones["Bone_Elbow_Wrist"].rest_length, 0.8, abs_tol=1e-3)
    assert math.isclose(skel.bones["Bone_Wrist_Tip"].rest_length, 0.6, abs_tol=1e-3)

    # Validates hierarchy without exceptions
    skel.validate_hierarchy()


def test_branching_tree_discovery():
    """Verify unsupervised discovery on a branching tree topology with multiple limb branches."""
    data = _generate_branching_tree(num_frames=80)
    engine = AutomaticSkeletonDiscovery(lambda_param=100.0)
    skel = engine.discover(data, creature_name="branching_creature")

    # Root should automatically be identified as Spine (closest to kinematic centroid)
    assert skel.root_name == "Spine"
    assert len(skel.joints) == 7
    assert len(skel.bones) == 6

    # Verify branch topology
    spine_children = set(skel.joints["Spine"].children_names)
    assert spine_children == {"Neck", "LeftShoulder", "RightShoulder"}

    assert skel.joints["Neck"].children_names == ["Head"]
    assert skel.joints["LeftShoulder"].children_names == ["LeftHand"]
    assert skel.joints["RightShoulder"].children_names == ["RightHand"]

    # All leaf joints have no children
    assert len(skel.joints["Head"].children_names) == 0
    assert len(skel.joints["LeftHand"].children_names) == 0
    assert len(skel.joints["RightHand"].children_names) == 0

    # Rest lengths must be positive and anatomically consistent
    for bone in skel.bones.values():
        assert bone.rest_length > 0.1

    skel.validate_hierarchy()


def test_quadruped_skeleton_discovery():
    """Verify discovery on quadruped motion recovering body trunk and 4 distinct limbs."""
    data = _generate_quadruped_motion(num_frames=100)
    engine = AutomaticSkeletonDiscovery(lambda_param=100.0)
    skel = engine.discover(data, creature_name="quadruped")

    assert len(skel.joints) == 11
    assert len(skel.bones) == 10

    # Pelvis or Chest is selected as core kinematic root
    assert skel.root_name in ["Pelvis", "Chest"]

    # Verify all 4 legs are properly connected to trunk
    for leg_hip, leg_foot in [
        ("Hip_FL", "Foot_FL"),
        ("Hip_FR", "Foot_FR"),
        ("Hip_BL", "Foot_BL"),
        ("Hip_BR", "Foot_BR"),
    ]:
        assert leg_hip in skel.joints
        assert leg_foot in skel.joints
        # Foot must be child of Hip
        assert skel.joints[leg_foot].parent_name == leg_hip

    # Bone validation
    skel.validate_hierarchy()


def test_discover_from_pose_sequence_helper():
    """Verify convenience helper discover_from_pose_sequence using PoseFrame input sequence."""
    data = _generate_branching_tree(num_frames=20)
    joint_names = list(data.keys())
    num_frames = len(data["Spine"])

    poses: List[PoseFrame] = []
    for t in range(num_frames):
        frame_joints: Dict[str, Joint3D] = {}
        for name in joint_names:
            pos = data[name][t]
            frame_joints[name] = Joint3D(
                name=name,
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
                confidence=1.0,
                visibility=1.0,
            )
        pose_frame = PoseFrame(
            frame_index=t,
            timestamp_sec=float(t * 0.033),
            joints=frame_joints,
        )
        poses.append(pose_frame)

    skel = discover_from_pose_sequence(poses, creature_name="pose_tree")

    assert isinstance(skel, Skeleton)
    assert len(skel.joints) == 7
    assert len(skel.bones) == 6
    assert skel.name == "pose_tree"
    skel.validate_hierarchy()


def test_noise_robustness():
    """Verify that discovery is robust against tracking noise on joint coordinates."""
    data = _generate_3link_pendulum(num_frames=80)
    # Inject Gaussian noise sigma=0.005 (0.5 cm noise on 1 m link)
    np.random.seed(42)
    noisy_data = {}
    for k, v in data.items():
        noisy_data[k] = v + np.random.normal(0.0, 0.005, size=v.shape).astype(np.float32)

    engine = AutomaticSkeletonDiscovery(lambda_param=100.0)
    skel = engine.discover(noisy_data, root_name="Base")

    assert len(skel.joints) == 4
    assert len(skel.bones) == 3

    # Exact chain is still cleanly recovered despite noise
    assert skel.joints["Base"].children_names == ["Elbow"]
    assert skel.joints["Elbow"].children_names == ["Wrist"]
    assert skel.joints["Wrist"].children_names == ["Tip"]

    skel.validate_hierarchy()


def test_numpy_3d_array_input():
    """Verify discovery with direct (T, N, 3) numpy array input and custom point names."""
    data = _generate_3link_pendulum(num_frames=40)
    point_names = ["Root", "Mid1", "Mid2", "End"]
    arr = np.stack([data["Base"], data["Elbow"], data["Wrist"], data["Tip"]], axis=1)

    skel = AutomaticSkeletonDiscovery.discover_skeleton(
        trajectory=arr,
        point_names=point_names,
        creature_name="array_arm",
        root_name="Root",
    )

    assert skel.name == "array_arm"
    assert skel.root_name == "Root"
    assert len(skel.joints) == 4
    assert len(skel.bones) == 3
    skel.validate_hierarchy()


def test_edge_case_single_joint():
    """Verify graceful handling of single isolated joint trajectory."""
    arr = np.zeros((10, 1, 3), dtype=np.float32)
    engine = AutomaticSkeletonDiscovery()
    skel = engine.discover(arr, point_names=["SingleJoint"])

    assert skel.root_name == "SingleJoint"
    assert len(skel.joints) == 1
    assert len(skel.bones) == 0
    skel.validate_hierarchy()


def test_error_handling():
    """Verify appropriate SkeletonError exceptions on invalid inputs."""
    engine = AutomaticSkeletonDiscovery()

    # Empty dictionary
    with pytest.raises(SkeletonError, match="Empty trajectory"):
        engine.discover({})

    # Mismatched point names count
    arr = np.zeros((10, 3, 3), dtype=np.float32)
    with pytest.raises(SkeletonError, match="Point names count"):
        engine.discover(arr, point_names=["J0", "J1"])

    # Invalid array dimensions
    with pytest.raises(SkeletonError, match="Expected trajectory array"):
        engine.discover(np.zeros((10, 3, 3, 3)))
