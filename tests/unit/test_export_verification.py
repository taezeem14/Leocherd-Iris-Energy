"""
Unit tests for Toolchain Export Verification (Milestone 4).

Covers:
1. Generalized inverse bind matrices (IBMs) affine inversion & column-major order.
2. glTF 2.0 bindpose cancellation invariant (M_j * IBM_j ≈ I_4x4).
3. Strict glTF validator compliance and negative detection.
4. BVH gimbal lock singularity recovery and Euler temporal continuity phase unwrapping.
5. Unity-compatible Generic Rig profile conformance.
6. Blender offline manifoldness and DQS volume preservation verification.
7. Multi-morphology end-to-end export across all 6 creature types
   (Human, Quadruped, Serpentine, Avian, Tentacular, Branching Tree).
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import numpy as np
import pytest

from iris.export.bvh import BVHExporter, quat_sequence_to_euler_zxy_deg, quat_to_euler_zxy_deg
from iris.export.gltf import GltfExporter
from iris.export.unity import UnityRigProfile, validate_unity_generic_rig
from iris.morphology.human import HumanMorphology
from iris.morphology.quadruped import QuadrupedMorphology
from iris.morphology.serpentine import SerpentineMorphology
from iris.retargeting.animation import SkeletalAnimation
from iris.rigging.orientation import QuaternionMath
from iris.rigging.proxy_mesh import ProxyMeshGenerator
from iris.rigging.skeleton import Skeleton
from scripts.validate_gltf_strict import GltfStrictValidator, validate_gltf_strict
from scripts.verify_blender_rig import verify_blender_asset_offline
from tests.fixtures.creature_trajectories import (
    generate_avian_trajectory,
    generate_branching_tree_trajectory,
    generate_human_trajectory,
    generate_quadruped_trajectory,
    generate_serpentine_trajectory,
    generate_tentacular_trajectory,
)
from tests.fixtures.procedural_skeletons import (
    create_avian_skeleton,
    create_branching_tree_skeleton,
    create_human_skeleton,
    create_quadruped_skeleton,
    create_serpentine_skeleton,
    create_tentacular_skeleton,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def sample_human_anim():
    skel = create_human_skeleton()
    T = 15
    times = np.linspace(0.0, 1.0, T, dtype=np.float32)
    root_trans = np.zeros((T, 3), dtype=np.float32)
    root_trans[:, 1] = 0.95 + 0.05 * np.sin(2.0 * np.pi * times)
    rotations = {
        j: np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
        for j in skel.joints
    }
    # Add gentle limb swing
    for t in range(T):
        ang = 0.3 * np.sin(2.0 * np.pi * times[t])
        q_swing = np.array([np.cos(ang / 2.0), np.sin(ang / 2.0), 0.0, 0.0], dtype=np.float32)
        if "LeftArm" in rotations:
            rotations["LeftArm"][t] = q_swing
        if "RightArm" in rotations:
            rotations["RightArm"][t] = np.array([np.cos(-ang / 2.0), np.sin(-ang / 2.0), 0.0, 0.0], dtype=np.float32)

    anim = SkeletalAnimation(
        name="HumanWalk",
        fps=15.0,
        timestamps=times,
        root_joint_name="Hips",
        root_translations=root_trans,
        rotations=rotations,
    )
    return skel, anim


# =============================================================================
# 1. Generalized Inverse Bind Matrices (IBMs)
# =============================================================================

def test_gltf_inverse_bind_matrices_affine_inversion(tmp_path):
    """
    Verifies that GltfExporter constructs full 4x4 rest world affine transforms
    M_j = [R_j | T_j] using rest world rotation and translation, computes true inverse
    M_inv = np.linalg.inv(M), and serializes in glTF column-major order.
    """
    skel = create_human_skeleton()
    # Apply non-trivial rest world rotations to specific joints
    angle_rad = math.radians(45.0)
    q_rot = np.array([math.cos(angle_rad / 2.0), math.sin(angle_rad / 2.0), 0.0, 0.0], dtype=np.float32)
    skel.joints["LeftArm"].world_rotation = q_rot
    skel.joints["RightArm"].world_rotation = QuaternionMath.conjugate(q_rot)

    T = 5
    times = np.linspace(0.0, 0.5, T, dtype=np.float32)
    anim = SkeletalAnimation(
        name="Test",
        fps=10.0,
        timestamps=times,
        root_joint_name="Hips",
        root_translations=np.zeros((T, 3), dtype=np.float32),
        rotations={j: np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1)) for j in skel.joints},
    )

    out_file = tmp_path / "test_ibm.gltf"
    exporter = GltfExporter()
    exporter.export(skel, anim, out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Read IBM buffer
    skin = doc["skins"][0]
    ibm_acc = doc["accessors"][skin["inverseBindMatrices"]]
    assert ibm_acc["type"] == "MAT4"
    assert ibm_acc["componentType"] == 5126

    bv = doc["bufferViews"][ibm_acc["bufferView"]]
    b64_data = doc["buffers"][0]["uri"].split(",", 1)[1]
    raw_buf = base64.b64decode(b64_data)
    offset = bv["byteOffset"] + ibm_acc.get("byteOffset", 0)
    ibm_mats = np.frombuffer(raw_buf[offset : offset + ibm_acc["count"] * 64], dtype=np.float32).reshape((-1, 4, 4))

    joint_names = list(skel.joints.keys())
    for j_idx, j_name in enumerate(joint_names):
        node = skel.joints[j_name]
        # Construct expected M_j
        M_expected = np.eye(4, dtype=np.float32)
        if hasattr(node, "world_rotation") and node.world_rotation is not None:
            M_expected[:3, :3] = QuaternionMath.quat_to_matrix(node.world_rotation)
        M_expected[:3, 3] = node.rest_world_pos
        M_inv_expected = np.linalg.inv(M_expected)

        # glTF column-major matrix in raw memory is M_inv.T
        col_major_mat = ibm_mats[j_idx]
        assert np.allclose(col_major_mat, M_inv_expected.T, atol=1e-5), (
            f"Joint '{j_name}': IBM does not match expected column-major affine inverse"
        )


def test_gltf_ibm_bindpose_cancellation(tmp_path, sample_human_anim):
    """
    Verifies the mathematical invariant that for every joint j in exported glTF,
    the joint's global rest transform G_j multiplied by its inverse bind matrix IBM_j
    cancels to the 4x4 identity matrix: G_j * IBM_j == I_4x4.
    """
    skel, anim = sample_human_anim
    out_file = tmp_path / "test_cancel.gltf"
    GltfExporter().export(skel, anim, out_file)

    # Validate with strict glTF validator checking bindpose cancellation
    res = validate_gltf_strict(out_file, check_ibms=True)
    assert res.is_valid, f"Bindpose cancellation validation failed: {res.errors}"
    assert len(res.errors) == 0


# =============================================================================
# 2. Strict glTF 2.0 Validator Compliance & Negative Cases
# =============================================================================

def test_gltf_strict_validation_compliance(tmp_path, sample_human_anim):
    skel, anim = sample_human_anim
    out_file = tmp_path / "valid.gltf"
    GltfExporter().export(skel, anim, out_file)

    validator = GltfStrictValidator()
    res = validator.validate(out_file, check_ibms=True)
    assert res.is_valid
    assert res.stats["node_count"] == len(skel.joints) + 1
    assert res.stats["total_skin_joints"] == len(skel.joints)
    assert res.stats["total_vertices"] > 0
    assert res.stats["animation_count"] == 1


def test_gltf_strict_validator_detects_corrupted_timestamps(tmp_path, sample_human_anim):
    skel, anim = sample_human_anim
    out_file = tmp_path / "corrupt_time.gltf"
    GltfExporter().export(skel, anim, out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Find time accessor and corrupt timestamps to be non-monotonic
    time_acc_idx = doc["animations"][0]["samplers"][0]["input"]
    bv_idx = doc["accessors"][time_acc_idx]["bufferView"]
    bv = doc["bufferViews"][bv_idx]
    b64_data = doc["buffers"][0]["uri"].split(",", 1)[1]
    raw_buf = bytearray(base64.b64decode(b64_data))

    # Overwrite second timestamp to be smaller than first
    off = bv["byteOffset"]
    times = np.frombuffer(raw_buf[off : off + 5 * 4], dtype=np.float32).copy()
    times[1] = times[0] - 1.0  # non-monotonic
    raw_buf[off : off + 5 * 4] = times.tobytes()

    doc["buffers"][0]["uri"] = f"data:application/octet-stream;base64,{base64.b64encode(raw_buf).decode('ascii')}"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(doc, f)

    res = validate_gltf_strict(out_file)
    assert not res.is_valid
    assert any("strictly monotonically increasing" in e for e in res.errors)


def test_gltf_strict_validator_detects_weights_sum_deviation(tmp_path, sample_human_anim):
    skel, anim = sample_human_anim
    out_file = tmp_path / "bad_weights.gltf"
    GltfExporter().export(skel, anim, out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Find WEIGHTS_0 accessor
    w_acc_idx = doc["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"]
    bv_idx = doc["accessors"][w_acc_idx]["bufferView"]
    bv = doc["bufferViews"][bv_idx]
    b64_data = doc["buffers"][0]["uri"].split(",", 1)[1]
    raw_buf = bytearray(base64.b64decode(b64_data))

    off = bv["byteOffset"]
    # Corrupt first vertex weight sum to 0.5
    bad_weights = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    raw_buf[off : off + 16] = bad_weights.tobytes()

    doc["buffers"][0]["uri"] = f"data:application/octet-stream;base64,{base64.b64encode(raw_buf).decode('ascii')}"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(doc, f)

    res = validate_gltf_strict(out_file)
    assert not res.is_valid
    assert any("sum to 1.0" in e for e in res.errors)


# =============================================================================
# 3. BVH Euler Decomposition, Gimbal Lock & Phase Unwrapping
# =============================================================================

def test_bvh_gimbal_lock_singularity_recovery():
    """
    Verifies that quat_to_euler_zxy_deg recovers roll angle (rot_z) accurately
    at exact pitch gimbal lock singularities (alpha = +90° and alpha = -90°)
    instead of returning 0.0.
    """
    for expected_roll in [30.0, 45.0, 60.0, -45.0]:
        rz = math.radians(expected_roll)
        # 1. Pitch = +90 deg
        rx = math.radians(90.0)
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        R_pos = Rz @ Rx
        q_pos = QuaternionMath.matrix_to_quat(R_pos)

        rot_z, rot_x, rot_y = quat_to_euler_zxy_deg(q_pos)
        assert abs(rot_x - 90.0) < 1e-2, f"Pitch should be 90.0, got {rot_x}"
        assert abs(rot_z - expected_roll) < 1e-2, f"Expected roll {expected_roll}, got {rot_z}"
        assert abs(rot_y) < 1e-2, f"Yaw should be 0.0, got {rot_y}"

        # 2. Pitch = -90 deg
        rx_neg = math.radians(-90.0)
        Rx_neg = np.array([[1, 0, 0], [0, math.cos(rx_neg), -math.sin(rx_neg)], [0, math.sin(rx_neg), math.cos(rx_neg)]])
        R_neg = Rz @ Rx_neg
        q_neg = QuaternionMath.matrix_to_quat(R_neg)

        rot_z_neg, rot_x_neg, rot_y_neg = quat_to_euler_zxy_deg(q_neg)
        assert abs(rot_x_neg - (-90.0)) < 1e-2, f"Pitch should be -90.0, got {rot_x_neg}"
        assert abs(rot_z_neg - expected_roll) < 1e-2, f"Expected roll {expected_roll}, got {rot_z_neg}"
        assert abs(rot_y_neg) < 1e-2, f"Yaw should be 0.0, got {rot_y_neg}"


def test_bvh_euler_temporal_continuity_unwrapping():
    """
    Verifies that quat_sequence_to_euler_zxy_deg performs temporal phase unwrapping
    across 180° boundaries, eliminating 360° discontinuities.
    """
    # Create smooth continuous rotation from 160 deg to 200 deg around Z
    angles_deg = np.linspace(160.0, 200.0, 21)
    quats = np.zeros((len(angles_deg), 4), dtype=np.float32)
    for i, a in enumerate(angles_deg):
        rad = math.radians(a)
        quats[i] = [math.cos(rad / 2.0), 0.0, 0.0, math.sin(rad / 2.0)]

    eulers = quat_sequence_to_euler_zxy_deg(quats)
    unwrapped_z = eulers[:, 0]

    # Check maximum delta between adjacent frames is small (no 360 jump)
    diffs = np.abs(np.diff(unwrapped_z))
    assert np.all(diffs < 10.0), f"Detected phase jump in unwrapped trajectory: max diff = {np.max(diffs)}"
    # End angle should match ~200.0
    assert abs(unwrapped_z[-1] - 200.0) < 1.0


def test_bvh_antipodal_quaternion_flip_neutralization():
    """
    Verifies that quat_sequence_to_euler_zxy_deg neutralizes antipodal sign flips
    (q_t . q_{t-1} < 0) so the output Euler curve remains smooth.
    """
    quats = np.zeros((10, 4), dtype=np.float32)
    for i in range(10):
        rad = math.radians(i * 5.0)
        q = np.array([math.cos(rad / 2.0), 0.0, 0.0, math.sin(rad / 2.0)], dtype=np.float32)
        # Flip sign on odd frames
        if i % 2 == 1:
            q = -q
        quats[i] = q

    eulers = quat_sequence_to_euler_zxy_deg(quats)
    diffs = np.abs(np.diff(eulers[:, 0]))
    assert np.all(diffs < 10.0), f"Antipodal flip caused Euler jump: {diffs}"


# =============================================================================
# 4. Unity Generic Rig Profile Conformance
# =============================================================================

def test_unity_generic_rig_profile_conformance(tmp_path, sample_human_anim):
    skel, anim = sample_human_anim
    out_file = tmp_path / "unity_valid.gltf"
    GltfExporter().export(skel, anim, out_file)

    res = validate_unity_generic_rig(out_file)
    assert res.is_valid, f"Unity generic rig validation failed: {res.errors}"
    assert res.stats["root_joint"] == "Hips"
    assert res.stats["joint_count"] == len(skel.joints)


def test_unity_profile_detects_non_uniform_scale(tmp_path, sample_human_anim):
    skel, anim = sample_human_anim
    out_file = tmp_path / "non_uniform_scale.gltf"
    GltfExporter().export(skel, anim, out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Introduce non-uniform scale on a bone node
    doc["nodes"][2]["scale"] = [1.0, 2.5, 1.0]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(doc, f)

    res = validate_unity_generic_rig(out_file)
    assert not res.is_valid
    assert any("non-uniform scale" in e for e in res.errors)


# =============================================================================
# 5. Blender Offline Manifoldness & DQS Volume Preservation
# =============================================================================

def test_blender_offline_manifoldness_and_volume_preservation(tmp_path, sample_human_anim):
    skel, anim = sample_human_anim
    verts, faces, weights, j_indices = ProxyMeshGenerator.create_procedural_mesh(skel)
    out_file = tmp_path / "blender_test.gltf"
    GltfExporter().export(skel, anim, out_file, mesh=(verts, faces, weights, j_indices))

    res = verify_blender_asset_offline(out_file)
    assert res["passed"], f"Blender offline verification failed: {res['errors']}"
    assert res["zero_area_faces"] == 0
    assert res["non_manifold_edges"] == 0
    assert res["vertex_count"] > 0
    assert res["bone_count"] == len(skel.joints)
    assert res["preserve_volume"] is True


def test_blender_offline_detects_non_manifold_geometry(tmp_path, sample_human_anim):
    skel, anim = sample_human_anim
    verts, faces, weights, j_indices = ProxyMeshGenerator.create_procedural_mesh(skel)

    # Intentionally duplicate a face to create a non-manifold edge (shared by 3 faces)
    faces_corrupted = np.vstack([faces, faces[0:1]])
    out_file = tmp_path / "corrupted_mesh.gltf"
    GltfExporter().export(skel, anim, out_file, mesh=(verts, faces_corrupted, weights, j_indices))

    res = verify_blender_asset_offline(out_file)
    assert not res["passed"]
    assert res["non_manifold_edges"] > 0
    assert any("non-manifold edges" in e for e in res["errors"])


# =============================================================================
# 6. Multi-Morphology Export Across All Creature Types
# =============================================================================

@pytest.mark.parametrize(
    "morph_name,skel_factory,traj_factory",
    [
        ("human", create_human_skeleton, generate_human_trajectory),
        ("quadruped", create_quadruped_skeleton, generate_quadruped_trajectory),
        ("serpentine", create_serpentine_skeleton, generate_serpentine_trajectory),
        ("avian", create_avian_skeleton, generate_avian_trajectory),
        ("tentacular", create_tentacular_skeleton, generate_tentacular_trajectory),
        ("branching_tree", create_branching_tree_skeleton, generate_branching_tree_trajectory),
    ],
)
def test_multi_morphology_export_all_creatures(tmp_path, morph_name, skel_factory, traj_factory):
    """
    Tests full toolchain export (glTF and BVH) and verification across all 6 morphologies:
    Human, Quadruped, Serpentine, Avian, Tentacular, and Branching Tree.
    """
    skel = skel_factory()
    traj = traj_factory(frames=10, fps=15.0)

    # Create SkeletalAnimation from trajectory
    T = traj.shape[0]
    times = np.linspace(0.0, (T - 1) / 15.0, T, dtype=np.float32)
    root_idx = traj.joint_names.index(skel.root_name) if skel.root_name in traj.joint_names else 0
    root_trans = traj[:, root_idx, :]
    rotations = {
        j: np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
        for j in skel.joints
    }

    anim = SkeletalAnimation(
        name=f"Anim_{morph_name}",
        fps=15.0,
        timestamps=times,
        root_joint_name=skel.root_name,
        root_translations=root_trans,
        rotations=rotations,
    )

    # 1. Export procedural watertight proxy mesh
    verts, faces, weights, j_indices = ProxyMeshGenerator.create_procedural_mesh(skel)

    gltf_file = tmp_path / f"{morph_name}.gltf"
    GltfExporter().export(skel, anim, gltf_file, mesh=(verts, faces, weights, j_indices))
    assert gltf_file.exists()

    # 2. Strict glTF validation
    gltf_res = validate_gltf_strict(gltf_file, check_ibms=True)
    assert gltf_res.is_valid, f"{morph_name} glTF strict validation failed: {gltf_res.errors}"

    # 3. Unity Generic Rig validation
    unity_res = validate_unity_generic_rig(gltf_file)
    assert unity_res.is_valid, f"{morph_name} Unity rig validation failed: {unity_res.errors}"

    # 4. Blender offline manifoldness & geometry validation
    blender_res = verify_blender_asset_offline(gltf_file)
    assert blender_res["passed"], f"{morph_name} Blender offline validation failed: {blender_res['errors']}"
    assert blender_res["zero_area_faces"] == 0
    assert blender_res["non_manifold_edges"] == 0

    # 5. Export BVH and verify file integrity
    bvh_file = tmp_path / f"{morph_name}.bvh"
    BVHExporter().export(skel, anim, bvh_file)
    assert bvh_file.exists()

    content = bvh_file.read_text(encoding="utf-8")
    assert "HIERARCHY" in content
    assert f"ROOT {skel.root_name}" in content
    assert "MOTION" in content
    assert f"Frames: {T}" in content
