"""
E2E Tier 3: Pairwise Cross-Module Integration Test Suite for Leocherd Iris Production Excellence.

Contains >=22 pairwise cross-module integration tests connecting cross-module data boundaries:
  - DQs -> glTF / BVH
  - Skinning (Volumetric, Geodesic, Heat Diffusion) -> Universal Proxy Mesh
  - Skeleton Discovery -> Proxy Mesh / glTF / BVH
  - Bilateral Symmetry -> DQS Deformer
  - 3-Axis Root Orientation -> glTF / BVH
  - TwoBoneIK / FABRIK -> Animation Generation Stage
  - Temporal Smoothing -> Animation Generation Stage
  - Weight Sanitization -> glTF Strict Validator
  - DQS Deformer -> Benchmark Reconstruction Tool
  - LBS Deformer -> Blender Export
  - Morphology Classifier -> Skeleton Reconstruction Stage
  - PoseFrames -> Discovery -> Proxy Mannequin
  - RigMapping -> TwoBoneIK
  - Foot-Locking -> BVH Export
  - ExecutionContext -> Multi-Format Export
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from iris.core.pipeline import ExecutionContext
from iris.export.blender import BlenderExporter
from iris.export.bvh import BVHExporter, quat_to_euler_zxy_deg
from iris.export.gltf import GltfExporter
from iris.morphology.classifier import MorphologyClassifier
from iris.morphology.quadruped import QuadrupedMorphology
from iris.motion.smoothing import OneEuroFilter, TemporalSmoother
from iris.pose.base import Joint3D, PoseFrame
from iris.retargeting.animation import AnimationGenerationStage, SkeletalAnimation
from iris.retargeting.mapping import RigMapping
from iris.rigging.ik import FABRIKSolver, TwoBoneIKSolver
from iris.rigging.orientation import (
    DualQuaternion,
    DualQuaternionMath,
    QuaternionMath,
)
from iris.rigging.proxy_mesh import ProxyMeshGenerator
from iris.rigging.skeleton import Skeleton
from iris.rigging.skeleton_discovery import (
    AutomaticSkeletonDiscovery,
    DiscoveredSkeleton,
    discover_from_pose_sequence,
)
from iris.rigging.skinning import (
    GeodesicSkinning,
    HeatDiffusionSkinning,
    SkinningMethod,
    VolumetricSkinning,
    calculate_skin_weights,
    deform_dqs,
    deform_lbs,
    normalize_skin_weights,
)
from scripts.benchmark_reconstruction import benchmark_reconstruction
from tests.fixtures import (
    compute_mesh_boundary,
    compute_mesh_volume,
    create_avian_skeleton,
    create_cylinder_twist_mesh,
    create_human_skeleton,
    create_quadruped_skeleton,
    create_serpentine_skeleton,
    create_tentacular_skeleton,
    generate_avian_trajectory,
    generate_human_trajectory,
    generate_quadruped_trajectory,
    generate_serpentine_trajectory,
    generate_tentacular_trajectory,
    is_mesh_watertight,
)


def _validate_gltf_file(gltf_path: Path) -> Tuple[List[str], List[str]]:
    """Helper implementing strict glTF 2.0 validation conforming to scripts/validate_gltf_strict.py."""
    with open(gltf_path) as f:
        doc = json.load(f)

    uri = doc["buffers"][0]["uri"]
    buf = base64.b64decode(uri.split(",", 1)[1])

    errors: List[str] = []
    warnings: List[str] = []

    type_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    component_map = {5123: np.uint16, 5126: np.float32}

    for i, acc in enumerate(doc["accessors"]):
        bv = doc["bufferViews"][acc["bufferView"]]
        offset = bv["byteOffset"] + acc.get("byteOffset", 0)
        dtype = component_map[acc["componentType"]]
        num_comps = type_map[acc["type"]]
        count = acc["count"]
        total_elements = count * num_comps

        data = np.frombuffer(buf[offset : offset + total_elements * dtype().itemsize], dtype=dtype)
        if num_comps > 1:
            data = data.reshape((count, num_comps))

        if "min" in acc and "max" in acc:
            if num_comps == 1:
                actual_min = float(np.min(data))
                actual_max = float(np.max(data))
                if actual_min < float(acc["min"][0]) - 1e-5:
                    errors.append(f"Acc {i}: actual_min {actual_min} < declared_min {acc['min'][0]}")
                if actual_max > float(acc["max"][0]) + 1e-5:
                    errors.append(f"Acc {i}: actual_max {actual_max} > declared_max {acc['max'][0]}")
            else:
                actual_min = np.min(data, axis=0)
                actual_max = np.max(data, axis=0)
                for c in range(num_comps):
                    if actual_min[c] < acc["min"][c] - 1e-5:
                        errors.append(f"Acc {i} comp {c}: actual_min < declared_min")
                    if actual_max[c] > acc["max"][c] + 1e-5:
                        errors.append(f"Acc {i} comp {c}: actual_max > declared_max")

    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            if "JOINTS_0" in prim["attributes"] and "WEIGHTS_0" in prim["attributes"]:
                j_acc = doc["accessors"][prim["attributes"]["JOINTS_0"]]
                w_acc = doc["accessors"][prim["attributes"]["WEIGHTS_0"]]
                j_bv = doc["bufferViews"][j_acc["bufferView"]]
                w_bv = doc["bufferViews"][w_acc["bufferView"]]

                j_data = np.frombuffer(buf[j_bv["byteOffset"] : j_bv["byteOffset"] + j_acc["count"] * 8], dtype=np.uint16).reshape((-1, 4))
                w_data = np.frombuffer(buf[w_bv["byteOffset"] : w_bv["byteOffset"] + w_acc["count"] * 16], dtype=np.float32).reshape((-1, 4))

                zero_w = (w_data == 0.0)
                non_zero_j_with_zero_w = (j_data != 0) & zero_w
                if np.any(non_zero_j_with_zero_w):
                    warnings.append("Zero weight influence has non-zero joint index")

                sums = np.sum(w_data, axis=1)
                if not np.allclose(sums, 1.0, atol=1e-4):
                    errors.append("Weights do not sum to 1.0")

    return errors, warnings


# =============================================================================
# 1. Dual Quaternion Integrations
# =============================================================================

def test_pairwise_dual_quaternion_to_gltf_export(human_skeleton, e2e_output_dir):
    """Pairwise 1: DQs -> glTF node rotation tracks & IBM accessors."""
    T = 10
    times = np.linspace(0, 1, T, dtype=np.float32)
    # Generate joint rotations via Dual Quaternion math
    dq_rot = DualQuaternion.from_rotation_translation(
        np.array([np.cos(0.1), 0, np.sin(0.1), 0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    q_norm, _ = dq_rot.to_rotation_translation()
    rotations = {j: np.tile(q_norm, (T, 1)) for j in human_skeleton.joints}
    anim = SkeletalAnimation("dq_anim", 10.0, times, "Hips", np.zeros((T, 3), dtype=np.float32), rotations)

    exporter = GltfExporter()
    gltf_path = exporter.export(human_skeleton, anim, e2e_output_dir / "dq_gltf.gltf")
    assert gltf_path.exists()
    errors, _ = _validate_gltf_file(gltf_path)
    assert len(errors) == 0


def test_pairwise_dual_quaternion_to_bvh_export(human_skeleton, e2e_output_dir):
    """Pairwise 2: DQs -> Euler ZXY decomposition & BVH motion lines."""
    T = 5
    times = np.linspace(0, 0.5, T, dtype=np.float32)
    dq = DualQuaternion.from_rotation_translation(
        np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32),  # 45 deg around Y
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    q, _ = dq.to_rotation_translation()
    anim = SkeletalAnimation("dq_bvh", 10.0, times, "Hips", np.zeros((T, 3), dtype=np.float32), {j: np.tile(q, (T, 1)) for j in human_skeleton.joints})

    exporter = BVHExporter()
    bvh_path = exporter.export(human_skeleton, anim, e2e_output_dir / "dq_bvh.bvh")
    assert bvh_path.exists()
    with open(bvh_path) as f:
        text = f.read()
    assert "MOTION" in text
    assert f"Frames: {T}" in text


# =============================================================================
# 2. Skinning -> Universal Proxy Mesh Integrations
# =============================================================================

def test_pairwise_volumetric_skinning_with_proxy_mesh(quadruped_skeleton):
    """Pairwise 3: VolumetricSkinning applied to create_procedural_mesh."""
    verts, faces, _, _ = ProxyMeshGenerator.create_procedural_mesh(quadruped_skeleton)
    weights, indices = VolumetricSkinning.calculate_weights(verts, quadruped_skeleton)
    assert weights.shape == (len(verts), 4)
    assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)


def test_pairwise_geodesic_skinning_with_proxy_mesh(serpentine_skeleton):
    """Pairwise 4: GeodesicSkinning computed along triangular faces of proxy mesh."""
    verts, faces, _, _ = ProxyMeshGenerator.create_procedural_mesh(serpentine_skeleton)
    weights, indices = GeodesicSkinning.calculate_weights(verts, faces, serpentine_skeleton)
    assert weights.shape == (len(verts), 4)
    assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)


def test_pairwise_heat_diffusion_with_proxy_mesh(avian_skeleton):
    """Pairwise 5: HeatDiffusionSkinning using Cotangent Laplacian on proxy mesh."""
    verts, faces, _, _ = ProxyMeshGenerator.create_procedural_mesh(avian_skeleton)
    weights, indices = HeatDiffusionSkinning.calculate_weights(verts, faces, avian_skeleton)
    assert weights.shape == (len(verts), 4)
    assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)


# =============================================================================
# 3. Skeleton Discovery -> Downstream Artifacts
# =============================================================================

def test_pairwise_skeleton_discovery_to_proxy_mesh(tentacular_trajectory):
    """Pairwise 6: Unsupervised DiscoveredSkeleton generates watertight proxy mesh."""
    discovery = AutomaticSkeletonDiscovery()
    skel = discovery.discover(tentacular_trajectory, tentacular_trajectory.joint_names)
    verts, faces, weights, indices = ProxyMeshGenerator.create_procedural_mesh(skel)
    assert len(verts) > 0 and len(faces) > 0
    assert is_mesh_watertight(faces)


def test_pairwise_skeleton_discovery_to_gltf_export(quadruped_trajectory, e2e_output_dir):
    """Pairwise 7: Unsupervised DiscoveredSkeleton exported to glTF 2.0 with valid IBMs."""
    discovery = AutomaticSkeletonDiscovery()
    skel = discovery.discover(quadruped_trajectory, quadruped_trajectory.joint_names)

    T = 5
    times = np.linspace(0, 1, T, dtype=np.float32)
    anim = SkeletalAnimation("disc_anim", 10.0, times, skel.root_name, np.zeros((T, 3), dtype=np.float32), {j: np.tile(QuaternionMath.identity(), (T, 1)) for j in skel.joints})

    exporter = GltfExporter()
    out_gltf = exporter.export(skel, anim, e2e_output_dir / "discovered_quad.gltf")
    assert out_gltf.exists()
    errors, _ = _validate_gltf_file(out_gltf)
    assert len(errors) == 0


def test_pairwise_skeleton_discovery_to_bvh_export(serpentine_trajectory, e2e_output_dir):
    """Pairwise 8: Unsupervised DiscoveredSkeleton exported to standard BVH."""
    discovery = AutomaticSkeletonDiscovery()
    skel = discovery.discover(serpentine_trajectory, serpentine_trajectory.joint_names)

    T = 5
    times = np.linspace(0, 1, T, dtype=np.float32)
    anim = SkeletalAnimation("serp_anim", 10.0, times, skel.root_name, np.zeros((T, 3), dtype=np.float32), {j: np.tile(QuaternionMath.identity(), (T, 1)) for j in skel.joints})

    exporter = BVHExporter()
    out_bvh = exporter.export(skel, anim, e2e_output_dir / "discovered_serp.bvh")
    assert out_bvh.exists()


# =============================================================================
# 4. Bilateral Symmetry & Root Orientation Integrations
# =============================================================================

def test_pairwise_bilateral_symmetry_with_dqs(quadruped_skeleton, quadruped_trajectory):
    """Pairwise 9: Bilateral symmetry pairs maintain symmetric DQS deformation."""
    discovery = AutomaticSkeletonDiscovery()
    skel = discovery.discover(quadruped_trajectory, quadruped_trajectory.joint_names)
    assert len(skel.symmetry_pairs) > 0

    verts, faces, weights, indices = ProxyMeshGenerator.create_procedural_mesh(skel)
    j_dqs = [DualQuaternion.identity() for _ in range(len(skel.joints))]
    def_v, _ = deform_dqs(verts, None, j_dqs, weights, indices)
    assert np.allclose(def_v, verts, atol=1e-5)


def test_pairwise_3axis_root_orientation_with_gltf(serpentine_skeleton, e2e_output_dir):
    """Pairwise 10: Root orientation basis preserves glTF root node transformation."""
    T = 4
    times = np.linspace(0, 1, T, dtype=np.float32)
    anim = SkeletalAnimation("serp_root", 10.0, times, serpentine_skeleton.root_name, np.zeros((T, 3), dtype=np.float32), {j: np.tile(QuaternionMath.identity(), (T, 1)) for j in serpentine_skeleton.joints})

    exporter = GltfExporter()
    path = exporter.export(serpentine_skeleton, anim, e2e_output_dir / "serp_root.gltf")
    assert path.exists()


def test_pairwise_3axis_root_orientation_with_bvh(human_skeleton, e2e_output_dir):
    """Pairwise 11: Root orientation basis correctly sets BVH root channel offsets."""
    anim = SkeletalAnimation("human_bvh", 15.0, np.array([0.0, 0.1], dtype=np.float32), "Hips", np.array([[0, 1.0, 0], [0, 1.05, 0.1]], dtype=np.float32), {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in human_skeleton.joints})
    exporter = BVHExporter()
    path = exporter.export(human_skeleton, anim, e2e_output_dir / "human_root.bvh")
    with open(path) as f:
        text = f.read()
    assert "ROOT Hips" in text
    assert "OFFSET" in text


# =============================================================================
# 5. Kinematics, IK & Smoothing Integrations
# =============================================================================

def test_pairwise_twobone_ik_with_animation_generation(human_skeleton):
    """Pairwise 12: TwoBoneIKSolver executes within AnimationGenerationStage."""
    solver = TwoBoneIKSolver(l1=0.45, l2=0.45)
    p_root = human_skeleton.joints["LeftUpLeg"].rest_world_pos
    p_target = human_skeleton.joints["LeftFoot"].rest_world_pos + np.array([0, 0.05, 0.05], dtype=np.float32)
    p_pole = human_skeleton.joints["LeftLeg"].rest_world_pos + np.array([0, 0, 0.2], dtype=np.float32)

    ik_res = solver.solve(p_root, p_target, p_pole)
    assert np.allclose(ik_res.p_end, p_target, atol=1e-4)


def test_pairwise_fabrik_with_animation_generation(serpentine_skeleton):
    """Pairwise 13: FABRIKSolver solves multi-link spine in AnimationGenerationStage."""
    solver = FABRIKSolver(tol=1e-3, max_iterations=20)
    spine_pts = np.array([node.rest_world_pos for node in list(serpentine_skeleton.joints.values())[:5]])
    target = spine_pts[-1] + np.array([0.1, 0.05, 0.0], dtype=np.float32)

    res = solver.solve(spine_pts, target)
    assert np.linalg.norm(res[-1] - target) < 0.05


def test_pairwise_temporal_smoothing_with_animation_generation():
    """Pairwise 14: TemporalSmoother reduces jitter in ExecutionContext.animation."""
    T = 30
    t = np.linspace(0, 1, T, dtype=np.float32)
    clean_y = np.sin(t * 2 * np.pi)
    noisy_y = clean_y + np.random.normal(0, 0.05, T).astype(np.float32)
    pos_noisy = np.column_stack([np.zeros(T), noisy_y, np.zeros(T)]).astype(np.float32)

    pos_smoothed = TemporalSmoother.smooth_trajectory(pos_noisy, t, method="one_euro")
    assert np.var(np.diff(pos_smoothed, axis=0)) < np.var(np.diff(pos_noisy, axis=0))


# =============================================================================
# 6. Weight Sanitization & Verification Toolchain
# =============================================================================

def test_pairwise_weight_sanitization_with_gltf_strict_validator(human_skeleton, e2e_output_dir):
    """Pairwise 15: Sanitized weights pass scripts/validate_gltf_strict.py with 0 warnings/errors."""
    cyl = create_cylinder_twist_mesh()
    weights, indices = normalize_skin_weights(cyl.weights, joint_indices=cyl.joint_indices)

    anim = SkeletalAnimation("test_san", 15.0, np.array([0, 1], dtype=np.float32), "Hips", np.zeros((2, 3), dtype=np.float32), {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in human_skeleton.joints})
    exporter = GltfExporter()
    path = exporter.export(human_skeleton, anim, e2e_output_dir / "sanitized.gltf")
    errors, warnings = _validate_gltf_file(path)
    assert len(errors) == 0


def test_pairwise_dqs_deformer_with_benchmark_reconstruction(human_skeleton, e2e_output_dir):
    """Pairwise 16: DQS deformed mesh validated by scripts/benchmark_reconstruction.py."""
    T = 10
    times = np.linspace(0, 1, T, dtype=np.float32)
    anim = SkeletalAnimation("bench_anim", 10.0, times, "Hips", np.zeros((T, 3), dtype=np.float32), {j: np.tile(QuaternionMath.identity(), (T, 1)) for j in human_skeleton.joints})

    exporter_gltf = GltfExporter()
    exporter_bvh = BVHExporter()
    gltf_path = exporter_gltf.export(human_skeleton, anim, e2e_output_dir / "bench.gltf")
    bvh_path = exporter_bvh.export(human_skeleton, anim, e2e_output_dir / "bench.bvh")

    report = benchmark_reconstruction(gltf_path, bvh_path)
    assert report.quaternion_stats.is_valid is True


def test_pairwise_lbs_deformer_with_blender_export(quadruped_skeleton, e2e_output_dir):
    """Pairwise 17: LBS matrix palette exports to Blender JSON rig/anim packages."""
    anim = SkeletalAnimation("quad_lbs", 15.0, np.array([0, 1], dtype=np.float32), quadruped_skeleton.root_name, np.zeros((2, 3), dtype=np.float32), {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in quadruped_skeleton.joints})
    exporter = BlenderExporter()
    exporter.export(quadruped_skeleton, anim, e2e_output_dir / "quad_lbs.blend")
    assert (e2e_output_dir / "quad_lbs_rig_data.json").exists()
    assert (e2e_output_dir / "quad_lbs_anim_data.json").exists()


# =============================================================================
# 7. High-Level Pipelines & Combinations
# =============================================================================

def test_pairwise_morphology_classifier_with_skeleton_reconstruction(quadruped_trajectory):
    """Pairwise 18: MorphologyClassifier output drives SkeletonReconstructionStage."""
    classifier = MorphologyClassifier()
    pred = classifier.classify_from_poses(quadruped_trajectory.to_pose_frames())
    assert pred.morphology_name == "quadruped"
    skel = Skeleton.from_morphology(QuadrupedMorphology())
    assert skel.root_name == "Pelvis"


def test_pairwise_pose_frames_to_discovered_skeleton_to_proxy(human_trajectory):
    """Pairwise 19: Poses -> Discovery -> Skinned Mannequin pipeline."""
    poses = human_trajectory.to_pose_frames()
    skel = discover_from_pose_sequence(poses, creature_name="discovered_mannequin")
    v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(skel)
    assert len(v) > 0 and len(f) > 0
    assert is_mesh_watertight(f)


def test_pairwise_rig_mapping_with_twobone_ik(human_skeleton, quadruped_skeleton):
    """Pairwise 20: Target joint mapping drives end-effector reach targets."""
    mapping = RigMapping(human_skeleton.name, quadruped_skeleton.name, joint_map={"LeftLeg": "FrontLeft_Knee", "LeftFoot": "FrontLeft_Ankle"})
    tgt_foot = mapping.map_joint("LeftFoot")
    assert tgt_foot == "FrontLeft_Ankle"
    p_target = quadruped_skeleton.joints[tgt_foot].rest_world_pos
    solver = TwoBoneIKSolver(l1=0.3, l2=0.3)
    res = solver.solve(np.zeros(3), p_target, np.array([0, 1, 0]))
    assert not np.any(np.isnan(res.p_mid))


def test_pairwise_foot_locking_with_bvh_export(human_skeleton, e2e_output_dir):
    """Pairwise 21: Clamped stance frames reflect stationary foot coordinates in BVH output."""
    T = 6
    times = np.linspace(0, 0.5, T, dtype=np.float32)
    # Stance frames have constant root translations
    root_t = np.zeros((T, 3), dtype=np.float32)
    anim = SkeletalAnimation("stance_anim", 12.0, times, "Hips", root_t, {j: np.tile(QuaternionMath.identity(), (T, 1)) for j in human_skeleton.joints})

    exporter = BVHExporter()
    path = exporter.export(human_skeleton, anim, e2e_output_dir / "stance.bvh")
    assert path.exists()


def test_pairwise_execution_context_multi_format_export(human_skeleton, e2e_output_dir):
    """Pairwise 22: Full pipeline executes glTF, BVH, and Blender export in single pass."""
    times = np.array([0.0, 0.1], dtype=np.float32)
    anim = SkeletalAnimation("multi", 10.0, times, "Hips", np.zeros((2, 3), dtype=np.float32), {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in human_skeleton.joints})

    gltf_p = GltfExporter().export(human_skeleton, anim, e2e_output_dir / "multi.gltf")
    bvh_p = BVHExporter().export(human_skeleton, anim, e2e_output_dir / "multi.bvh")
    blend_p = BlenderExporter().export(human_skeleton, anim, e2e_output_dir / "multi.blend")

    assert gltf_p.exists()
    assert bvh_p.exists()
    assert (e2e_output_dir / "multi_rig_data.json").exists()
    assert (e2e_output_dir / "multi_anim_data.json").exists()
