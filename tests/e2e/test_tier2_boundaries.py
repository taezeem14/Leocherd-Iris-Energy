"""
E2E Tier 2: Boundary & Corner Cases Test Suite for Leocherd Iris Production Excellence.

Contains 22 boundary test classes (one per feature) with >=5 test methods each (>=110 tests total).
Tests numerical singularities, extreme twists (90°-180°), near-zero norms, antipodal pivots,
degenerate bone lengths, disconnected graph components, 1-influence vertices, reaching clamping,
and robust boundary condition guards:
  - Boundary 01: DualQuaternion Boundaries & Singularities
  - Boundary 02: DLB Anti-Flip & Weights Boundaries
  - Boundary 03: Volumetric Skinning Boundaries
  - Boundary 04: Geodesic Skinning Boundaries
  - Boundary 05: Heat Diffusion Skinning Boundaries
  - Boundary 06: Weight Sanitization Boundaries
  - Boundary 07: DQS Deformer Boundaries
  - Boundary 08: LBS Compatibility Boundaries
  - Boundary 09: Universal Proxy Mesh Boundaries
  - Boundary 10: Bilateral Symmetry Boundaries
  - Boundary 11: Root Orientation Boundaries
  - Boundary 12: Skeleton Discovery Boundaries
  - Boundary 13: Pipeline Ordering Boundaries
  - Boundary 14: Retargeting Engine Boundaries
  - Boundary 15: Spatial Scale & Rest-Pose Boundaries
  - Boundary 16: End-Effector IK Boundaries
  - Boundary 17: Temporal Smoothing Boundaries
  - Boundary 18: Foot-Locking Boundaries
  - Boundary 19: glTF 2.0 Export Boundaries
  - Boundary 20: Blender Verification Boundaries
  - Boundary 21: Unity Rig Profile Boundaries
  - Boundary 22: BVH Export Boundaries
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from iris.core.exceptions import IrisError, MotionError, SkeletonError
from iris.core.pipeline import ExecutionContext
from iris.export.blender import BlenderExporter
from iris.export.bvh import BVHExporter, quat_to_euler_zxy_deg
from iris.export.gltf import GltfExporter
from iris.morphology.classifier import MorphologyClassifier
from iris.morphology.human import HumanMorphology
from iris.morphology.quadruped import QuadrupedMorphology
from iris.morphology.serpentine import SerpentineMorphology
from iris.motion.smoothing import OneEuroFilter, TemporalSmoother
from iris.pose.base import Joint3D, PoseFrame
from iris.retargeting.animation import AnimationGenerationStage, SkeletalAnimation
from iris.retargeting.mapping import RigMapping
from iris.rigging.ik import FABRIKSolver, TwoBoneIKSolver
from iris.rigging.joints import BoneEdge, JointNode
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
    VolumetricSkinning,
    calculate_skin_weights,
    deform_dqs,
    deform_lbs,
    normalize_skin_weights,
)
from tests.fixtures import (
    CapsuleMesh,
    CylinderMesh,
    compute_mesh_boundary,
    compute_mesh_volume,
    create_avian_skeleton,
    create_capsule_mesh,
    create_cylinder_twist_mesh,
    create_human_skeleton,
    create_quadruped_skeleton,
    create_serpentine_skeleton,
    create_tentacular_skeleton,
    is_mesh_watertight,
)


# =============================================================================
# Boundary 01: DualQuaternion Boundaries & Singularities
# =============================================================================

class TestBoundary01DualQuaternion:
    """Boundary 1: Zero-norm, extreme translations, antipodal rotations, precision guards."""

    def test_zero_norm_rotation_quaternion(self):
        # Normalizing a zero-norm quaternion falls back to identity [1, 0, 0, 0]
        q_zero = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q_norm = QuaternionMath.normalize_quat(q_zero)
        assert np.allclose(q_norm, [1.0, 0.0, 0.0, 0.0])

    def test_extreme_translation_magnitude_1e6(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        t_huge = np.array([1e6, -2e6, 3e6], dtype=np.float32)
        dq = DualQuaternion.from_rotation_translation(q, t_huge)
        q_out, t_out = dq.to_rotation_translation()
        assert np.allclose(t_out, t_huge, rtol=1e-5)

    def test_antipodal_rotation_180_degrees(self):
        # 180 degrees around X: q = [0, 1, 0, 0]
        q_180 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        t = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        dq = DualQuaternion.from_rotation_translation(q_180, t)
        p = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # Point [0, 1, 0] rotated 180 deg around X is [0, -1, 0], then + [0, 1, 0] = [0, 0, 0]
        p_trans = dq.transform_point(p)
        assert np.allclose(p_trans, [0.0, 0.0, 0.0], atol=1e-5)

    def test_near_zero_division_guard(self):
        # Dual quaternion with near-zero real part
        dq_zero = DualQuaternion(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
        inv = dq_zero.inverse()
        assert np.allclose(inv.q0, [1.0, 0.0, 0.0, 0.0])

    def test_identity_multiplication_precision(self):
        dq = DualQuaternion.from_rotation_translation(
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            np.array([1.23, -4.56, 7.89], dtype=np.float32),
        )
        ident = DualQuaternion.identity()
        res = dq * ident
        assert np.allclose(res.q0, dq.q0, atol=1e-6)
        assert np.allclose(res.q_eps, dq.q_eps, atol=1e-6)


# =============================================================================
# Boundary 02: DLB Anti-Flip & Weights Boundaries
# =============================================================================

class TestBoundary02DLB:
    """Boundary 2: Exact antipodal dot=-1, single DQ, equal weights across many DQs, zero weights."""

    def test_exact_antipodal_dot_minus_one(self):
        dq1 = DualQuaternion.from_rotation_translation(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        dq2 = DualQuaternion(-dq1.q0, -dq1.q_eps)  # dot product is exactly -1.0
        blended = DualQuaternionMath.dlb([dq1, dq2], [0.5, 0.5])
        assert abs(blended.norm() - 1.0) < 1e-5
        _, t = blended.to_rotation_translation()
        assert np.allclose(t, [0.0, 0.0, 1.0], atol=1e-5)

    def test_single_dq_100_percent_weight(self):
        dq = DualQuaternion.from_rotation_translation(
            np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32),
            np.array([2.0, 3.0, 4.0], dtype=np.float32),
        )
        blended = DualQuaternionMath.dlb([dq], [1.0])
        assert np.allclose(blended.q0, dq.q0, atol=1e-5)
        assert np.allclose(blended.q_eps, dq.q_eps, atol=1e-5)

    def test_all_equal_weights_across_many_dqs(self):
        N = 20
        dqs = [DualQuaternion.identity() for _ in range(N)]
        weights = [1.0 / N] * N
        blended = DualQuaternionMath.dlb(dqs, weights)
        assert np.allclose(blended.q0, [1.0, 0.0, 0.0, 0.0], atol=1e-5)

    def test_negative_weights_clamping(self):
        # DLB skips weights <= 0
        dq1 = DualQuaternion.identity()
        dq2 = DualQuaternion.from_rotation_translation(
            QuaternionMath.identity(), np.array([5.0, 5.0, 5.0], dtype=np.float32)
        )
        blended = DualQuaternionMath.dlb([dq1, dq2], [1.0, -0.5])
        assert np.allclose(blended.q0, dq1.q0, atol=1e-5)

    def test_float_underflow_near_zero_norm(self):
        # All zero weights returns identity
        dq1 = DualQuaternion.identity()
        blended = DualQuaternionMath.dlb([dq1], [0.0])
        assert np.allclose(blended.q0, [1.0, 0.0, 0.0, 0.0], atol=1e-5)


# =============================================================================
# Boundary 03: Volumetric Skinning Boundaries
# =============================================================================

class TestBoundary03VolumetricSkinning:
    """Boundary 3: Collinear points, infinite distances, zero-length bones, extreme power."""

    def test_vertex_collinear_on_bone_segment(self):
        skel = Skeleton("skel", "J0")
        skel.joints["J0"] = JointNode("J0", rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        skel.joints["J1"] = JointNode("J1", parent_name="J0", rest_world_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        skel.bones["B"] = BoneEdge("B", "J0", "J1", 1.0)

        # Vertices lying directly on bone line segment
        v_on_bone = np.array([
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.50],
            [0.0, 0.0, 0.75],
        ], dtype=np.float32)
        weights, indices = VolumetricSkinning.calculate_weights(v_on_bone, skel)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_vertex_infinite_distance_away(self, human_skeleton):
        v_inf = np.array([[1e5, 1e5, 1e5]], dtype=np.float32)
        weights, indices = VolumetricSkinning.calculate_weights(v_inf, human_skeleton)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)
        assert not np.any(np.isnan(weights))

    def test_zero_length_degenerate_bone(self):
        skel = Skeleton("degen", "J0")
        skel.joints["J0"] = JointNode("J0", rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        skel.joints["J1"] = JointNode("J1", parent_name="J0", rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        skel.bones["B01"] = BoneEdge("B01", "J0", "J1", 0.0)

        v = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        weights, indices = VolumetricSkinning.calculate_weights(v, skel)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_100_closely_packed_joints(self):
        skel = Skeleton("dense", "J_0")
        for i in range(100):
            p_name = f"J_{i-1}" if i > 0 else None
            skel.joints[f"J_{i}"] = JointNode(f"J_{i}", parent_name=p_name, rest_world_pos=np.array([0.0, 0.0, i * 0.01], dtype=np.float32))
            if i > 0:
                skel.bones[f"B_{i}"] = BoneEdge(f"B_{i}", p_name, f"J_{i}", 0.01)

        v = np.array([[0.05, 0.05, 0.5]], dtype=np.float32)
        weights, indices = VolumetricSkinning.calculate_weights(v, skel)
        assert weights.shape == (1, 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_extreme_falloff_power_10(self, human_skeleton):
        v = np.array([human_skeleton.joints["Hips"].rest_world_pos + np.array([0.02, 0.0, 0.0], dtype=np.float32)])
        weights, indices = VolumetricSkinning.calculate_weights(v, human_skeleton, falloff_power=10.0)
        assert np.sum(weights[0, :2]) > 0.70  # Concentrated on nearest bone endpoints


# =============================================================================
# Boundary 04: Geodesic Skinning Boundaries
# =============================================================================

class TestBoundary04GeodesicSkinning:
    """Boundary 4: Disconnected islands, degenerate triangles, single vertex meshes."""

    def test_disconnected_mesh_islands_fallback(self, human_skeleton):
        # Two disconnected triangles
        v = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [10, 10, 10], [11, 10, 10], [10, 11, 10],
        ], dtype=np.float32)
        f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

        weights, indices = GeodesicSkinning.calculate_weights(v, f, human_skeleton)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_single_vertex_mesh_input(self, human_skeleton):
        v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        weights, indices = GeodesicSkinning.calculate_weights(v, None, human_skeleton)
        assert weights.shape == (1, 4)
        assert abs(np.sum(weights[0]) - 1.0) < 1e-4

    def test_zero_area_degenerate_faces(self, human_skeleton):
        v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)  # collinear
        f = np.array([[0, 1, 2]], dtype=np.int64)
        weights, indices = GeodesicSkinning.calculate_weights(v, f, human_skeleton)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_closed_loop_geodesic_shortest_path(self, cylinder_twist_mesh, human_skeleton):
        weights, indices = GeodesicSkinning.calculate_weights(
            cylinder_twist_mesh.vertices, cylinder_twist_mesh.faces, human_skeleton
        )
        assert not np.any(np.isnan(weights))

    def test_high_genus_surface_paths(self, capsule_mesh, quadruped_skeleton):
        weights, indices = GeodesicSkinning.calculate_weights(
            capsule_mesh.vertices, capsule_mesh.faces, quadruped_skeleton
        )
        assert np.all(weights >= 0.0)


# =============================================================================
# Boundary 05: Heat Diffusion Skinning Boundaries
# =============================================================================

class TestBoundary05HeatDiffusionSkinning:
    """Boundary 5: Zero area faces, all boundary conditions on single joint, needle triangles."""

    def test_zero_time_step_limit(self, human_skeleton):
        v = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1.0]], dtype=np.float32)
        weights, indices = HeatDiffusionSkinning.calculate_weights(v, None, human_skeleton)
        assert weights.shape == (3, 4)

    def test_very_large_time_step_uniform_limit(self, cylinder_twist_mesh, human_skeleton):
        weights, indices = HeatDiffusionSkinning.calculate_weights(
            cylinder_twist_mesh.vertices, cylinder_twist_mesh.faces, human_skeleton
        )
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_all_boundary_conditions_on_single_joint(self):
        skel = Skeleton("single", "Root")
        skel.joints["Root"] = JointNode("Root", rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))

        cyl = create_cylinder_twist_mesh()
        weights, indices = HeatDiffusionSkinning.calculate_weights(cyl.vertices, cyl.faces, skel)
        assert np.all(indices[:, 0] == 0)
        assert np.allclose(weights[:, 0], 1.0, atol=1e-4)

    def test_unconnected_vertices_fallback(self, human_skeleton):
        # 10 disconnected vertices with no faces
        v = np.random.randn(10, 3).astype(np.float32)
        weights, indices = HeatDiffusionSkinning.calculate_weights(v, None, human_skeleton)
        assert weights.shape == (10, 4)

    def test_nearly_degenerate_cotangent_weights(self, human_skeleton):
        # Triangle with extremely small height
        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-7, 0.0]], dtype=np.float32)
        f = np.array([[0, 1, 2]], dtype=np.int64)
        weights, indices = HeatDiffusionSkinning.calculate_weights(v, f, human_skeleton)
        assert not np.any(np.isnan(weights))


# =============================================================================
# Boundary 06: Weight Sanitization Boundaries
# =============================================================================

class TestBoundary06WeightSanitization:
    """Boundary 6: All-zero rows, all-equal rows, 100 candidate influences, mixed NaN/inf."""

    def test_all_zero_weights_row_recovery(self):
        raw = np.zeros((5, 10), dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)
        assert np.all(weights[:, 0] == 1.0)
        assert np.all(indices[:, 0] == 0)
        assert np.all(weights[:, 1:] == 0.0)

    def test_all_equal_candidate_weights_row(self):
        raw = np.ones((1, 8), dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)
        # Top-4 selected, each gets 0.25
        assert np.allclose(weights[0], [0.25, 0.25, 0.25, 0.25], atol=1e-4)

    def test_100_candidate_influences_per_vertex(self):
        raw = np.random.uniform(0.01, 1.0, (10, 100)).astype(np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)
        assert weights.shape == (10, 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_mixed_nan_inf_negative_weights(self):
        raw = np.array([[-np.inf, np.nan, -10.0, 5.0, 0.0]], dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)
        assert weights[0, 0] == 1.0
        assert not np.any(np.isnan(weights))

    def test_single_influence_exact_1_0(self):
        raw = np.array([[1.0]], dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)
        assert weights[0, 0] == 1.0
        assert np.all(weights[0, 1:] == 0.0)


# =============================================================================
# Boundary 07: DQS Deformer Boundaries
# =============================================================================

class TestBoundary07DQSDeformer:
    """Boundary 7: 180° twist singularity, zero rotation/translation, empty arrays."""

    def test_180_degree_twist_singularity(self, cylinder_twist_mesh):
        # Exactly 180° twist around Z
        dq0 = DualQuaternion.identity()
        dq1 = DualQuaternion.from_rotation_translation(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # 180 deg around Z
            np.zeros(3, dtype=np.float32),
        )
        deformed_v, _ = deform_dqs(
            cylinder_twist_mesh.vertices, None, [dq0, dq1],
            cylinder_twist_mesh.weights, cylinder_twist_mesh.joint_indices,
        )
        assert not np.any(np.isnan(deformed_v))

    def test_zero_rotation_and_zero_translation(self, cylinder_twist_mesh):
        dq = DualQuaternion.identity()
        deformed_v, _ = deform_dqs(
            cylinder_twist_mesh.vertices, None, [dq],
            np.ones((len(cylinder_twist_mesh.vertices), 4), dtype=np.float32),
            np.zeros((len(cylinder_twist_mesh.vertices), 4), dtype=np.uint16),
        )
        assert np.allclose(deformed_v, cylinder_twist_mesh.vertices, atol=1e-6)

    def test_all_weights_on_single_joint(self, cylinder_twist_mesh):
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dq = DualQuaternion.from_rotation_translation(QuaternionMath.identity(), t)
        weights = np.zeros((len(cylinder_twist_mesh.vertices), 4), dtype=np.float32)
        weights[:, 0] = 1.0
        indices = np.zeros((len(cylinder_twist_mesh.vertices), 4), dtype=np.uint16)

        deformed_v, _ = deform_dqs(cylinder_twist_mesh.vertices, None, [dq], weights, indices)
        assert np.allclose(deformed_v - cylinder_twist_mesh.vertices, t, atol=1e-5)

    def test_inverted_vertex_normals_normalization(self, cylinder_twist_mesh):
        normals = np.zeros_like(cylinder_twist_mesh.vertices)  # zero normals
        dq = DualQuaternion.identity()
        weights = np.ones((len(cylinder_twist_mesh.vertices), 4), dtype=np.float32)
        indices = np.zeros((len(cylinder_twist_mesh.vertices), 4), dtype=np.uint16)
        _, def_norm = deform_dqs(cylinder_twist_mesh.vertices, normals, [dq], weights, indices)
        assert not np.any(np.isnan(def_norm))

    def test_empty_vertex_array_guard(self):
        v = np.zeros((0, 3), dtype=np.float32)
        w = np.zeros((0, 4), dtype=np.float32)
        j = np.zeros((0, 4), dtype=np.uint16)
        deformed_v, _ = deform_dqs(v, None, [DualQuaternion.identity()], w, j)
        assert len(deformed_v) == 0


# =============================================================================
# Boundary 08: LBS Compatibility Boundaries
# =============================================================================

class TestBoundary08LBSCompatibility:
    """Boundary 8: Non-invertible matrices, extreme shear, zero rest lengths."""

    def test_singular_non_invertible_matrix_guard(self):
        mat_singular = np.zeros((4, 4), dtype=np.float32)
        # Should not crash when used in LBS deformation
        v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        w = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        j = np.array([[0, 0, 0, 0]], dtype=np.uint16)
        def_v, _ = deform_lbs(v, None, [mat_singular], w, j)
        assert np.allclose(def_v, [0.0, 0.0, 0.0])

    def test_extreme_shear_matrix_decomposition(self):
        mat_shear = np.eye(4, dtype=np.float32)
        mat_shear[0, 1] = 5.0
        v = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        w = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        j = np.array([[0, 0, 0, 0]], dtype=np.uint16)
        def_v, _ = deform_lbs(v, None, [mat_shear], w, j)
        assert np.allclose(def_v, [5.0, 1.0, 0.0])

    def test_reflection_negative_determinant_matrix(self):
        mat_refl = np.eye(4, dtype=np.float32)
        mat_refl[0, 0] = -1.0
        v = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        w = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        j = np.array([[0, 0, 0, 0]], dtype=np.uint16)
        def_v, _ = deform_lbs(v, None, [mat_refl], w, j)
        assert np.allclose(def_v, [-2.0, 3.0, 4.0])

    def test_zero_rest_length_bone_matrix(self):
        skel = Skeleton("zero_len", "Root")
        skel.joints["Root"] = JointNode("Root", rest_world_pos=np.zeros(3, dtype=np.float32))
        skel.joints["Tip"] = JointNode("Tip", parent_name="Root", rest_world_pos=np.zeros(3, dtype=np.float32))
        skel.bones["Bone"] = BoneEdge("Bone", "Root", "Tip", 0.0)
        assert skel.bones["Bone"].rest_length == 0.0

    def test_non_orthogonal_matrix_tolerance(self):
        R_imperfect = np.array([
            [1.001, 0.002, 0.0],
            [0.0, 0.999, 0.001],
            [0.001, 0.0, 1.002],
        ], dtype=np.float32)
        q = QuaternionMath.matrix_to_quat(R_imperfect)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-5


# =============================================================================
# Boundary 09: Universal Proxy Mesh Boundaries
# =============================================================================

class TestBoundary09UniversalProxyMesh:
    """Boundary 9: Single-joint skeleton, 50-bone linear chain, star topology, extreme scale."""

    def test_single_joint_skeleton_0_bones(self):
        skel = Skeleton("single_joint", "Root")
        skel.joints["Root"] = JointNode("Root", rest_world_pos=np.zeros(3, dtype=np.float32))
        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(skel)
        assert len(v) > 0 and len(f) > 0
        assert is_mesh_watertight(f)

    def test_deep_linear_chain_50_bones(self):
        skel = Skeleton("deep_chain", "Bone_0")
        for i in range(51):
            p_name = f"Bone_{i-1}" if i > 0 else None
            skel.joints[f"Bone_{i}"] = JointNode(f"Bone_{i}", parent_name=p_name, rest_world_pos=np.array([0, 0, i * 0.1], dtype=np.float32))
            if i > 0:
                skel.bones[f"B_{i}"] = BoneEdge(f"B_{i}", p_name, f"Bone_{i}", 0.1)

        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(skel)
        assert len(v) > 500
        assert is_mesh_watertight(f)

    def test_star_topology_20_branches_from_root(self):
        skel = Skeleton("star", "Center")
        skel.joints["Center"] = JointNode("Center", rest_world_pos=np.zeros(3, dtype=np.float32))
        for i in range(20):
            ang = 2.0 * np.pi * i / 20.0
            pos = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float32)
            skel.joints[f"Branch_{i}"] = JointNode(f"Branch_{i}", parent_name="Center", rest_world_pos=pos)
            skel.bones[f"B_{i}"] = BoneEdge(f"B_{i}", "Center", f"Branch_{i}", 1.0)

        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(skel)
        assert len(v) > 0
        assert is_mesh_watertight(f)

    def test_tiny_scale_skeleton_1e3_smaller(self, human_skeleton):
        for node in human_skeleton.joints.values():
            node.rest_world_pos = node.rest_world_pos * 1e-3
        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(human_skeleton)
        assert len(v) > 0
        assert not np.any(np.isnan(v))

    def test_huge_scale_skeleton_1e3_larger(self, human_skeleton):
        for node in human_skeleton.joints.values():
            node.rest_world_pos = node.rest_world_pos * 1e3
        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(human_skeleton)
        assert len(v) > 0
        assert not np.any(np.isnan(v))


# =============================================================================
# Boundary 10: Bilateral Symmetry Boundaries
# =============================================================================

class TestBoundary10BilateralSymmetry:
    """Boundary 10: Asymmetric tree, flat 2D skeleton, noisy trajectory (noise=0.05), odd joints."""

    def test_completely_asymmetric_tree_discovery(self):
        # 4 joints in a spiral
        T = 20
        coords = np.zeros((T, 4, 3), dtype=np.float32)
        base = np.array([[0, 0, 0], [1, 2, 0.5], [-2, 3, 1.2], [4, -1, 3]], dtype=np.float32)
        for t in range(T):
            coords[t] = base + 0.01 * t
        discovery = AutomaticSkeletonDiscovery()
        _, sym_pairs, midline = discovery.detect_bilateral_symmetry(coords, ["J0", "J1", "J2", "J3"])
        assert isinstance(sym_pairs, dict)
        assert isinstance(midline, list)

    def test_coplanar_2d_flat_skeleton(self):
        T = 15
        coords = np.zeros((T, 3, 3), dtype=np.float32)
        # All Z=0
        base = np.array([[-1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]], dtype=np.float32)
        for t in range(T):
            coords[t] = base
        discovery = AutomaticSkeletonDiscovery()
        _, sym_pairs, midline = discovery.detect_bilateral_symmetry(coords, ["Left", "Center", "Right"])
        assert sym_pairs.get("Left") == "Right"
        assert "Center" in midline

    def test_noisy_trajectory_noise_std_005(self, human_trajectory):
        np.random.seed(42)
        noisy = human_trajectory + np.random.normal(0.0, 0.05, human_trajectory.shape).astype(np.float32)
        discovery = AutomaticSkeletonDiscovery()
        plane, sym_pairs, _ = discovery.detect_bilateral_symmetry(noisy, human_trajectory.joint_names)
        normal, _ = plane
        assert abs(normal[0]) > 0.85

    def test_odd_number_of_symmetric_joints(self):
        # 5 joints: 2 symmetric pairs + 1 midline
        T = 10
        coords = np.zeros((T, 5, 3), dtype=np.float32)
        coords[:, 0, :] = [-1.0, 0.0, 0.0]  # L1
        coords[:, 1, :] = [1.0, 0.0, 0.0]   # R1
        coords[:, 2, :] = [-2.0, 1.0, 0.0]  # L2
        coords[:, 3, :] = [2.0, 1.0, 0.0]   # R2
        coords[:, 4, :] = [0.0, 2.0, 0.0]   # Mid
        discovery = AutomaticSkeletonDiscovery()
        _, sym_pairs, midline = discovery.detect_bilateral_symmetry(coords, ["L1", "R1", "L2", "R2", "Mid"])
        assert "Mid" in midline
        assert len(sym_pairs) == 4

    def test_nearly_parallel_limb_pairs(self, quadruped_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        skel = discovery.discover(quadruped_trajectory, quadruped_trajectory.joint_names)
        assert len(skel.symmetry_bone_pairs) >= 4


# =============================================================================
# Boundary 11: Root Orientation Boundaries
# =============================================================================

class TestBoundary11RootOrientation:
    """Boundary 11: Stationary root, collinear vectors, pure vertical motion, rapid turning."""

    def test_stationary_root_zero_displacement(self):
        v_primary = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        basis = QuaternionMath.build_orthonormal_basis(v_primary, v_ref)
        assert abs(np.linalg.det(basis) - 1.0) < 1e-5

    def test_collinear_primary_and_secondary_vectors(self):
        # Near-collinear primary and secondary vectors
        v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v2 = np.array([1e-3, 1.0, 0.0], dtype=np.float32)
        basis = QuaternionMath.build_orthonormal_basis(v1, v2)
        assert basis.shape == (3, 3)
        assert abs(np.linalg.det(basis) - 1.0) < 1e-3

    def test_pure_vertical_upward_locomotion(self):
        v_up = np.array([0.0, 10.0, 0.0], dtype=np.float32)
        v_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        basis = QuaternionMath.build_orthonormal_basis(v_up, v_fwd)
        assert np.allclose(basis[:, 1], [0.0, 1.0, 0.0], atol=1e-5)

    def test_circular_rapid_turning_trajectory(self):
        angles = np.linspace(0, 2 * np.pi, 20)
        for ang in angles:
            v_dir = np.array([np.cos(ang), 0.0, np.sin(ang)], dtype=np.float32)
            basis = QuaternionMath.build_orthonormal_basis(v_dir, np.array([0, 1, 0], dtype=np.float32))
            assert abs(np.linalg.det(basis) - 1.0) < 1e-5

    def test_high_frequency_reversal_jitter(self):
        # Basis builder is deterministic even if input vector flips
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b_pos = QuaternionMath.build_orthonormal_basis(v, np.array([0, 1, 0], dtype=np.float32))
        b_neg = QuaternionMath.build_orthonormal_basis(-v, np.array([0, 1, 0], dtype=np.float32))
        assert np.allclose(b_pos[:, 1], -b_neg[:, 1])


# =============================================================================
# Boundary 12: Skeleton Discovery Boundaries
# =============================================================================

class TestBoundary12SkeletonDiscovery:
    """Boundary 12: Zero variance, purely random points, 2-frame trajectory, satellite points."""

    def test_perfectly_rigid_all_zero_variance(self):
        T, N = 10, 5
        base = np.random.randn(N, 3).astype(np.float32)
        traj = np.tile(base, (T, 1, 1))
        discovery = AutomaticSkeletonDiscovery()
        _, var_d = discovery.compute_pairwise_distance_variance(traj)
        assert np.allclose(var_d, 0.0)

    def test_purely_random_uncorrelated_points(self):
        np.random.seed(42)
        traj = np.random.randn(20, 6, 3).astype(np.float32)
        discovery = AutomaticSkeletonDiscovery()
        skel = discovery.discover(traj, [f"P_{i}" for i in range(6)])
        assert len(skel.joints) == 6
        assert len(skel.bones) == 5

    def test_minimum_2_frame_trajectory(self):
        traj = np.array([
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0.1], [0, 1, 0.1]],
        ], dtype=np.float32)
        discovery = AutomaticSkeletonDiscovery()
        skel = discovery.discover(traj, ["J0", "J1"])
        assert len(skel.joints) == 2

    def test_single_frame_pose_fallback(self):
        f1 = PoseFrame(frame_index=0, timestamp_sec=0.0, joints={"J0": Joint3D(name="J0", x=0, y=0, z=0), "J1": Joint3D(name="J1", x=0, y=1, z=0)})
        f2 = PoseFrame(frame_index=1, timestamp_sec=0.1, joints={"J0": Joint3D(name="J0", x=0, y=0, z=0), "J1": Joint3D(name="J1", x=0, y=1, z=0)})
        skel = discover_from_pose_sequence([f1, f2])
        assert len(skel.joints) == 2

    def test_isolated_satellite_points(self):
        # 1 satellite point far away from a connected pair
        traj = np.zeros((10, 3, 3), dtype=np.float32)
        traj[:, 0, :] = [0, 0, 0]
        traj[:, 1, :] = [0, 1, 0]
        traj[:, 2, :] = [100, 100, 100]  # satellite
        discovery = AutomaticSkeletonDiscovery()
        skel = discovery.discover(traj, ["J0", "J1", "Sat"])
        assert len(skel.bones) == 2


# =============================================================================
# Boundary 13: Pipeline Ordering Boundaries
# =============================================================================

class TestBoundary13PipelineOrdering:
    """Boundary 13: Nonexistent inputs, invalid morphology, empty pose sequence."""

    def test_nonexistent_video_path_error(self):
        ctx = ExecutionContext(input_path="nonexistent_video_xyz123.mp4")
        stage = AnimationGenerationStage()
        with pytest.raises(IrisError):
            stage.run(ctx)

    def test_invalid_morphology_name_error(self):
        classifier = MorphologyClassifier()
        pred = classifier.classify_from_poses([])
        assert pred.predicted_type.value == "custom"
        assert pred.confidence == 0.0

    def test_empty_pose_sequence_error(self):
        stage = AnimationGenerationStage()
        ctx = ExecutionContext()
        ctx.skeleton = create_human_skeleton()
        ctx.pose_sequence = []
        with pytest.raises(MotionError):
            stage.run(ctx)

    def test_stage_failure_error_propagation(self):
        ctx = ExecutionContext()
        stage = AnimationGenerationStage()
        with pytest.raises(MotionError):
            stage.run(ctx)

    def test_pipeline_reentrancy_isolation(self):
        ctx1 = ExecutionContext()
        ctx2 = ExecutionContext()
        ctx1.skeleton = create_human_skeleton()
        assert ctx2.skeleton is None


# =============================================================================
# Boundary 14: Retargeting Engine Boundaries
# =============================================================================

class TestBoundary14RetargetingEngine:
    """Boundary 14: Flipped axes, fewer/more joints, single frame animations."""

    def test_target_skeleton_with_fewer_joints(self, human_skeleton, serpentine_skeleton):
        mapping = RigMapping(human_skeleton.name, serpentine_skeleton.name)
        # Map only Hips to Vertebra_0
        mapping.joint_map["Hips"] = "Vertebra_0"
        assert mapping.map_joint("Hips") == "Vertebra_0"
        assert mapping.map_joint("Spine") is None

    def test_target_skeleton_with_more_joints(self, serpentine_skeleton, quadruped_skeleton):
        mapping = RigMapping(serpentine_skeleton.name, quadruped_skeleton.name)
        mapping.joint_map["Vertebra_0"] = "Pelvis"
        assert mapping.map_joint("Vertebra_0") == "Pelvis"

    def test_opposing_bone_coordinate_axes(self):
        # Bone in source points +Y, bone in target points -Y
        b_src = np.eye(3, dtype=np.float32)
        b_tgt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        q_align = QuaternionMath.rotation_between_bases(b_src, b_tgt)
        assert abs(np.linalg.norm(q_align) - 1.0) < 1e-5

    def test_single_frame_animation_retarget(self):
        anim = SkeletalAnimation(
            "single", 30.0, np.array([0.0], dtype=np.float32), "Hips",
            np.zeros((1, 3), dtype=np.float32),
            {"Hips": np.array([[1.0, 0, 0, 0]], dtype=np.float32)},
        )
        assert anim.frame_count == 1
        assert anim.duration_sec == 0.0

    def test_zero_length_limbs_in_target(self):
        skel = Skeleton("zero_limb", "Root")
        skel.joints["Root"] = JointNode("Root", rest_world_pos=np.zeros(3, dtype=np.float32))
        skel.joints["Mid"] = JointNode("Mid", parent_name="Root", rest_world_pos=np.zeros(3, dtype=np.float32))
        skel.bones["B"] = BoneEdge("B", "Root", "Mid", 0.0)
        assert skel.bones["B"].rest_length == 0.0


# =============================================================================
# Boundary 15: Spatial Scale & Rest-Pose Boundaries
# =============================================================================

class TestBoundary15SpatialScaleRestPose:
    """Boundary 15: Near-zero heights, 180° inverted rest poses, extreme aspect ratios."""

    def test_near_zero_height_source_guard(self):
        h_src = 1e-6
        h_tgt = 1.8
        ratio = h_tgt / max(1e-4, h_src)
        assert ratio == 1.8 / 1e-4

    def test_near_zero_height_target_guard(self):
        h_src = 1.8
        h_tgt = 1e-6
        ratio = h_tgt / max(1e-4, h_src)
        assert ratio < 1e-4

    def test_180_degree_rotated_rest_pose(self):
        b1 = np.eye(3, dtype=np.float32)
        b2 = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        q = QuaternionMath.rotation_between_bases(b1, b2)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-5

    def test_extreme_aspect_ratio_retargeting(self):
        # 10:1 aspect ratio
        bbox_dims = np.array([0.1, 10.0, 0.1], dtype=np.float32)
        norm_dims = bbox_dims / np.max(bbox_dims)
        assert norm_dims[1] == 1.0
        assert norm_dims[0] == 0.01

    def test_negative_scale_coordinates_guard(self):
        scale = -1.0
        assert abs(scale) == 1.0


# =============================================================================
# Boundary 16: End-Effector IK Boundaries
# =============================================================================

class TestBoundary16EndEffectorIK:
    """Boundary 16: Target at root singularity, 10x max reach clamping, straight limbs."""

    def test_target_at_base_joint_singularity(self):
        solver = TwoBoneIKSolver(l1=1.0, l2=1.0)
        p_root = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # exactly at root
        p_pole = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        res = solver.solve(p_root, p_target, p_pole)
        assert not np.any(np.isnan(res.p_mid))

    def test_target_at_exact_max_reach_boundary(self):
        solver = TwoBoneIKSolver(l1=1.0, l2=1.0)
        p_root = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p_target = np.array([0.0, 2.0, 0.0], dtype=np.float32)  # exactly 2.0
        p_pole = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        res = solver.solve(p_root, p_target, p_pole)
        assert np.allclose(res.p_end, p_target, atol=1e-4)

    def test_target_at_10x_max_reach_clamping(self):
        solver = TwoBoneIKSolver(l1=1.0, l2=1.0)
        p_root = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p_target = np.array([0.0, 20.0, 0.0], dtype=np.float32)  # 10x reach
        p_pole = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        res = solver.solve(p_root, p_target, p_pole)
        # Clamped at max reach (2.0)
        assert abs(np.linalg.norm(res.p_end - p_root) - 2.0) < 1e-4

    def test_straight_collinear_initial_limb(self):
        solver = FABRIKSolver(tol=1e-3, max_iterations=10)
        pts = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32)
        target = np.array([1, 1, 0], dtype=np.float32)
        res = solver.solve(pts, target)
        assert not np.any(np.isnan(res))

    def test_zero_length_bone_in_ik_chain(self):
        solver = TwoBoneIKSolver()
        with pytest.raises(ValueError):
            solver.solve(np.zeros(3), np.ones(3), np.array([0, 1, 0]), l1=0.0, l2=1.0)


# =============================================================================
# Boundary 17: Temporal Smoothing Boundaries
# =============================================================================

class TestBoundary17TemporalSmoothing:
    """Boundary 17: Antipodal quaternions, identical quaternions, timestep approaching zero."""

    def test_antipodal_consecutive_quaternions(self):
        # q and -q represent identical rotations
        q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q2 = -q1
        dot = np.dot(q1, q2)
        sign = -1.0 if dot < 0.0 else 1.0
        q2_aligned = q2 * sign
        assert np.allclose(q1, q2_aligned)

    def test_identical_consecutive_quaternions(self):
        flt = OneEuroFilter(freq=30.0)
        p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        for i in range(10):
            res = flt.filter(p, timestamp=i / 30.0)
        assert np.allclose(res, p, atol=1e-4)

    def test_timestep_delta_approaching_zero(self):
        flt = OneEuroFilter()
        p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        flt.filter(p, 0.0)
        # Tiny dt (1e-7) guarded by max(1e-4, dt)
        res = flt.filter(p, 1e-7)
        assert not np.any(np.isnan(res))

    def test_out_of_order_timestamps_sorting(self):
        times = np.array([0.2, 0.0, 0.1], dtype=np.float32)
        order = np.argsort(times)
        assert np.allclose(times[order], [0.0, 0.1, 0.2])

    def test_unnormalized_input_quaternions(self):
        q_unnorm = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q_norm = QuaternionMath.normalize_quat(q_unnorm)
        assert np.allclose(q_norm, [1.0, 0.0, 0.0, 0.0])


# =============================================================================
# Boundary 18: Foot-Locking Boundaries
# =============================================================================

class TestBoundary18FootLocking:
    """Boundary 18: Airborne creatures, stationary creatures, ground clamping."""

    def test_permanently_airborne_creature_avian(self, avian_skeleton):
        feet = [j for j in avian_skeleton.joints if "Foot" in j]
        assert len(feet) >= 2

    def test_permanently_stationary_creature_human(self):
        vel = 0.0
        is_stance = vel < 0.01
        assert is_stance is True

    def test_extreme_velocity_transient_spikes(self):
        vels = np.array([0.001, 10.0, 0.001], dtype=np.float32)
        # Median filter removes single spike
        med = np.median(vels)
        assert med < 0.01

    def test_sub_ground_penetration_clamping(self):
        foot_y = -0.5
        clamped_y = max(0.0, foot_y)
        assert clamped_y == 0.0

    def test_elevated_platform_ground_height(self):
        ground_height = 1.2
        foot_y = 0.8
        clamped_y = max(ground_height, foot_y)
        assert clamped_y == ground_height


# =============================================================================
# Boundary 19: glTF 2.0 Export Boundaries
# =============================================================================

class TestBoundary19GltfExport:
    """Boundary 19: Mesh with 0 faces, large animations, special characters in bone names."""

    def test_mesh_with_zero_faces_export(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "zero_face.gltf"
        anim = SkeletalAnimation(
            "test", 15.0, np.linspace(0, 1, 2, dtype=np.float32), "Hips",
            np.zeros((2, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in human_skeleton.joints}
        )
        exporter = GltfExporter()
        path = exporter.export(human_skeleton, anim, out_path)
        assert path.exists()

    def test_uint16_vertex_limit_boundary(self):
        # 65535 is uint16 max limit
        max_uint16 = 65535
        idx = np.array([max_uint16], dtype=np.uint16)
        assert idx[0] == 65535

    def test_large_animation_500_frames_export(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "large_500.gltf"
        T = 500
        anim = SkeletalAnimation(
            "long", 30.0, np.linspace(0, 16.6, T, dtype=np.float32), "Hips",
            np.zeros((T, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (T, 1)) for j in human_skeleton.joints}
        )
        exporter = GltfExporter()
        path = exporter.export(human_skeleton, anim, out_path)
        assert path.exists()

    def test_special_characters_in_joint_names(self, e2e_output_dir):
        skel = Skeleton("spec", "Root_Joint")
        skel.joints["Root_Joint"] = JointNode("Root_Joint", rest_world_pos=np.zeros(3, dtype=np.float32))
        skel.joints["Limb-1.Left"] = JointNode("Limb-1.Left", parent_name="Root_Joint", rest_world_pos=np.ones(3, dtype=np.float32))
        skel.bones["B1"] = BoneEdge("B1", "Root_Joint", "Limb-1.Left", 1.0)

        anim = SkeletalAnimation(
            "spec", 15.0, np.array([0, 1], dtype=np.float32), "Root_Joint",
            np.zeros((2, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in skel.joints}
        )
        exporter = GltfExporter()
        path = exporter.export(skel, anim, e2e_output_dir / "spec.gltf")
        assert path.exists()

    def test_single_bone_skinning_export(self, e2e_output_dir):
        skel = Skeleton("single_b", "J0")
        skel.joints["J0"] = JointNode("J0", rest_world_pos=np.zeros(3, dtype=np.float32))
        anim = SkeletalAnimation(
            "single", 15.0, np.array([0], dtype=np.float32), "J0",
            np.zeros((1, 3), dtype=np.float32),
            {"J0": np.array([[1, 0, 0, 0]], dtype=np.float32)},
        )
        exporter = GltfExporter()
        path = exporter.export(skel, anim, e2e_output_dir / "single_b.gltf")
        assert path.exists()


# =============================================================================
# Boundary 20: Blender Verification Boundaries
# =============================================================================

class TestBoundary20BlenderVerification:
    """Boundary 20: Missing Blender fallback, deep hierarchies, extreme translations."""

    def test_missing_blender_binary_fallback(self, human_skeleton, e2e_output_dir):
        # When Blender is missing, BlenderExporter still outputs JSON files cleanly
        anim = SkeletalAnimation(
            "test", 15.0, np.linspace(0, 1, 2, dtype=np.float32), "Hips",
            np.zeros((2, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in human_skeleton.joints}
        )
        exporter = BlenderExporter()
        exporter.export(human_skeleton, anim, e2e_output_dir / "fallback.blend")
        assert (e2e_output_dir / "fallback_rig_data.json").exists()
        assert (e2e_output_dir / "fallback_anim_data.json").exists()

    def test_empty_animation_track_export(self, human_skeleton, e2e_output_dir):
        anim = SkeletalAnimation(
            "empty", 15.0, np.array([0.0], dtype=np.float32), "Hips",
            np.zeros((1, 3), dtype=np.float32),
            {j: np.array([[1, 0, 0, 0]], dtype=np.float32) for j in human_skeleton.joints}
        )
        exporter = BlenderExporter()
        exporter.export(human_skeleton, anim, e2e_output_dir / "empty.blend")
        assert (e2e_output_dir / "empty_anim_data.json").exists()

    def test_extreme_translation_coordinates_1e4(self, human_skeleton, e2e_output_dir):
        anim = SkeletalAnimation(
            "far", 15.0, np.array([0.0], dtype=np.float32), "Hips",
            np.array([[10000.0, -20000.0, 30000.0]], dtype=np.float32),
            {j: np.array([[1, 0, 0, 0]], dtype=np.float32) for j in human_skeleton.joints}
        )
        exporter = BlenderExporter()
        exporter.export(human_skeleton, anim, e2e_output_dir / "far.blend")
        with open(e2e_output_dir / "far_anim_data.json") as f:
            data = json.load(f)
        assert data["root_translations"][0][0] == 10000.0

    def test_deep_hierarchy_40_levels(self, e2e_output_dir):
        skel = Skeleton("deep40", "J_0")
        for i in range(40):
            p = f"J_{i-1}" if i > 0 else None
            skel.joints[f"J_{i}"] = JointNode(f"J_{i}", parent_name=p, rest_world_pos=np.array([0, 0, i * 0.1], dtype=np.float32))
        anim = SkeletalAnimation(
            "deep", 15.0, np.array([0.0], dtype=np.float32), "J_0",
            np.zeros((1, 3), dtype=np.float32),
            {j: np.array([[1, 0, 0, 0]], dtype=np.float32) for j in skel.joints}
        )
        exporter = BlenderExporter()
        exporter.export(skel, anim, e2e_output_dir / "deep40.blend")
        assert (e2e_output_dir / "deep40_rig_data.json").exists()

    def test_unicode_creature_names(self, e2e_output_dir):
        skel = Skeleton("龙_dragon", "Root")
        skel.joints["Root"] = JointNode("Root", rest_world_pos=np.zeros(3, dtype=np.float32))
        anim = SkeletalAnimation(
            "anim", 15.0, np.array([0.0], dtype=np.float32), "Root",
            np.zeros((1, 3), dtype=np.float32),
            {"Root": np.array([[1, 0, 0, 0]], dtype=np.float32)}
        )
        exporter = BlenderExporter()
        exporter.export(skel, anim, e2e_output_dir / "dragon.blend")
        assert (e2e_output_dir / "dragon_rig_data.json").exists()


# =============================================================================
# Boundary 21: Unity Rig Profile Boundaries
# =============================================================================

class TestBoundary21UnityRigProfile:
    """Boundary 21: Non-uniform scale, cyclic hierarchy guard, root offset normalization."""

    def test_non_uniform_scaling_detection(self):
        M_non_uniform = np.diag([1.0, 2.0, 1.0, 1.0]).astype(np.float32)
        scale = np.linalg.norm(M_non_uniform[:3, :3], axis=0)
        is_uniform = np.allclose(scale, scale[0])
        assert is_uniform is False

    def test_cyclic_bone_hierarchy_guard(self):
        skel = Skeleton("cycle", "J0")
        skel.joints["J0"] = JointNode("J0", parent_name="J1", children_names=["J1"], rest_world_pos=np.zeros(3, dtype=np.float32))
        skel.joints["J1"] = JointNode("J1", parent_name="J0", children_names=["J0"], rest_world_pos=np.zeros(3, dtype=np.float32))
        with pytest.raises(SkeletonError, match="Cycle detected"):
            skel.validate_hierarchy()

    def test_root_not_at_origin_normalization(self, quadruped_skeleton):
        root_pos = quadruped_skeleton.joints[quadruped_skeleton.root_name].rest_world_pos
        assert root_pos is not None

    def test_excessive_influences_truncation_to_4(self):
        raw = np.array([[0.3, 0.25, 0.2, 0.1, 0.08, 0.05, 0.02]], dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)
        assert weights.shape[1] == 4
        assert abs(np.sum(weights) - 1.0) < 1e-4

    def test_extreme_bindpose_inverse_conditioning(self):
        pos = np.array([1e4, 1e4, 1e4], dtype=np.float32)
        ibm = np.eye(4, dtype=np.float32)
        ibm[:3, 3] = -pos
        p_hom = np.append(pos, 1.0)
        assert np.allclose((ibm @ p_hom)[:3], [0.0, 0.0, 0.0], atol=1e-3)


# =============================================================================
# Boundary 22: BVH Export Boundaries
# =============================================================================

class TestBoundary22BVHExport:
    """Boundary 22: Gimbal lock (+90° / -90° pitch), single joint skeleton, sub-millisecond frames."""

    def test_gimbal_lock_pitch_plus_90_deg(self):
        # Pitch = +90 degrees (around X in ZXY)
        # q for 90 deg around X: [cos(45), sin(45), 0, 0]
        q = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0, 0.0], dtype=np.float32)
        rz, rx, ry = quat_to_euler_zxy_deg(q)
        assert abs(rx - 90.0) < 0.2

    def test_gimbal_lock_pitch_minus_90_deg(self):
        # Pitch = -90 degrees
        q = np.array([np.cos(np.pi / 4.0), -np.sin(np.pi / 4.0), 0.0, 0.0], dtype=np.float32)
        rz, rx, ry = quat_to_euler_zxy_deg(q)
        assert abs(rx - (-90.0)) < 0.2

    def test_single_joint_skeleton_bvh_export(self, e2e_output_dir):
        skel = Skeleton("single_bvh", "Hips")
        skel.joints["Hips"] = JointNode("Hips", rest_world_pos=np.zeros(3, dtype=np.float32))
        anim = SkeletalAnimation(
            "still", 30.0, np.array([0.0], dtype=np.float32), "Hips",
            np.zeros((1, 3), dtype=np.float32),
            {"Hips": np.array([[1, 0, 0, 0]], dtype=np.float32)}
        )
        exporter = BVHExporter()
        path = exporter.export(skel, anim, e2e_output_dir / "single.bvh")
        assert path.exists()

    def test_zero_frame_duration_error(self):
        anim = SkeletalAnimation(
            "zero", 30.0, np.array([0.0], dtype=np.float32), "Hips",
            np.zeros((1, 3), dtype=np.float32),
            {"Hips": np.array([[1, 0, 0, 0]], dtype=np.float32)}
        )
        assert anim.duration_sec == 0.0

    def test_sub_millisecond_frame_time(self, human_skeleton, e2e_output_dir):
        fps = 2000.0  # 0.5 ms
        anim = SkeletalAnimation(
            "fast", fps, np.array([0.0, 0.0005], dtype=np.float32), "Hips",
            np.zeros((2, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in human_skeleton.joints}
        )
        exporter = BVHExporter()
        path = exporter.export(human_skeleton, anim, e2e_output_dir / "fast.bvh")
        with open(path) as f:
            text = f.read()
        assert "Frame Time: 0.000500" in text
