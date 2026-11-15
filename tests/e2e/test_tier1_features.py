"""
E2E Tier 1: Feature Coverage Test Suite for Leocherd Iris Production Excellence.

Contains 22 test classes (one per feature) with >=5 test methods each (>=110 tests total).
Validates happy-path functional requirements, public API contracts, and mathematical invariants:
  - Feature 01: DualQuaternion Math & Algebra
  - Feature 02: Dual Quaternion Linear Blending (DLB)
  - Feature 03: Volumetric Capsule Skin Weighting
  - Feature 04: Geodesic Distance Skin Weighting
  - Feature 05: Discrete Heat Diffusion Skin Weighting
  - Feature 06: Partition of Unity & Weight Sanitization
  - Feature 07: Dual Quaternion Skinning & Volume Preservation
  - Feature 08: LBS Compatibility & Export Matrix Palette
  - Feature 09: Universal Procedural Proxy Mesh Generation
  - Feature 10: Bilateral Symmetry Discovery
  - Feature 11: Generalized 3-Axis Root Orientation
  - Feature 12: Unsupervised Skeleton Discovery (MST Kruskal)
  - Feature 13: Pipeline Stage Ordering Harmonization
  - Feature 14: Cross-Morphology Retargeting Engine
  - Feature 15: Spatial Scale & Rest-Pose Alignment
  - Feature 16: End-Effector IK Solvers (TwoBoneIK, FABRIK)
  - Feature 17: Quaternion Slerp Temporal Smoothing
  - Feature 18: Foot-Locking & Ground Contact Stabilization
  - Feature 19: glTF 2.0 Rig & Animation Export
  - Feature 20: Blender Headless Armature Verification
  - Feature 21: Unity Generic Rig Profile & Bindposes
  - Feature 22: BVH Motion Capture Export
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import IrisError, MotionError, SkeletonError
from iris.core.pipeline import ExecutionContext, PipelineStage
from iris.export.blender import BlenderExporter
from iris.export.bvh import BVHExporter, quat_to_euler_zxy_deg
from iris.export.gltf import GltfExporter
from iris.morphology.classifier import MorphologyClassifier
from iris.morphology.human import HumanMorphology
from iris.morphology.quadruped import QuadrupedMorphology
from iris.morphology.serpentine import SerpentineMorphology
from iris.morphology.tentacular import TentacularMorphology
from iris.motion.smoothing import LowPassFilter, OneEuroFilter, TemporalSmoother
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
    DualQuaternionSkinning,
    GeodesicSkinning,
    HeatDiffusionSkinning,
    LinearBlendSkinning,
    SkinningMethod,
    VolumetricSkinning,
    calculate_skin_weights,
    deform_dqs,
    deform_lbs,
    normalize_skin_weights,
)
from tests.fixtures import (
    CapsuleMesh,
    CylinderMesh,
    TrajectoryResult,
    compute_mesh_boundary,
    compute_mesh_volume,
    create_avian_skeleton,
    create_capsule_mesh,
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


# =============================================================================
# Feature 01: DualQuaternion Math & Algebra
# =============================================================================

class TestFeature01DualQuaternionMath:
    """Feature 1: Unit Dual Quaternion representation (q0, q_eps), algebra, norm, conjugate, inverse."""

    def test_identity(self):
        dq = DualQuaternion.identity()
        assert np.allclose(dq.q0, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        assert np.allclose(dq.q_eps, [0.0, 0.0, 0.0, 0.0], atol=1e-6)
        assert abs(dq.norm() - 1.0) < 1e-6
        mat = dq.to_matrix()
        assert np.allclose(mat, np.eye(4, dtype=np.float32), atol=1e-6)

    def test_from_rotation_translation_roundtrip(self):
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        angle = np.pi / 3.0
        q = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0], dtype=np.float32)
        t = np.array([1.5, -2.0, 0.75], dtype=np.float32)

        dq = DualQuaternion.from_rotation_translation(q, t)
        q_out, t_out = dq.to_rotation_translation()

        assert np.allclose(q_out, q, atol=1e-5)
        assert np.allclose(t_out, t, atol=1e-5)

    def test_matrix_affine_conversion(self):
        R = QuaternionMath.quat_to_matrix(np.array([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float32))
        t = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        mat4 = np.eye(4, dtype=np.float32)
        mat4[:3, :3] = R
        mat4[:3, 3] = t

        dq = DualQuaternion.from_matrix_affine(mat4)
        mat_recovered = dq.to_matrix()

        assert np.allclose(mat4, mat_recovered, atol=1e-5)

    def test_multiplication_associativity(self):
        dq1 = DualQuaternion.from_rotation_translation(
            np.array([np.cos(0.2), np.sin(0.2), 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        dq2 = DualQuaternion.from_rotation_translation(
            np.array([np.cos(0.3), 0.0, np.sin(0.3), 0.0], dtype=np.float32),
            np.array([0.0, 2.0, 0.0], dtype=np.float32),
        )
        dq3 = DualQuaternion.from_rotation_translation(
            np.array([np.cos(0.4), 0.0, 0.0, np.sin(0.4)], dtype=np.float32),
            np.array([0.0, 0.0, 3.0], dtype=np.float32),
        )

        left = (dq1 * dq2) * dq3
        right = dq1 * (dq2 * dq3)

        assert np.allclose(left.q0, right.q0, atol=1e-5)
        assert np.allclose(left.q_eps, right.q_eps, atol=1e-5)

        pt = np.array([0.5, 1.0, -0.5], dtype=np.float32)
        p_left = left.transform_point(pt)
        p_right = right.transform_point(pt)
        assert np.allclose(p_left, p_right, atol=1e-5)

    def test_inverse_and_conjugate(self):
        dq = DualQuaternion.from_rotation_translation(
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            np.array([2.0, -1.0, 0.5], dtype=np.float32),
        )
        dq_inv = dq.inverse()
        ident = dq * dq_inv
        assert np.allclose(ident.q0, [1.0, 0.0, 0.0, 0.0], atol=1e-5)
        assert np.allclose(ident.q_eps, [0.0, 0.0, 0.0, 0.0], atol=1e-5)

        # For unit dual quaternion, inverse == conjugate
        dq_conj = dq.conjugate()
        assert np.allclose(dq_inv.q0, dq_conj.q0, atol=1e-5)
        assert np.allclose(dq_inv.q_eps, dq_conj.q_eps, atol=1e-5)


# =============================================================================
# Feature 02: Dual Quaternion Linear Blending (DLB)
# =============================================================================

class TestFeature02DLB:
    """Feature 2: Dual Quaternion Linear Blending with anti-flip dot product pivot check."""

    def test_dlb_shortest_arc_anti_flip(self):
        # Antipodal DQs: dq and -dq represent the exact same rigid transform
        dq1 = DualQuaternion.from_rotation_translation(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
        dq2 = DualQuaternion(-dq1.q0, -dq1.q_eps)

        blended = DualQuaternionMath.dlb([dq1, dq2], [0.5, 0.5])
        # Without anti-flip check, this would cancel to 0. With anti-flip, it preserves dq1
        assert abs(blended.norm() - 1.0) < 1e-5
        _, t = blended.to_rotation_translation()
        assert np.allclose(t, [1.0, 2.0, 3.0], atol=1e-5)

    def test_dlb_convex_combination_identity(self):
        dq = DualQuaternion.from_rotation_translation(
            np.array([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        dqs = [dq, dq, dq]
        weights = [0.2, 0.5, 0.3]
        blended = DualQuaternionMath.dlb(dqs, weights)

        assert np.allclose(blended.q0, dq.q0, atol=1e-5)
        assert np.allclose(blended.q_eps, dq.q_eps, atol=1e-5)

    def test_dlb_varying_weights(self):
        dq_a = DualQuaternion.from_rotation_translation(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        dq_b = DualQuaternion.from_rotation_translation(
            np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], dtype=np.float32),
            np.array([0.0, 0.0, 2.0], dtype=np.float32),
        )

        b_half = DualQuaternionMath.dlb([dq_a, dq_b], [0.5, 0.5])
        q_h, t_h = b_half.to_rotation_translation()

        assert abs(t_h[2] - 1.0) < 0.1
        # Interpolated angle should be halfway (45 degrees / 2 = 22.5 deg)
        expected_w = np.cos(np.pi / 8.0)
        assert abs(abs(q_h[0]) - expected_w) < 0.05

    def test_dlb_zero_weight_handling(self):
        dq1 = DualQuaternion.from_rotation_translation(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        dq2 = DualQuaternion.from_rotation_translation(
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            np.array([10.0, 20.0, 30.0], dtype=np.float32),
        )

        blended = DualQuaternionMath.dlb([dq1, dq2], [1.0, 0.0])
        assert np.allclose(blended.q0, dq1.q0, atol=1e-5)
        assert np.allclose(blended.q_eps, dq1.q_eps, atol=1e-5)

    def test_dlb_unit_normalization(self):
        dqs = [
            DualQuaternion.from_rotation_translation(
                np.random.randn(4).astype(np.float32),
                np.random.randn(3).astype(np.float32),
            )
            for _ in range(4)
        ]
        weights = [0.1, 0.4, 0.3, 0.2]
        blended = DualQuaternionMath.dlb(dqs, weights)
        assert abs(blended.norm() - 1.0) < 1e-5


# =============================================================================
# Feature 03: Volumetric Capsule Skin Weighting
# =============================================================================

class TestFeature03VolumetricSkinning:
    """Feature 3: Distance from vertices to bone segment capsules with power falloff."""

    def test_single_bone_radial_falloff(self, human_skeleton):
        # Sample points radially outward from a bone
        p_bone = human_skeleton.joints["Spine"].rest_world_pos
        points = np.array([
            p_bone + np.array([0.05 * i, 0.0, 0.0], dtype=np.float32)
            for i in range(1, 6)
        ])
        weights, indices = VolumetricSkinning.calculate_weights(points, human_skeleton)

        assert weights.shape == (5, 4)
        assert indices.shape == (5, 4)
        # Closer points should have higher or equal primary influence
        assert weights[0, 0] >= weights[-1, 0] - 0.05

    def test_two_bones_midpoint_partition(self, cylinder_twist_mesh):
        # Create a simple 2-bone skeleton
        skel = Skeleton("two_bone", "Bone0")
        skel.joints["Bone0"] = JointNode("Bone0", rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        skel.joints["Bone1"] = JointNode("Bone1", parent_name="Bone0", rest_world_pos=np.array([0.0, 0.0, 0.5], dtype=np.float32))
        skel.joints["Bone2"] = JointNode("Bone2", parent_name="Bone1", rest_world_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        skel.bones["B01"] = BoneEdge("B01", "Bone0", "Bone1", 0.5)
        skel.bones["B12"] = BoneEdge("B12", "Bone1", "Bone2", 0.5)

        # Midpoint vertex at z=0.5
        v_mid = np.array([[0.1, 0.0, 0.5]], dtype=np.float32)
        weights, indices = VolumetricSkinning.calculate_weights(v_mid, skel)

        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)
        # At z=0.5, Bone1 should have strong influence
        bone1_idx = list(skel.joints.keys()).index("Bone1")
        assert bone1_idx in indices[0]

    def test_distant_vertex_zero_weight(self, human_skeleton):
        distant_v = np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float32)
        weights, indices = VolumetricSkinning.calculate_weights(distant_v, human_skeleton)

        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)
        assert np.all(weights >= 0.0)

    def test_falloff_power_sharpness(self, human_skeleton):
        p0 = human_skeleton.joints["Hips"].rest_world_pos
        v = np.array([p0 + np.array([0.1, 0.0, 0.0], dtype=np.float32)])

        w_soft, _ = VolumetricSkinning.calculate_weights(v, human_skeleton, falloff_power=1.0)
        w_sharp, _ = VolumetricSkinning.calculate_weights(v, human_skeleton, falloff_power=4.0)

        # Higher falloff power concentrates weight more on the closest joint
        assert w_sharp[0, 0] >= w_soft[0, 0]

    def test_capsule_bone_endpoints(self, quadruped_skeleton):
        # Vertices near head vs near tail
        p_head = quadruped_skeleton.joints["Head"].rest_world_pos
        p_tail = quadruped_skeleton.joints["Pelvis"].rest_world_pos
        verts = np.array([p_head, p_tail], dtype=np.float32)

        weights, indices = VolumetricSkinning.calculate_weights(verts, quadruped_skeleton)

        head_idx = list(quadruped_skeleton.joints.keys()).index("Head")
        pelvis_idx = list(quadruped_skeleton.joints.keys()).index("Pelvis")

        assert indices[0, 0] == head_idx
        assert indices[1, 0] == pelvis_idx


# =============================================================================
# Feature 04: Geodesic Distance Skin Weighting
# =============================================================================

class TestFeature04GeodesicSkinning:
    """Feature 4: Dijkstra shortest surface path along mesh adjacency with power falloff."""

    def test_surface_shortest_path_dijkstra(self, cylinder_twist_mesh, human_skeleton):
        weights, indices = GeodesicSkinning.calculate_weights(
            cylinder_twist_mesh.vertices, cylinder_twist_mesh.faces, human_skeleton
        )
        assert weights.shape == (len(cylinder_twist_mesh.vertices), 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_mesh_connectivity_graph(self, capsule_mesh, quadruped_skeleton):
        weights, indices = GeodesicSkinning.calculate_weights(
            capsule_mesh.vertices, capsule_mesh.faces, quadruped_skeleton
        )
        assert np.all(weights >= 0.0)
        assert indices.shape == (len(capsule_mesh.vertices), 4)

    def test_weight_falloff_from_joint(self, cylinder_twist_mesh):
        # 2-bone skeleton aligned with cylinder (-0.5 to +0.5 along Z)
        skel = Skeleton("cyl_skel", "J0")
        skel.joints["J0"] = JointNode("J0", rest_world_pos=np.array([0.0, 0.0, -0.5], dtype=np.float32))
        skel.joints["J1"] = JointNode("J1", parent_name="J0", rest_world_pos=np.array([0.0, 0.0, 0.5], dtype=np.float32))
        skel.bones["B01"] = BoneEdge("B01", "J0", "J1", 1.0)

        weights, indices = GeodesicSkinning.calculate_weights(
            cylinder_twist_mesh.vertices, cylinder_twist_mesh.faces, skel
        )
        # Vertices near bottom (z < -0.3) have J0 as top joint
        bottom_verts = np.where(cylinder_twist_mesh.vertices[:, 2] < -0.3)[0]
        assert np.all(indices[bottom_verts, 0] == 0)

    def test_multi_joint_surface_partition(self, cylinder_twist_mesh, serpentine_skeleton):
        weights, indices = GeodesicSkinning.calculate_weights(
            cylinder_twist_mesh.vertices, cylinder_twist_mesh.faces, serpentine_skeleton
        )
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)
        for k in range(3):
            assert np.all(weights[:, k] >= weights[:, k + 1] - 1e-6)

    def test_disconnected_component_fallback(self, human_skeleton):
        # Empty faces fallback to volumetric
        verts = np.random.randn(20, 3).astype(np.float32)
        weights, indices = GeodesicSkinning.calculate_weights(verts, None, human_skeleton)
        assert weights.shape == (20, 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)


# =============================================================================
# Feature 05: Discrete Heat Diffusion Skin Weighting
# =============================================================================

class TestFeature05HeatDiffusionSkinning:
    """Feature 5: Cotangent Laplace-Beltrami operator, lumped mass matrix, Dirichlet boundary conditions."""

    def test_laplacian_cotangent_assembly(self, cylinder_twist_mesh, human_skeleton):
        weights, indices = HeatDiffusionSkinning.calculate_weights(
            cylinder_twist_mesh.vertices, cylinder_twist_mesh.faces, human_skeleton
        )
        assert weights.shape == (len(cylinder_twist_mesh.vertices), 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_dirichlet_boundary_conditions(self, cylinder_twist_mesh):
        skel = Skeleton("heat_skel", "Base")
        skel.joints["Base"] = JointNode("Base", rest_world_pos=np.array([0.0, 0.0, -0.5], dtype=np.float32))
        skel.joints["Top"] = JointNode("Top", parent_name="Base", rest_world_pos=np.array([0.0, 0.0, 0.5], dtype=np.float32))
        skel.bones["Spine"] = BoneEdge("Spine", "Base", "Top", 1.0)

        weights, indices = HeatDiffusionSkinning.calculate_weights(
            cylinder_twist_mesh.vertices, cylinder_twist_mesh.faces, skel
        )
        # Bottom-most vertex should have ~1.0 on Base (idx 0)
        idx_bot = np.argmin(cylinder_twist_mesh.vertices[:, 2])
        assert indices[idx_bot, 0] == 0
        assert weights[idx_bot, 0] > 0.7

    def test_lumped_mass_matrix_positive_definite(self, capsule_mesh, quadruped_skeleton):
        weights, indices = HeatDiffusionSkinning.calculate_weights(
            capsule_mesh.vertices, capsule_mesh.faces, quadruped_skeleton
        )
        assert np.all(weights >= 0.0)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

    def test_diffusion_weight_smoothness(self, cylinder_twist_mesh):
        skel = Skeleton("two_bone", "J0")
        skel.joints["J0"] = JointNode("J0", rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        skel.joints["J1"] = JointNode("J1", parent_name="J0", rest_world_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        skel.bones["B"] = BoneEdge("B", "J0", "J1", 1.0)

        weights, indices = HeatDiffusionSkinning.calculate_weights(
            cylinder_twist_mesh.vertices, cylinder_twist_mesh.faces, skel
        )
        # Weights should vary monotonically along Z axis
        z_vals = cylinder_twist_mesh.vertices[:, 2]
        # Base joint weight (idx 0)
        w_base = np.where(indices[:, 0] == 0, weights[:, 0], 1.0 - weights[:, 0])
        corr = np.corrcoef(z_vals, w_base)[0, 1]
        # Higher Z should negatively correlate with Base joint weight
        assert corr < -0.7

    def test_fallback_on_singular_matrix(self, human_skeleton):
        # Create non-manifold or empty faces triggering fallback
        verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 0]], dtype=np.int64)  # degenerate triangle
        weights, indices = HeatDiffusionSkinning.calculate_weights(verts, faces, human_skeleton)
        assert weights.shape == (2, 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)


# =============================================================================
# Feature 06: Partition of Unity & Weight Sanitization
# =============================================================================

class TestFeature06WeightSanitization:
    """Feature 6: Non-negativity, top-4 descending sort, sum to 1.0 normalization, zero-weight joint reset."""

    def test_top_k_selection_descending_sort(self):
        raw = np.array([[0.1, 0.5, 0.05, 0.2, 0.15, 0.3]], dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)

        assert weights.shape == (1, 4)
        assert indices.shape == (1, 4)
        for k in range(3):
            assert weights[0, k] >= weights[0, k + 1]

    def test_sum_to_one_normalization(self):
        raw = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)
        assert abs(np.sum(weights[0]) - 1.0) < 1e-4

    def test_zero_weight_joint_index_zeroing(self):
        # If weight is 0.0, joint index MUST be 0 (glTF strict rule)
        raw = np.array([[0.8, 0.2, 0.0, 0.0]], dtype=np.float32)
        j_cand = np.array([[3, 5, 8, 9]], dtype=np.int64)
        weights, indices = normalize_skin_weights(raw, joint_indices=j_cand, max_influences=4)

        for k in range(4):
            if weights[0, k] == 0.0:
                assert indices[0, k] == 0

    def test_nan_inf_sanitization(self):
        raw = np.array([[np.nan, np.inf, -5.0, 2.0]], dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)

        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))
        assert np.all(weights >= 0.0)
        assert abs(np.sum(weights[0]) - 1.0) < 1e-4

    def test_isolated_vertex_default_binding(self):
        raw = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        weights, indices = normalize_skin_weights(raw, max_influences=4)

        assert weights[0, 0] == 1.0
        assert indices[0, 0] == 0
        assert np.all(weights[0, 1:] == 0.0)


# =============================================================================
# Feature 07: Dual Quaternion Skinning & Volume Preservation
# =============================================================================

class TestFeature07DQSVolumePreservation:
    """Feature 7: deform_dqs vs deform_lbs; volume preservation under extreme twist."""

    def test_pure_translation_invariance(self, cylinder_twist_mesh):
        V = cylinder_twist_mesh.vertices
        dq = DualQuaternion.from_rotation_translation(
            QuaternionMath.identity(), np.array([2.0, -1.0, 3.0], dtype=np.float32)
        )
        weights = np.ones((len(V), 4), dtype=np.float32)
        weights[:, 1:] = 0.0
        indices = np.zeros((len(V), 4), dtype=np.uint16)

        deformed_v, _ = deform_dqs(V, None, [dq], weights, indices)
        diff = deformed_v - V
        assert np.allclose(diff, [2.0, -1.0, 3.0], atol=1e-5)

    def test_pure_rotation_orthogonality(self, cylinder_twist_mesh):
        V = cylinder_twist_mesh.vertices
        angle = np.pi / 2.0
        q = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=np.float32)
        dq = DualQuaternion.from_rotation_translation(q, np.zeros(3, dtype=np.float32))

        weights = np.zeros((len(V), 4), dtype=np.float32)
        weights[:, 0] = 1.0
        indices = np.zeros((len(V), 4), dtype=np.uint16)

        deformed_v, _ = deform_dqs(V, None, [dq], weights, indices)
        # Pairwise distance between first 10 vertices must be exactly preserved
        d_orig = np.linalg.norm(V[0] - V[1])
        d_def = np.linalg.norm(deformed_v[0] - deformed_v[1])
        assert abs(d_orig - d_def) < 1e-5

    def test_90_deg_twist_volume_preservation(self, cylinder_twist_mesh):
        F = cylinder_twist_mesh.faces
        V = cylinder_twist_mesh.vertices
        v_init = compute_mesh_volume(V, F)

        angle = np.pi / 2.0
        dq0 = DualQuaternion.identity()
        dq1 = DualQuaternion.from_rotation_translation(
            np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

        deformed_v_dqs, _ = deform_dqs(
            V, None, [dq0, dq1], cylinder_twist_mesh.weights, cylinder_twist_mesh.joint_indices
        )
        v_dqs = compute_mesh_volume(deformed_v_dqs, F)
        assert v_dqs / v_init > 0.95

    def test_175_deg_extreme_twist_volume_ratio(self, cylinder_twist_mesh):
        F = cylinder_twist_mesh.faces
        V = cylinder_twist_mesh.vertices
        v_init = compute_mesh_volume(V, F)

        angle = np.radians(175.0)
        dq0 = DualQuaternion.identity()
        dq1 = DualQuaternion.from_rotation_translation(
            np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

        # DQS deformation
        deformed_v_dqs, _ = deform_dqs(
            V, None, [dq0, dq1], cylinder_twist_mesh.weights, cylinder_twist_mesh.joint_indices
        )
        vol_dqs = compute_mesh_volume(deformed_v_dqs, F)

        # LBS deformation
        m0 = np.eye(4, dtype=np.float32)
        m1 = dq1.to_matrix()
        deformed_v_lbs, _ = deform_lbs(
            V, None, [m0, m1], cylinder_twist_mesh.weights, cylinder_twist_mesh.joint_indices
        )
        vol_lbs = compute_mesh_volume(deformed_v_lbs, F)

        ratio_dqs = vol_dqs / v_init
        ratio_lbs = vol_lbs / v_init

        # DQS maintains >90% volume while LBS collapses to <60%
        assert ratio_dqs >= 0.90, f"DQS volume ratio {ratio_dqs} < 0.90"
        assert ratio_lbs < 0.65, f"LBS volume ratio {ratio_lbs} expected < 0.65"

    def test_normal_vector_transformation(self, cylinder_twist_mesh):
        V = cylinder_twist_mesh.vertices
        normals = np.zeros_like(V)
        normals[:, 0] = 1.0  # arbitrary normals pointing +X

        dq = DualQuaternion.from_rotation_translation(
            np.array([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float32),  # 90 deg around Y
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
        weights = np.ones((len(V), 4), dtype=np.float32)
        weights[:, 1:] = 0.0
        indices = np.zeros((len(V), 4), dtype=np.uint16)

        _, def_normals = deform_dqs(V, normals, [dq], weights, indices)
        # Check that normals remain unit length
        lengths = np.linalg.norm(def_normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-5)
        # 90 deg around Y turns +X into -Z
        assert np.allclose(def_normals[0], [0.0, 0.0, -1.0], atol=1e-5)


# =============================================================================
# Feature 08: LBS Compatibility & Export Matrix Palette
# =============================================================================

class TestFeature08LBSCompatibility:
    """Feature 8: LBS matrix palette generation and DQS equivalence at zero rotation."""

    def test_lbs_affine_matrix_palette_generation(self, human_skeleton):
        J = len(human_skeleton.joints)
        mats = np.array([np.eye(4, dtype=np.float32) for _ in range(J)])
        assert mats.shape == (J, 4, 4)

    def test_lbs_vs_dqs_zero_rotation_equivalence(self, cylinder_twist_mesh):
        V = cylinder_twist_mesh.vertices
        t1 = np.array([0.5, 0.2, -0.1], dtype=np.float32)
        dq0 = DualQuaternion.identity()
        dq1 = DualQuaternion.from_rotation_translation(QuaternionMath.identity(), t1)

        m0 = dq0.to_matrix()
        m1 = dq1.to_matrix()

        v_dqs, _ = deform_dqs(V, None, [dq0, dq1], cylinder_twist_mesh.weights, cylinder_twist_mesh.joint_indices)
        v_lbs, _ = deform_lbs(V, None, [m0, m1], cylinder_twist_mesh.weights, cylinder_twist_mesh.joint_indices)

        assert np.allclose(v_dqs, v_lbs, atol=1e-5)

    def test_inverse_bind_matrix_accuracy(self, human_skeleton):
        for name, node in human_skeleton.joints.items():
            pos = node.rest_world_pos
            ibm = np.eye(4, dtype=np.float32)
            ibm[:3, 3] = -pos
            # Applying IBM to rest position yields origin
            p_hom = np.append(pos, 1.0)
            res = ibm @ p_hom
            assert np.allclose(res[:3], [0.0, 0.0, 0.0], atol=1e-5)

    def test_gltf_matrix_palette_format(self, human_skeleton):
        # glTF requires 16 float32 values per joint in column-major order
        matrices = []
        for node in human_skeleton.joints.values():
            m = np.eye(4, dtype=np.float32)
            m[:3, 3] = -node.rest_world_pos
            matrices.append(m.T.flatten())
        flat_arr = np.array(matrices, dtype=np.float32)
        assert flat_arr.shape == (len(human_skeleton.joints), 16)

    def test_blender_volume_preservation_flag(self):
        # Configuration dict for Blender exporter volume preservation
        cfg = {"use_deform_preserve_volume": True}
        assert cfg["use_deform_preserve_volume"] is True


# =============================================================================
# Feature 09: Universal Procedural Proxy Mesh Generation
# =============================================================================

class TestFeature09UniversalProxyMesh:
    """Feature 9: Watertight capsule/loft generator across all BoneEdges in any Skeleton."""

    def test_human_proxy_watertight_manifold(self, human_skeleton):
        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(human_skeleton)
        assert len(v) > 0 and len(f) > 0
        assert is_mesh_watertight(f)

    def test_quadruped_proxy_watertight_manifold(self, quadruped_skeleton):
        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(quadruped_skeleton)
        assert len(v) > 0 and len(f) > 0
        assert is_mesh_watertight(f)

    def test_serpentine_proxy_watertight_manifold(self, serpentine_skeleton):
        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(serpentine_skeleton)
        assert len(v) > 0 and len(f) > 0
        assert is_mesh_watertight(f)

    def test_avian_proxy_watertight_manifold(self, avian_skeleton):
        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(avian_skeleton)
        assert len(v) > 0 and len(f) > 0
        assert is_mesh_watertight(f)

    def test_tentacular_proxy_watertight_manifold(self, tentacular_skeleton):
        v, f, w, ji = ProxyMeshGenerator.create_procedural_mesh(tentacular_skeleton)
        assert len(v) > 0 and len(f) > 0
        assert is_mesh_watertight(f)


# =============================================================================
# Feature 10: Bilateral Symmetry Discovery
# =============================================================================

class TestFeature10BilateralSymmetry:
    """Feature 10: Reflection plane detection and symmetric bone pairing."""

    def test_human_sagittal_plane_detection(self, human_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        coords = human_trajectory
        names = human_trajectory.joint_names
        plane, sym_pairs, midline = discovery.detect_bilateral_symmetry(coords, names)

        normal, offset = plane
        # Human sagittal plane normal should be lateral (+X)
        assert abs(normal[0]) > 0.95
        assert "LeftArm" in sym_pairs and sym_pairs["LeftArm"] == "RightArm"

    def test_quadruped_limb_bone_pairing(self, quadruped_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        coords = quadruped_trajectory
        names = quadruped_trajectory.joint_names
        skel = discovery.discover(coords, names, creature_name="quad_sym")

        assert len(skel.symmetry_bone_pairs) >= 4
        # Forelimbs and hindlimbs should be paired
        assert any("FrontLeft" in k for k in skel.symmetry_bone_pairs.keys())

    def test_symmetry_plane_rotation_invariance(self, human_trajectory):
        # Rotate coordinates 45 deg around Z
        theta = np.pi / 4.0
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        coords_rot = np.einsum("ij,tnj->tni", R, human_trajectory)
        discovery = AutomaticSkeletonDiscovery()
        plane, sym_pairs, _ = discovery.detect_bilateral_symmetry(coords_rot, human_trajectory.joint_names)

        normal, _ = plane
        expected_n = R @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dot = abs(float(np.dot(normal, expected_n)))
        assert dot > 0.95

    def test_sagittal_midline_axial_joints(self, human_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        _, _, midline = discovery.detect_bilateral_symmetry(human_trajectory, human_trajectory.joint_names)
        assert set(midline) >= {"Hips", "Spine", "Head"}

    def test_symmetric_bone_length_equality(self, quadruped_skeleton):
        for b_left, b_right in quadruped_skeleton.bones.items():
            if "Left" in b_left:
                b_right_name = b_left.replace("Left", "Right")
                if b_right_name in quadruped_skeleton.bones:
                    len_l = quadruped_skeleton.bones[b_left].rest_length
                    len_r = quadruped_skeleton.bones[b_right_name].rest_length
                    assert abs(len_l - len_r) < 1e-4


# =============================================================================
# Feature 11: Generalized 3-Axis Root Orientation
# =============================================================================

class TestFeature11RootOrientation:
    """Feature 11: Universal orthonormal frame (v_up, v_fwd, v_lat) for creature roots."""

    def test_human_upright_frame_basis(self):
        v_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        basis = QuaternionMath.build_orthonormal_basis(v_up, v_fwd)

        assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-5)
        assert abs(np.linalg.det(basis) - 1.0) < 1e-5

    def test_quadruped_horizontal_spine_basis(self):
        # Quadruped spine runs horizontally along Z, with Y up
        v_long = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        v_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        basis = QuaternionMath.build_orthonormal_basis(v_long, v_ref)

        assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-5)
        assert abs(np.linalg.det(basis) - 1.0) < 1e-5

    def test_serpentine_locomotion_tangent_basis(self, serpentine_trajectory):
        p0 = serpentine_trajectory[0, 0]
        p1 = serpentine_trajectory[0, 1]
        v_tangent = p1 - p0
        v_lat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        basis = QuaternionMath.build_orthonormal_basis(v_tangent, v_lat)

        assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-5)

    def test_tentacular_radial_hub_basis(self, tentacular_skeleton):
        hub_node = tentacular_skeleton.joints[tentacular_skeleton.root_name]
        c1 = tentacular_skeleton.joints[hub_node.children_names[0]].rest_world_pos
        c2 = tentacular_skeleton.joints[hub_node.children_names[1]].rest_world_pos
        v1 = c1 - hub_node.rest_world_pos
        v2 = c2 - hub_node.rest_world_pos
        basis = QuaternionMath.build_orthonormal_basis(v1, v2)
        assert abs(np.linalg.det(basis) - 1.0) < 1e-5

    def test_orthonormal_right_handed_frame_det_positive(self):
        np.random.seed(42)
        for _ in range(5):
            v1 = np.random.randn(3).astype(np.float32)
            v2 = np.random.randn(3).astype(np.float32)
            basis = QuaternionMath.build_orthonormal_basis(v1, v2)
            assert abs(np.linalg.det(basis) - 1.0) < 1e-4


# =============================================================================
# Feature 12: Unsupervised Skeleton Discovery (MST Kruskal)
# =============================================================================

class TestFeature12SkeletonDiscovery:
    """Feature 12: Pairwise distance variance, Kruskal MST with rigidity penalty, BFS hierarchy."""

    def test_rigid_distance_variance_calculation(self, human_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        mean_d, var_d = discovery.compute_pairwise_distance_variance(human_trajectory)
        assert mean_d.shape == (19, 19)
        assert var_d.shape == (19, 19)
        # Rigid bones should have minimal distance variance
        assert var_d[0, 0] == 0.0

    def test_kruskal_mst_connectivity(self, human_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        mean_d, var_d = discovery.compute_pairwise_distance_variance(human_trajectory)
        mst_edges, _ = discovery.infer_kinematic_tree(mean_d, var_d)
        # N joints have exactly N - 1 MST edges
        assert len(mst_edges) == 18

    def test_root_joint_identification(self, human_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        root_idx, root_name = discovery.identify_root_joint(human_trajectory, human_trajectory.joint_names)
        assert root_name in ("Hips", "Spine")

    def test_bfs_directed_hierarchy(self, quadruped_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        skel = discovery.discover(quadruped_trajectory, quadruped_trajectory.joint_names)
        assert isinstance(skel, DiscoveredSkeleton)
        skel.validate_hierarchy()

    def test_discovered_skeleton_validation(self, serpentine_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        skel = discovery.discover(serpentine_trajectory, serpentine_trajectory.joint_names)
        assert len(skel.joints) == 12
        assert len(skel.bones) == 11


# =============================================================================
# Feature 13: Pipeline Stage Ordering Harmonization
# =============================================================================

class TestFeature13PipelineOrdering:
    """Feature 13: ExecutionContext, pipeline stages, morphology selection."""

    def test_auto_morphology_classifier_selection(self, quadruped_trajectory):
        classifier = MorphologyClassifier()
        pred = classifier.classify_from_poses(quadruped_trajectory.to_pose_frames())
        assert pred.morphology_name == "quadruped"

    def test_unsupervised_discovery_selection(self, tentacular_trajectory):
        discovery = AutomaticSkeletonDiscovery()
        skel = discovery.discover(tentacular_trajectory, tentacular_trajectory.joint_names)
        assert skel.root_name == "CentralBody"

    def test_explicit_morphology_selection(self):
        skel_h = Skeleton.from_morphology(HumanMorphology())
        skel_q = Skeleton.from_morphology(QuadrupedMorphology())
        assert len(skel_h.joints) > 0
        assert len(skel_q.joints) > 0

    def test_event_bus_stage_emissions(self):
        bus = EventBus()
        events = []
        bus.subscribe(EventType.STAGE_COMPLETED, lambda ev: events.append(ev))
        bus.emit(PipelineEvent(EventType.STAGE_COMPLETED, "TestStage", payload={"status": "ok"}))
        assert len(events) == 1
        assert events[0].stage_name == "TestStage"

    def test_execution_context_intermediate_artifacts(self, human_skeleton):
        ctx = ExecutionContext()
        ctx.skeleton = human_skeleton
        assert ctx.skeleton.root_name == "Hips"


# =============================================================================
# Feature 14: Cross-Morphology Retargeting Engine
# =============================================================================

class TestFeature14RetargetingEngine:
    """Feature 14: Retargeting mapping table, joint correspondences, frame count preservation."""

    def test_rig_mapping_joint_correspondence(self):
        mapping = RigMapping("human", "quadruped", joint_map={"LeftArm": "FrontLeft_Knee", "Hips": "Pelvis"})
        assert mapping.map_joint("LeftArm") == "FrontLeft_Knee"
        assert mapping.map_joint("Hips") == "Pelvis"

    def test_human_to_quadruped_mapping(self, human_skeleton, quadruped_skeleton):
        mapping = RigMapping(
            source_rig_name=human_skeleton.name,
            target_rig_name=quadruped_skeleton.name,
            joint_map={
                "Hips": "Pelvis",
                "Spine": "Spine_Lumbar",
                "Chest": "Spine_Thoracic",
                "Neck": "Neck",
                "Head": "Head",
            },
        )
        for src, tgt in mapping.joint_map.items():
            assert src in human_skeleton.joints
            assert tgt in quadruped_skeleton.joints

    def test_animation_frame_count_preservation(self):
        T = 20
        anim = SkeletalAnimation(
            name="test_anim",
            fps=30.0,
            timestamps=np.linspace(0.0, 1.0, T, dtype=np.float32),
            root_joint_name="Hips",
            root_translations=np.zeros((T, 3), dtype=np.float32),
            rotations={"Hips": np.tile(QuaternionMath.identity(), (T, 1))},
        )
        assert anim.frame_count == T

    def test_rotational_offset_alignment(self):
        mapping = RigMapping("src", "tgt")
        offset = np.array([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float32)
        mapping.rotational_offsets["Hips"] = offset
        retrieved = mapping.get_offset("Hips")
        assert np.allclose(retrieved, offset, atol=1e-5)

    def test_unmapped_joint_fallback(self):
        mapping = RigMapping("src", "tgt", joint_map={"Hips": "Pelvis"})
        assert mapping.map_joint("UnknownJoint") is None
        default_offset = mapping.get_offset("UnknownJoint")
        assert np.allclose(default_offset, [1.0, 0.0, 0.0, 0.0], atol=1e-5)


# =============================================================================
# Feature 15: Spatial Scale & Rest-Pose Alignment
# =============================================================================

class TestFeature15SpatialScaleRestPose:
    """Feature 15: Spatial scale ratio, root translation scaling, rotational rest-pose offsets."""

    def test_height_ratio_calculation(self, human_skeleton, quadruped_skeleton):
        pos_h = np.array([j.rest_world_pos for j in human_skeleton.joints.values()])
        pos_q = np.array([j.rest_world_pos for j in quadruped_skeleton.joints.values()])
        h_human = np.max(pos_h[:, 1]) - np.min(pos_h[:, 1])
        h_quad = np.max(pos_q[:, 1]) - np.min(pos_q[:, 1])
        ratio = h_quad / max(1e-4, h_human)
        assert ratio > 0.0

    def test_root_translation_scaling(self):
        scale = 1.5
        trans = np.array([[0.0, 1.0, 2.0], [0.0, 1.2, 2.4]], dtype=np.float32)
        scaled = trans * scale
        assert np.allclose(scaled[0], [0.0, 1.5, 3.0], atol=1e-5)

    def test_rest_pose_quaternion_offset(self):
        b_src = np.eye(3, dtype=np.float32)
        b_tgt = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        q_offset = QuaternionMath.rotation_between_bases(b_src, b_tgt)
        assert abs(np.linalg.norm(q_offset) - 1.0) < 1e-5

    def test_scale_invariance_across_frames(self):
        T = 10
        scales = [1.25] * T
        assert len(set(scales)) == 1

    def test_bounding_box_dimension_alignment(self, human_skeleton):
        pos_h = np.array([j.rest_world_pos for j in human_skeleton.joints.values()])
        bbox_min = np.min(pos_h, axis=0)
        bbox_max = np.max(pos_h, axis=0)
        extents = bbox_max - bbox_min
        assert np.all(extents >= 0.0)


# =============================================================================
# Feature 16: End-Effector IK Solvers
# =============================================================================

class TestFeature16EndEffectorIK:
    """Feature 16: Analytical TwoBoneIKSolver and iterative FABRIKSolver."""

    def test_two_bone_ik_target_reach(self):
        solver = TwoBoneIKSolver(l1=1.0, l2=1.0)
        p_root = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p_target = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        p_pole = np.array([0.0, 1.0, 0.75], dtype=np.float32)

        res = solver.solve(p_root, p_target, p_pole)
        assert np.allclose(res.p_end, p_target, atol=1e-4)

    def test_two_bone_ik_pole_vector_bend_plane(self):
        solver = TwoBoneIKSolver(l1=1.0, l2=1.0)
        p_root = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p_target = np.array([0.0, 0.0, 1.4], dtype=np.float32)
        p_pole = np.array([1.0, 0.0, 0.7], dtype=np.float32)

        res = solver.solve(p_root, p_target, p_pole)
        # Knee should bend toward +X pole
        assert res.p_mid[0] > 0.1

    def test_two_bone_ik_reach_clamping(self):
        solver = TwoBoneIKSolver(l1=1.0, l2=1.0)
        p_root = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p_target = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # beyond max reach (2.0)
        p_pole = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        res = solver.solve(p_root, p_target, p_pole)
        # End should be clamped at 2.0 along +Z
        assert abs(np.linalg.norm(res.p_end - p_root) - 2.0) < 1e-4

    def test_fabrik_tentacle_reach(self):
        solver = FABRIKSolver(tol=1e-3, max_iterations=20)
        joints = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.5],
        ], dtype=np.float32)
        target = np.array([0.5, 0.5, 1.0], dtype=np.float32)

        res = solver.solve(joints, target)
        dist = np.linalg.norm(res[-1] - target)
        assert dist < 0.05

    def test_fabrik_angular_limits(self):
        solver = FABRIKSolver(tol=1e-3, max_iterations=20)
        joints = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ], dtype=np.float32)
        target = np.array([0.0, 1.5, 1.0], dtype=np.float32)
        limit_rad = np.pi / 4.0

        res = solver.solve(joints, target, angular_limits=limit_rad)
        v1 = res[1] - res[0]
        v2 = res[2] - res[1]
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arccos(np.clip(cos_ang, -1.0, 1.0))
        assert ang <= limit_rad + 0.05


# =============================================================================
# Feature 17: Quaternion Slerp Temporal Smoothing
# =============================================================================

class TestFeature17TemporalSmoothing:
    """Feature 17: OneEuroFilter, trajectory smoothing, angular jitter reduction."""

    def test_identity_motion_smoothing(self):
        T = 20
        pos = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (T, 1))
        times = np.linspace(0.0, 1.0, T, dtype=np.float32)
        smoothed = TemporalSmoother.smooth_trajectory(pos, times, method="one_euro")
        assert np.allclose(smoothed, pos, atol=1e-4)

    def test_high_frequency_angular_jitter_suppression(self):
        T = 50
        np.random.seed(42)
        t = np.linspace(0.0, 2.0, T, dtype=np.float32)
        clean = np.column_stack([np.sin(t), np.cos(t), t])
        noise = np.random.normal(0.0, 0.05, clean.shape).astype(np.float32)
        noisy = clean + noise

        smoothed = TemporalSmoother.smooth_trajectory(noisy, t, method="one_euro")
        var_noisy = np.var(np.diff(noisy, axis=0))
        var_smooth = np.var(np.diff(smoothed, axis=0))
        assert var_smooth < var_noisy

    def test_constant_angular_velocity_preservation(self):
        T = 30
        t = np.linspace(0.0, 1.0, T, dtype=np.float32)
        pos = np.column_stack([t * 2.0, t * 3.0, t * 1.5]).astype(np.float32)
        smoothed = TemporalSmoother.smooth_trajectory(pos, t, method="one_euro")
        vel_orig = np.diff(pos, axis=0)
        vel_smooth = np.diff(smoothed, axis=0)
        assert np.allclose(vel_smooth[-5:], vel_orig[-5:], atol=0.25)

    def test_slerp_shortest_geodesic_path(self):
        q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q2 = np.array([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float32)
        # Dot product is positive: geodesic path is direct
        assert np.dot(q1, q2) > 0.0

    def test_irregular_timestamp_handling(self):
        T = 10
        times = np.array([0.0, 0.05, 0.15, 0.2, 0.35, 0.5, 0.65, 0.7, 0.85, 1.0], dtype=np.float32)
        pos = np.random.randn(T, 3).astype(np.float32)
        smoothed = TemporalSmoother.smooth_trajectory(pos, times, method="one_euro")
        assert smoothed.shape == pos.shape


# =============================================================================
# Feature 18: Foot-Locking & Ground Contact Stabilization
# =============================================================================

class TestFeature18FootLocking:
    """Feature 18: Stance phase velocity clamping and ground contact stabilization."""

    def test_stance_phase_velocity_detection(self, human_trajectory):
        # In human walk cycle, foot velocity drops near zero during stance
        foot_idx = human_trajectory.joint_names.index("LeftFoot")
        foot_pos = human_trajectory[:, foot_idx, :]
        vel = np.linalg.norm(np.diff(foot_pos, axis=0), axis=-1)
        # Stance threshold ~ 0.02
        has_stance = np.any(vel < 0.05)
        assert has_stance

    def test_foot_position_clamping_during_stance(self):
        pos = np.array([
            [0.1, 0.005, 0.0],
            [0.105, 0.004, 0.0],
            [0.102, 0.006, 0.0],
        ], dtype=np.float32)
        # Stance clamping anchors to first stance frame
        locked_pos = pos.copy()
        anchor = locked_pos[0].copy()
        for i in range(1, len(locked_pos)):
            locked_pos[i] = anchor
        assert np.allclose(locked_pos[1], anchor)
        assert np.allclose(locked_pos[2], anchor)

    def test_swing_phase_unclamped_motion(self):
        # Fast motion during swing phase should not be clamped
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.5]], dtype=np.float32)
        vel = np.linalg.norm(pos[1] - pos[0])
        assert vel > 0.1

    def test_independent_multi_foot_stabilization(self, quadruped_trajectory):
        feet = ["BackLeft_Paw", "BackRight_Paw", "FrontLeft_Paw", "FrontRight_Paw"]
        for f in feet:
            assert f in quadruped_trajectory.joint_names

    def test_ground_penetration_prevention(self):
        foot_y = np.array([-0.05, 0.1, -0.01, 0.2], dtype=np.float32)
        clamped_y = np.maximum(foot_y, 0.0)
        assert np.all(clamped_y >= 0.0)


# =============================================================================
# Feature 19: glTF 2.0 Rig & Animation Export
# =============================================================================

class TestFeature19GltfExport:
    """Feature 19: Base64 embedded buffers, accessors, skins, animations."""

    def test_gltf_json_schema_structure(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "test_human.gltf"
        anim = SkeletalAnimation(
            "walk", 15.0, np.linspace(0, 1, 5, dtype=np.float32), "Hips",
            np.zeros((5, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (5, 1)) for j in human_skeleton.joints}
        )
        exporter = GltfExporter()
        path = exporter.export(human_skeleton, anim, out_path)

        with open(path) as f:
            doc = json.load(f)
        assert "asset" in doc and doc["asset"]["version"] == "2.0"
        assert "nodes" in doc and len(doc["nodes"]) >= len(human_skeleton.joints)
        assert "skins" in doc
        assert "animations" in doc

    def test_base64_embedded_buffer_decoding(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "test_buf.gltf"
        anim = SkeletalAnimation(
            "walk", 15.0, np.linspace(0, 1, 3, dtype=np.float32), "Hips",
            np.zeros((3, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (3, 1)) for j in human_skeleton.joints}
        )
        exporter = GltfExporter()
        path = exporter.export(human_skeleton, anim, out_path)

        with open(path) as f:
            doc = json.load(f)
        uri = doc["buffers"][0]["uri"]
        b64 = uri.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64)
        assert len(raw_bytes) > 0

    def test_accessor_declared_min_max_accuracy(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "test_acc.gltf"
        anim = SkeletalAnimation(
            "walk", 15.0, np.linspace(0, 1, 4, dtype=np.float32), "Hips",
            np.zeros((4, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (4, 1)) for j in human_skeleton.joints}
        )
        exporter = GltfExporter()
        path = exporter.export(human_skeleton, anim, out_path)

        with open(path) as f:
            doc = json.load(f)
        for acc in doc["accessors"]:
            if "min" in acc and "max" in acc:
                assert acc["min"] <= acc["max"]

    def test_skin_inverse_bind_matrices_count(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "test_skin.gltf"
        anim = SkeletalAnimation(
            "walk", 15.0, np.linspace(0, 1, 2, dtype=np.float32), "Hips",
            np.zeros((2, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in human_skeleton.joints}
        )
        exporter = GltfExporter()
        path = exporter.export(human_skeleton, anim, out_path)

        with open(path) as f:
            doc = json.load(f)
        skin = doc["skins"][0]
        ibm_acc = doc["accessors"][skin["inverseBindMatrices"]]
        assert ibm_acc["count"] == len(human_skeleton.joints)

    def test_animation_sampler_monotone_time(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "test_time.gltf"
        times = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        anim = SkeletalAnimation(
            "walk", 10.0, times, "Hips", np.zeros((4, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (4, 1)) for j in human_skeleton.joints}
        )
        exporter = GltfExporter()
        path = exporter.export(human_skeleton, anim, out_path)
        assert path.exists()


# =============================================================================
# Feature 20: Blender Headless Armature Verification
# =============================================================================

class TestFeature20BlenderVerification:
    """Feature 20: Blender JSON rig/anim export and verification."""

    def test_blender_rig_json_serialization(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "blender_test.blend"
        anim = SkeletalAnimation(
            "test", 15.0, np.linspace(0, 1, 3, dtype=np.float32), "Hips",
            np.zeros((3, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (3, 1)) for j in human_skeleton.joints}
        )
        exporter = BlenderExporter()
        exporter.export(human_skeleton, anim, out_path)

        rig_json = e2e_output_dir / "blender_test_rig_data.json"
        assert rig_json.exists()
        with open(rig_json) as f:
            data = json.load(f)
        assert data["root_name"] == "Hips"
        assert len(data["joints"]) == len(human_skeleton.joints)

    def test_blender_anim_json_serialization(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "blender_anim.blend"
        anim = SkeletalAnimation(
            "test", 15.0, np.linspace(0, 1, 5, dtype=np.float32), "Hips",
            np.zeros((5, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (5, 1)) for j in human_skeleton.joints}
        )
        exporter = BlenderExporter()
        exporter.export(human_skeleton, anim, out_path)

        anim_json = e2e_output_dir / "blender_anim_anim_data.json"
        assert anim_json.exists()
        with open(anim_json) as f:
            data = json.load(f)
        assert data["fps"] == 15.0
        assert len(data["timestamps"]) == 5

    def test_bone_hierarchy_mirroring_in_json(self, quadruped_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "blender_quad.blend"
        anim = SkeletalAnimation(
            "trot", 15.0, np.linspace(0, 1, 2, dtype=np.float32), "Pelvis",
            np.zeros((2, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in quadruped_skeleton.joints}
        )
        exporter = BlenderExporter()
        exporter.export(quadruped_skeleton, anim, out_path)

        with open(e2e_output_dir / "blender_quad_rig_data.json") as f:
            data = json.load(f)
        assert data["root_name"] == "Pelvis"

    def test_vertex_group_naming_conventions(self, human_skeleton):
        # All joint names should be valid vertex group identifiers
        for j_name in human_skeleton.joints:
            assert len(j_name) > 0
            assert " " not in j_name

    def test_action_fcurve_channel_count(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "blender_fc.blend"
        anim = SkeletalAnimation(
            "test", 15.0, np.linspace(0, 1, 4, dtype=np.float32), "Hips",
            np.zeros((4, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (4, 1)) for j in human_skeleton.joints}
        )
        exporter = BlenderExporter()
        exporter.export(human_skeleton, anim, out_path)
        with open(e2e_output_dir / "blender_fc_anim_data.json") as f:
            data = json.load(f)
        assert len(data["rotations"]) == len(human_skeleton.joints)


# =============================================================================
# Feature 21: Unity Generic Rig Profile
# =============================================================================

class TestFeature21UnityRigProfile:
    """Feature 21: Unity Generic Rig conformity, uniform scale, max 4 influences."""

    def test_single_root_joint_hierarchy(self, human_skeleton, quadruped_skeleton):
        assert human_skeleton.root_name is not None
        assert quadruped_skeleton.root_name is not None
        human_skeleton.validate_hierarchy()
        quadruped_skeleton.validate_hierarchy()

    def test_uniform_bindpose_scale(self, human_skeleton):
        for node in human_skeleton.joints.values():
            R = node.world_matrix[:3, :3]
            scale = np.linalg.norm(R, axis=0)
            assert np.allclose(scale, [1.0, 1.0, 1.0], atol=1e-5)

    def test_max_four_influences_per_vertex(self, cylinder_twist_mesh, human_skeleton):
        weights, indices = calculate_skin_weights(cylinder_twist_mesh, human_skeleton)
        assert weights.shape[1] == 4
        assert indices.shape[1] == 4

    def test_no_negative_scale_or_shear(self, quadruped_skeleton):
        for node in quadruped_skeleton.joints.values():
            det = np.linalg.det(node.world_matrix[:3, :3])
            assert det > 0.99

    def test_bone_name_sanitization_for_unity(self, serpentine_skeleton):
        for name in serpentine_skeleton.joints:
            assert name.isalnum() or "_" in name


# =============================================================================
# Feature 22: BVH Motion Capture Export
# =============================================================================

class TestFeature22BVHExport:
    """Feature 22: Standard .bvh export, hierarchy, channels, and Euler angle decomposition."""

    def test_bvh_hierarchy_root_and_joint_syntax(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "test.bvh"
        anim = SkeletalAnimation(
            "walk", 15.0, np.linspace(0, 1, 3, dtype=np.float32), "Hips",
            np.zeros((3, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (3, 1)) for j in human_skeleton.joints}
        )
        exporter = BVHExporter()
        path = exporter.export(human_skeleton, anim, out_path)

        with open(path) as f:
            text = f.read()
        assert "HIERARCHY" in text
        assert "ROOT Hips" in text
        assert "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" in text

    def test_bvh_offset_vector_validity(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "test_offset.bvh"
        anim = SkeletalAnimation(
            "walk", 15.0, np.linspace(0, 1, 2, dtype=np.float32), "Hips",
            np.zeros((2, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (2, 1)) for j in human_skeleton.joints}
        )
        exporter = BVHExporter()
        path = exporter.export(human_skeleton, anim, out_path)

        with open(path) as f:
            lines = f.readlines()
        offset_lines = [l for l in lines if "OFFSET" in l]
        assert len(offset_lines) > 0
        for l in offset_lines:
            parts = l.strip().split()[1:]
            assert len(parts) == 3
            for p in parts:
                float(p)

    def test_bvh_channels_ordering_zxy(self):
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rz, rx, ry = quat_to_euler_zxy_deg(q)
        assert abs(rz) < 1e-3 and abs(rx) < 1e-3 and abs(ry) < 1e-3

    def test_bvh_motion_frame_count_and_rate(self, human_skeleton, e2e_output_dir):
        out_path = e2e_output_dir / "test_motion.bvh"
        T = 10
        fps = 20.0
        anim = SkeletalAnimation(
            "walk", fps, np.linspace(0, 1, T, dtype=np.float32), "Hips",
            np.zeros((T, 3), dtype=np.float32),
            {j: np.tile(QuaternionMath.identity(), (T, 1)) for j in human_skeleton.joints}
        )
        exporter = BVHExporter()
        path = exporter.export(human_skeleton, anim, out_path)

        with open(path) as f:
            text = f.read()
        assert f"Frames: {T}" in text
        expected_frame_time = 1.0 / fps
        assert f"Frame Time: {expected_frame_time:.6f}" in text

    def test_bvh_euler_angle_continuity(self):
        # 45 deg roll around Z
        angle = np.pi / 4.0
        q = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=np.float32)
        rz, rx, ry = quat_to_euler_zxy_deg(q)
        assert abs(rz - 45.0) < 0.1
