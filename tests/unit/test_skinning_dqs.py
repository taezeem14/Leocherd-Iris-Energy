"""
Unit tests for DualQuaternion algebra, Dual Quaternion Linear Blending (DLB),
advanced skin weight calculation (Volumetric, Geodesic, Heat Diffusion),
partition of unity sanitization, and the candy-wrapper volume preservation benchmark.
"""

import math
from typing import Tuple
import numpy as np
import pytest

from iris.rigging.joints import BoneEdge, JointNode
from iris.rigging.orientation import DualQuaternion, DualQuaternionMath, QuaternionMath
from iris.rigging.skeleton import Skeleton
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


def _create_simple_skeleton() -> Skeleton:
    """Creates a 3-joint, 2-bone linear arm skeleton along the Y axis."""
    skel = Skeleton(name="arm", root_name="root")
    j0 = JointNode(name="root", parent_name=None, rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    j1 = JointNode(name="elbow", parent_name="root", rest_world_pos=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    j2 = JointNode(name="wrist", parent_name="elbow", rest_world_pos=np.array([0.0, 2.0, 0.0], dtype=np.float32))

    skel.joints["root"] = j0
    skel.joints["elbow"] = j1
    skel.joints["wrist"] = j2

    skel.bones["upper_arm"] = BoneEdge(name="upper_arm", parent_joint="root", child_joint="elbow", rest_length=1.0)
    skel.bones["forearm"] = BoneEdge(name="forearm", parent_joint="elbow", child_joint="wrist", rest_length=1.0)

    return skel


def _create_cylinder_mesh(
    radius: float = 0.5,
    height: float = 2.0,
    rings: int = 15,
    radial_segments: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a watertight cylindrical mesh centered along Z from -height/2 to +height/2.
    Returns: (vertices, normals, faces)
    """
    verts = []
    normals = []
    z_coords = np.linspace(-height / 2.0, height / 2.0, rings)

    # Lateral vertices and normals
    for z in z_coords:
        for seg in range(radial_segments):
            angle = 2.0 * math.pi * seg / radial_segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            verts.append([radius * cos_a, radius * sin_a, z])
            normals.append([cos_a, sin_a, 0.0])

    verts = np.array(verts, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    # Lateral faces
    faces = []
    for r in range(rings - 1):
        for s in range(radial_segments):
            next_s = (s + 1) % radial_segments
            v0 = r * radial_segments + s
            v1 = r * radial_segments + next_s
            v2 = (r + 1) * radial_segments + s
            v3 = (r + 1) * radial_segments + next_s
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])

    # End caps
    bottom_center_idx = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, -height / 2.0]]])
    normals = np.vstack([normals, [[0.0, 0.0, -1.0]]])

    top_center_idx = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, height / 2.0]]])
    normals = np.vstack([normals, [[0.0, 0.0, 1.0]]])

    for s in range(radial_segments):
        next_s = (s + 1) % radial_segments
        # Bottom cap (outward normal -Z)
        faces.append([bottom_center_idx, next_s, s])
        # Top cap (outward normal +Z)
        top_offset = (rings - 1) * radial_segments
        faces.append([top_center_idx, top_offset + s, top_offset + next_s])

    return verts, normals, np.array(faces, dtype=np.int64)


def _compute_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Computes enclosed signed volume of a watertight mesh via divergence theorem."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1, v2)
    dot = np.sum(v0 * cross, axis=1)
    return float(np.abs(np.sum(dot)) / 6.0)


# =========================================================================
# 1. DualQuaternion Algebra & Math Tests
# =========================================================================

class TestDualQuaternionAlgebra:
    def test_identity(self):
        dq = DualQuaternion.identity()
        assert np.allclose(dq.q0, [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(dq.q_eps, [0.0, 0.0, 0.0, 0.0])
        assert math.isclose(dq.norm(), 1.0, abs_tol=1e-6)

        q, t = dq.to_rotation_translation()
        assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(t, [0.0, 0.0, 0.0])

    def test_rotation_translation_roundtrip(self):
        # 45 degree rotation around Y axis, translated by [1.5, -2.0, 3.5]
        angle = math.radians(45.0)
        q = np.array([math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0], dtype=np.float32)
        t = np.array([1.5, -2.0, 3.5], dtype=np.float32)

        dq = DualQuaternion.from_rotation_translation(q, t)
        q_rec, t_rec = dq.to_rotation_translation()

        assert np.allclose(q, q_rec, atol=1e-5)
        assert np.allclose(t, t_rec, atol=1e-5)

    def test_matrix_affine_roundtrip(self):
        # Create affine transformation matrix
        angle = math.radians(60.0)
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        t = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        mat4 = np.eye(4, dtype=np.float32)
        mat4[:3, :3] = R
        mat4[:3, 3] = t

        dq = DualQuaternion.from_matrix_affine(mat4)
        mat4_rec = dq.to_matrix()

        assert np.allclose(mat4, mat4_rec, atol=1e-5)

    def test_multiplication_associativity_and_inverse(self):
        # Create two rigid transformations
        q1 = QuaternionMath.rotation_between_vectors(np.array([1, 0, 0]), np.array([0, 1, 0]))
        t1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dq1 = DualQuaternion.from_rotation_translation(q1, t1)

        q2 = QuaternionMath.rotation_between_vectors(np.array([0, 1, 0]), np.array([0, 0, 1]))
        t2 = np.array([-2.0, 0.5, 1.0], dtype=np.float32)
        dq2 = DualQuaternion.from_rotation_translation(q2, t2)

        q3 = QuaternionMath.rotation_between_vectors(np.array([0, 0, 1]), np.array([1, 1, 1]))
        t3 = np.array([0.0, -1.0, 2.0], dtype=np.float32)
        dq3 = DualQuaternion.from_rotation_translation(q3, t3)

        # Associativity: (dq1 * dq2) * dq3 == dq1 * (dq2 * dq3)
        res_a = (dq1 * dq2) * dq3
        res_b = dq1 * (dq2 * dq3)
        assert np.allclose(res_a.q0, res_b.q0, atol=1e-5)
        assert np.allclose(res_a.q_eps, res_b.q_eps, atol=1e-5)

        # Multiplicative inverse: dq1 * dq1.inverse() == Identity
        dq1_inv = dq1.inverse()
        identity_test = (dq1 * dq1_inv).normalize()
        assert np.allclose(identity_test.q0, [1.0, 0.0, 0.0, 0.0], atol=1e-5)
        assert np.allclose(identity_test.q_eps, [0.0, 0.0, 0.0, 0.0], atol=1e-5)

    def test_point_and_normal_transformation(self):
        angle = math.radians(90.0)
        # 90 deg around Z: rotates [1, 0, 0] to [0, 1, 0]
        q = np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)], dtype=np.float32)
        t = np.array([5.0, 10.0, 15.0], dtype=np.float32)
        dq = DualQuaternion.from_rotation_translation(q, t)

        point = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Point rotates and translates: [0, 1, 0] + [5, 10, 15] = [5, 11, 15]
        p_trans = dq.transform_point(point)
        assert np.allclose(p_trans, [5.0, 11.0, 15.0], atol=1e-5)

        # Normal ONLY rotates (no translation): [0, 1, 0]
        n_trans = dq.transform_normal(normal)
        assert np.allclose(n_trans, [0.0, 1.0, 0.0], atol=1e-5)


# =========================================================================
# 2. Dual Quaternion Linear Blending (DLB) Tests
# =========================================================================

class TestDualQuaternionLinearBlending:
    def test_dlb_basic_interpolation(self):
        dq0 = DualQuaternion.identity()
        # 90 deg rotation around Y
        angle = math.radians(90.0)
        q1 = np.array([math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0], dtype=np.float32)
        t1 = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        dq1 = DualQuaternion.from_rotation_translation(q1, t1)

        # 50/50 blend -> 45 deg rotation around Y, translation [0, 1, 0]
        blended = DualQuaternionMath.dlb([dq0, dq1], [0.5, 0.5])
        q_blend, t_blend = blended.to_rotation_translation()

        expected_q = np.array([math.cos(math.radians(22.5)), 0.0, math.sin(math.radians(22.5)), 0.0], dtype=np.float32)
        assert np.allclose(q_blend, expected_q, atol=1e-5)
        assert np.allclose(t_blend, [0.0, 1.0, 0.0], atol=1e-5)

    def test_dlb_anti_flip_dot_product_pivot(self):
        # Test that blending with opposite quaternion sign does NOT cause cancellation
        dq0 = DualQuaternion.identity()
        # Same transformation represented with flipped quaternion sign: -q and -q_eps
        angle = math.radians(30.0)
        q1 = np.array([math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0], dtype=np.float32)
        t1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dq1 = DualQuaternion.from_rotation_translation(q1, t1)

        # Artificially negate dq1 components (represents the exact same rigid motion)
        dq1_flipped = DualQuaternion(-dq1.q0, -dq1.q_eps)

        # Without anti-flip, (dq0 + dq1_flipped) would destructively cancel
        blended = DualQuaternionMath.dlb([dq0, dq1_flipped], [0.5, 0.5])
        assert not np.allclose(blended.q0, [0.0, 0.0, 0.0, 0.0])
        assert math.isclose(blended.norm(), 1.0, abs_tol=1e-5)

        # Verify translation is smoothly interpolated
        _, t_blend = blended.to_rotation_translation()
        assert np.allclose(t_blend, [0.5, 0.0, 0.0], atol=1e-4)


# =========================================================================
# 3. Partition of Unity & Weight Sanitization Tests
# =========================================================================

class TestWeightSanitization:
    def test_normalize_skin_weights_strict_contract(self):
        # Raw weights with negatives, non-normalized values, and unsorted orders
        raw = np.array([
            [-0.5, 2.0, 0.0, 1.0, 3.0],
            [0.1, 0.2, 0.3, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],  # All zero edge case
        ], dtype=np.float32)

        weights, indices = normalize_skin_weights(raw, max_influences=4)

        # 1. Output shape (V, 4)
        assert weights.shape == (3, 4)
        assert indices.shape == (3, 4)
        assert indices.dtype == np.uint16

        # 2. Strict non-negativity
        assert np.all(weights >= 0.0)

        # 3. Descending sort per vertex
        for v in range(3):
            for k in range(3):
                assert weights[v, k] >= weights[v, k + 1]

        # 4. Sum to 1.0 +- 1e-4
        sums = np.sum(weights, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-4)

        # 5. All-zero edge case defaults to w_0 = 1.0, j_0 = 0
        assert weights[2, 0] == 1.0
        assert weights[2, 1] == 0.0
        assert indices[2, 0] == 0

        # 6. Zero-weight joint index sanitization (if w == 0.0 => j == 0)
        for v in range(3):
            for k in range(4):
                if weights[v, k] == 0.0:
                    assert indices[v, k] == 0

    def test_normalize_skin_weights_nan_inf_sanitization(self):
        # Raw weights containing NaN, +Inf, -Inf, and all-NaN / all-Inf rows
        raw = np.array([
            [np.nan, 2.0, 1.0, 0.5],             # Mixed NaN and valid finite values
            [np.nan, np.nan, np.nan, np.nan],     # All NaN row (must trigger fallback)
            [np.inf, 2.0, 1.0, 0.5],             # Positive Inf
            [-np.inf, 3.0, 0.0, 1.0],            # Negative Inf
            [np.inf, -np.inf, np.nan, 0.0],      # Combination of Inf, -Inf, NaN, 0.0 (must trigger fallback)
            [1.0, np.nan, -np.inf, 0.0],         # Single finite value + invalid values
        ], dtype=np.float32)

        # Custom joint indices where non-zero indices are assigned to every candidate influence
        cand_indices = np.array([
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
            [50, 51, 52, 53],
            [60, 61, 62, 63],
        ], dtype=np.uint16)

        weights, indices = normalize_skin_weights(raw, joint_indices=cand_indices, max_influences=4)

        # 1. Output shape and dtype
        V = raw.shape[0]
        assert weights.shape == (V, 4)
        assert indices.shape == (V, 4)

        # 2. Absolutely zero NaN or Inf in weights or indices
        assert not np.isnan(weights).any(), "Output weights must not contain NaN"
        assert not np.isinf(weights).any(), "Output weights must not contain Inf"

        # 3. Strict non-negativity
        assert np.all(weights >= 0.0)

        # 4. Descending sort per vertex
        for v in range(V):
            for k in range(3):
                assert weights[v, k] >= weights[v, k + 1]

        # 5. Partition of unity: sum to 1.0 +- 1e-4
        sums = np.sum(weights, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-4)

        # 6. All-NaN row (v=1) and all-invalid row (v=4) must trigger root fallback
        for v_fallback in [1, 4]:
            assert weights[v_fallback, 0] == 1.0
            assert np.all(weights[v_fallback, 1:] == 0.0)
            assert indices[v_fallback, 0] == 0
            assert np.all(indices[v_fallback, 1:] == 0)

        # 7. Zero-weight influence index check: if w[v, k] == 0.0, j[v, k] MUST be 0
        for v in range(V):
            for k in range(4):
                if weights[v, k] == 0.0:
                    assert indices[v, k] == 0, f"Vertex {v} influence {k} has zero weight but joint index {indices[v, k]} != 0"

        # 8. Row 0 specific check: valid values [2.0, 1.0, 0.5] from indices [11, 12, 13]
        # Sum of valid values = 3.5 -> 2/3.5, 1/3.5, 0.5/3.5, 0.0
        assert np.isclose(weights[0, 0], 2.0 / 3.5, atol=1e-4)
        assert indices[0, 0] == 11
        assert np.isclose(weights[0, 1], 1.0 / 3.5, atol=1e-4)
        assert indices[0, 1] == 12
        assert np.isclose(weights[0, 2], 0.5 / 3.5, atol=1e-4)
        assert indices[0, 2] == 13
        assert weights[0, 3] == 0.0
        assert indices[0, 3] == 0


# =========================================================================
# 4. Volumetric, Geodesic, and Heat Diffusion Weight Calculation Tests
# =========================================================================

class TestSkinWeightingEngines:
    def test_volumetric_capsule_skinning(self):
        skel = _create_simple_skeleton()
        # Test vertices along Y axis:
        # v0 near root (0, 0.1, 0)
        # v1 near elbow (0, 1.0, 0)
        # v2 near wrist (0, 1.9, 0)
        verts = np.array([
            [0.0, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.9, 0.0],
        ], dtype=np.float32)

        weights, indices = VolumetricSkinning.calculate_weights(verts, skel, max_influences=4)

        assert weights.shape == (3, 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)

        # Vertex near root should have root (index 0) as primary influence
        assert indices[0, 0] == 0
        # Vertex near wrist should have wrist (index 2) as primary influence
        assert indices[2, 0] == 2

    def test_geodesic_skinning(self):
        skel = _create_simple_skeleton()
        verts, _, faces = _create_cylinder_mesh(radius=0.2, height=2.0, rings=10, radial_segments=8)

        weights, indices = GeodesicSkinning.calculate_weights(verts, faces, skel, max_influences=4)

        assert weights.shape == (len(verts), 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)
        assert np.all(weights >= 0.0)

    def test_heat_diffusion_skinning(self):
        skel = _create_simple_skeleton()
        verts, _, faces = _create_cylinder_mesh(radius=0.2, height=2.0, rings=10, radial_segments=8)

        weights, indices = HeatDiffusionSkinning.calculate_weights(verts, faces, skel, max_influences=4)

        assert weights.shape == (len(verts), 4)
        assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-4)
        assert np.all(weights >= 0.0)

        # Test graceful fallback when faces are None or degenerate
        fallback_w, fallback_j = HeatDiffusionSkinning.calculate_weights(verts, None, skel, max_influences=4)
        assert fallback_w.shape == (len(verts), 4)
        assert np.allclose(np.sum(fallback_w, axis=1), 1.0, atol=1e-4)

    def test_calculate_skin_weights_unified_interface(self):
        skel = _create_simple_skeleton()
        verts, _, faces = _create_cylinder_mesh(radius=0.2, height=2.0, rings=8, radial_segments=6)

        # Test tuple (vertices, faces)
        w_tuple, j_tuple = calculate_skin_weights((verts, faces), skel, method=SkinningMethod.HEAT_DIFFUSION)
        assert w_tuple.shape == (len(verts), 4)
        assert j_tuple.shape == (len(verts), 4)
        assert j_tuple.dtype == np.uint16

        # Test method enum vs string
        w_geo, j_geo = calculate_skin_weights((verts, faces), skel, method="geodesic")
        assert np.allclose(np.sum(w_geo, axis=1), 1.0, atol=1e-4)

        w_vol, j_vol = calculate_skin_weights(verts, skel, method=SkinningMethod.VOLUMETRIC)
        assert np.allclose(np.sum(w_vol, axis=1), 1.0, atol=1e-4)


# =========================================================================
# 5. Candy-Wrapper Volume Preservation Benchmark: DQS vs LBS
# =========================================================================

class TestVolumePreservationBenchmark:
    """
    Validates that Dual Quaternion Skinning (DQS) eliminates the classic 'candy-wrapper'
    volume collapse artifact observed in Linear Blend Skinning (LBS) under extreme joint twists (90° to 175°).
    """

    def test_candy_wrapper_elimination_cylinder_twist(self):
        radius = 0.5
        height = 2.0
        rings = 21
        radial_segments = 24
        verts, normals, faces = _create_cylinder_mesh(radius=radius, height=height, rings=rings, radial_segments=radial_segments)

        # Skin cylinder to two joints:
        # Joint 0 at z = -1.0 (base)
        # Joint 1 at z = +1.0 (top)
        # Weights blend linearly along Z from z = -1.0 to z = +1.0:
        # w1 = (z - (-1)) / 2 = (z + 1) / 2
        # w0 = 1 - w1
        z_vals = verts[:, 2]
        t = np.clip((z_vals + 1.0) / 2.0, 0.0, 1.0)
        w0 = 1.0 - t
        w1 = t

        weights = np.column_stack([w0, w1, np.zeros_like(w0), np.zeros_like(w0)]).astype(np.float32)
        joint_indices = np.zeros((len(verts), 4), dtype=np.uint16)
        joint_indices[:, 1] = 1

        # Re-sanitize with normalize_skin_weights to guarantee strict contract
        weights, joint_indices = normalize_skin_weights(weights, joint_indices, max_influences=4)

        initial_volume = _compute_mesh_volume(verts, faces)
        assert initial_volume > 0.0

        # Test extreme twist angles: 90°, 135°, 175°
        twist_angles_deg = [90.0, 135.0, 175.0]

        # Identify vertices at the exact midpoint ring (z = 0, where w0 == 0.5 and w1 == 0.5)
        midpoint_mask = np.isclose(verts[:, 2], 0.0, atol=0.05)
        assert np.sum(midpoint_mask) >= radial_segments

        for angle_deg in twist_angles_deg:
            angle_rad = math.radians(angle_deg)

            # 1. Setup joint transformations
            # Joint 0: Identity
            # Joint 1: Rotation around Z axis by angle_rad
            c = math.cos(angle_rad)
            s = math.sin(angle_rad)

            # 4x4 Matrices for LBS
            M0 = np.eye(4, dtype=np.float32)
            M1 = np.array([
                [c, -s, 0.0, 0.0],
                [s,  c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=np.float32)
            joint_matrices = np.stack([M0, M1])

            # Dual Quaternions for DQS
            dq0 = DualQuaternion.identity()
            q1 = np.array([math.cos(angle_rad / 2.0), 0.0, 0.0, math.sin(angle_rad / 2.0)], dtype=np.float32)
            dq1 = DualQuaternion.from_rotation_translation(q1, np.zeros(3, dtype=np.float32))
            joint_dqs = [dq0, dq1]

            # 2. Deform mesh
            deformed_lbs_v, deformed_lbs_n = deform_lbs(verts, normals, joint_matrices, weights, joint_indices)
            deformed_dqs_v, deformed_dqs_n = deform_dqs(verts, normals, joint_dqs, weights, joint_indices)

            # 3. Measure midpoint radius (where candy-wrapper pinching occurs)
            mid_lbs_xy = deformed_lbs_v[midpoint_mask, :2]
            mid_lbs_radius = float(np.mean(np.linalg.norm(mid_lbs_xy, axis=1)))

            mid_dqs_xy = deformed_dqs_v[midpoint_mask, :2]
            mid_dqs_radius = float(np.mean(np.linalg.norm(mid_dqs_xy, axis=1)))

            # Theoretical LBS midpoint radius is radius * cos(angle / 2)
            expected_lbs_mid_radius = radius * math.cos(angle_rad / 2.0)
            assert math.isclose(mid_lbs_radius, expected_lbs_mid_radius, rel_tol=0.05)

            # DQS perfectly preserves the midpoint radius:
            assert math.isclose(mid_dqs_radius, radius, rel_tol=0.02)

            # 4. Measure enclosed volume
            vol_lbs = _compute_mesh_volume(deformed_lbs_v, faces)
            vol_dqs = _compute_mesh_volume(deformed_dqs_v, faces)

            # At extreme twist (175 deg):
            if angle_deg >= 170.0:
                # LBS midpoint radius shrinks by over 90% (pinched to near zero)
                assert mid_lbs_radius < 0.1 * radius
                # DQS midpoint radius is preserved to over 98%
                assert mid_dqs_radius > 0.98 * radius

                # LBS volume collapses significantly
                assert vol_lbs < 0.60 * initial_volume
                # DQS volume is preserved to over 90%
                assert vol_dqs > 0.90 * initial_volume

                # Verify DQS has 0 normal inversions (normals still point outward radially)
                radial_dir = deformed_dqs_v[midpoint_mask, :2]
                radial_norm = np.linalg.norm(radial_dir, axis=1, keepdims=True)
                radial_dir = radial_dir / np.maximum(radial_norm, 1e-8)
                normal_xy = deformed_dqs_n[midpoint_mask, :2]
                dots = np.sum(radial_dir * normal_xy, axis=1)
                assert np.all(dots > 0.90), "DQS normals inverted or distorted"
