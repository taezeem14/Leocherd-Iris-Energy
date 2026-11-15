"""
Procedural Synthetic Mesh Fixtures & Deformation Verification Tools.

Provides procedural 3D meshes and geometric validation algorithms:
1. Articulated Twist Cylinder (DQS vs LBS candy-wrapper volume loss benchmark)
2. Articulated Capsule Mesh (watertight limb proxy)
3. Enclosed Volume Calculator (Divergence Theorem / signed tetrahedral sums)
4. Boundary Edge Detector & Watertightness Verifier
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from iris.rigging.joints import BoneEdge, JointNode
from iris.rigging.orientation import DualQuaternion, QuaternionMath
from iris.rigging.skeleton import Skeleton
from iris.rigging.skinning import (
    deform_dqs,
    deform_lbs,
    normalize_skin_weights,
)


# =============================================================================
# Geometric Analysis & Invariant Calculation Utilities
# =============================================================================

def compute_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Computes enclosed signed volume of a watertight triangular mesh via the Divergence Theorem
    (signed tetrahedron decomposition):
        V = 1/6 * |sum_{f=(a,b,c)} v_a . (v_b x v_c)|

    Parameters:
        vertices: Array of vertex coordinates (V, 3)
        faces: Array of triangular face indices (F, 3)

    Returns:
        float: Enclosed volume in cubic units (always non-negative)
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tri_faces = np.asarray(faces, dtype=np.int64)
    if len(tri_faces) == 0:
        return 0.0

    v0 = verts[tri_faces[:, 0]]
    v1 = verts[tri_faces[:, 1]]
    v2 = verts[tri_faces[:, 2]]
    cross = np.cross(v1, v2)
    dot = np.sum(v0 * cross, axis=1)
    return float(np.abs(np.sum(dot)) / 6.0)


def compute_mesh_boundary(faces: np.ndarray) -> np.ndarray:
    """
    Detects open boundary edges in a triangular mesh.
    In a closed watertight 2-manifold surface, every undirected edge must be shared
    by exactly two adjacent faces. Edges belonging to exactly 1 face are boundary edges.

    Parameters:
        faces: Array of triangular face indices (F, 3)

    Returns:
        np.ndarray: Array of boundary edge pairs of shape (E_boundary, 2).
                    Empty array of shape (0, 2) if the mesh is watertight.
    """
    tri_faces = np.asarray(faces, dtype=np.int64)
    if len(tri_faces) == 0:
        return np.empty((0, 2), dtype=np.int64)

    # Extract all 3 edges per face as sorted vertex index pairs
    edges = []
    for face in tri_faces:
        u0, u1, u2 = int(face[0]), int(face[1]), int(face[2])
        edges.append((min(u0, u1), max(u0, u1)))
        edges.append((min(u1, u2), max(u1, u2)))
        edges.append((min(u2, u0), max(u2, u0)))

    edge_counts = Counter(edges)
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

    if not boundary_edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(boundary_edges, dtype=np.int64)


def is_mesh_watertight(faces: np.ndarray) -> bool:
    """Returns True if the mesh has 0 open boundary edges and is a closed manifold."""
    return len(compute_mesh_boundary(faces)) == 0


# =============================================================================
# Cylinder Twist Mesh (DQS vs LBS Benchmark)
# =============================================================================

class CylinderMesh(tuple):
    """
    Subclass of tuple (vertices, normals, faces) with named accessors
    and built-in twist deformation benchmarking helpers.
    """

    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    joint_indices: np.ndarray
    skeleton: Optional[Skeleton]
    radius: float
    height: float

    def __new__(
        cls,
        vertices: np.ndarray,
        normals: np.ndarray,
        faces: np.ndarray,
        weights: np.ndarray,
        joint_indices: np.ndarray,
        skeleton: Optional[Skeleton] = None,
        radius: float = 0.2,
        height: float = 1.0,
    ) -> "CylinderMesh":
        v = np.asarray(vertices, dtype=np.float32)
        n = np.asarray(normals, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int64)
        obj = super().__new__(cls, (v, n, f))
        obj.vertices = v
        obj.normals = n
        obj.faces = f
        obj.weights = np.asarray(weights, dtype=np.float32)
        obj.joint_indices = np.asarray(joint_indices, dtype=np.uint16)
        obj.skeleton = skeleton
        obj.radius = float(radius)
        obj.height = float(height)
        return obj

    def get_midpoint_mask(self, atol: float = 0.05) -> np.ndarray:
        """Returns boolean mask of vertices located at the central neck ring (z ≈ 0)."""
        return np.isclose(self.vertices[:, 2], 0.0, atol=atol)

    def compute_volume(self, deformed_vertices: Optional[np.ndarray] = None) -> float:
        """Computes enclosed volume for the rest mesh or given deformed vertex array."""
        v = deformed_vertices if deformed_vertices is not None else self.vertices
        return compute_mesh_volume(v, self.faces)

    def midpoint_radius(self, deformed_vertices: np.ndarray) -> float:
        """Computes average cross-sectional radius at the midpoint ring (z ≈ 0)."""
        mask = self.get_midpoint_mask()
        if not np.any(mask):
            return 0.0
        xy = deformed_vertices[mask, :2]
        return float(np.mean(np.linalg.norm(xy, axis=1)))

    def deform_lbs(self, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """Deforms cylinder using Linear Blend Skinning under specified axial twist angle."""
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)

        M0 = np.eye(4, dtype=np.float32)
        M1 = np.array([
            [c, -s, 0.0, 0.0],
            [s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        joint_matrices = np.stack([M0, M1])

        return deform_lbs(
            self.vertices,
            self.normals,
            joint_matrices,
            self.weights,
            self.joint_indices,
        )

    def deform_dqs(self, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """Deforms cylinder using Dual Quaternion Skinning under specified axial twist angle."""
        angle_rad = math.radians(angle_deg)
        dq0 = DualQuaternion.identity()
        q1 = np.array([math.cos(angle_rad / 2.0), 0.0, 0.0, math.sin(angle_rad / 2.0)], dtype=np.float32)
        dq1 = DualQuaternion.from_rotation_translation(q1, np.zeros(3, dtype=np.float32))
        joint_dqs = [dq0, dq1]

        return deform_dqs(
            self.vertices,
            self.normals,
            joint_dqs,
            self.weights,
            self.joint_indices,
        )


def create_cylinder_twist_mesh(
    radius: float = 0.2,
    height: float = 1.0,
    rings: int = 20,
    radial_segments: int = 16,
    num_bones: int = 2,
    return_skinning: bool = False,
) -> Union[CylinderMesh, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates a watertight cylindrical mesh centered along Z from -height/2 to +height/2,
    with skin weights configured for dual quaternion vs linear blend twist testing.

    Parameters:
        radius: Cylinder radius (default: 0.2)
        height: Cylinder length along Z (default: 1.0)
        rings: Number of longitudinal vertex rings along Z (default: 20)
        radial_segments: Number of angular slices around circumference (default: 16)
        num_bones: Number of bone segments along Z (2 or 3, default: 2)
        return_skinning: If True, returns 5-tuple (verts, normals, faces, weights, indices).
                         If False, returns CylinderMesh (which unpacks as 3-tuple verts, normals, faces
                         while exposing .weights and helper methods).
    """
    verts_list = []
    normals_list = []
    z_coords = np.linspace(-height / 2.0, height / 2.0, rings, dtype=np.float32)

    # 1. Lateral cylinder wall vertices and outward unit normals
    for z in z_coords:
        for seg in range(radial_segments):
            angle = 2.0 * math.pi * seg / radial_segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            verts_list.append([radius * cos_a, radius * sin_a, z])
            normals_list.append([cos_a, sin_a, 0.0])

    verts = np.array(verts_list, dtype=np.float32)
    normals = np.array(normals_list, dtype=np.float32)

    # 2. Lateral faces (quads split into 2 CCW triangles)
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

    # 3. Bottom Cap (z = -height/2, outward normal [0, 0, -1])
    bottom_center_idx = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, -height / 2.0]]])
    normals = np.vstack([normals, [[0.0, 0.0, -1.0]]])

    # 4. Top Cap (z = +height/2, outward normal [0, 0, +1])
    top_center_idx = len(verts)
    verts = np.vstack([verts, [[0.0, 0.0, height / 2.0]]])
    normals = np.vstack([normals, [[0.0, 0.0, 1.0]]])

    for s in range(radial_segments):
        next_s = (s + 1) % radial_segments
        # Bottom cap: CCW seen from outside (-Z)
        faces.append([bottom_center_idx, next_s, s])
        # Top cap: CCW seen from outside (+Z)
        top_offset = (rings - 1) * radial_segments
        faces.append([top_center_idx, top_offset + s, top_offset + next_s])

    faces_arr = np.array(faces, dtype=np.int64)

    # 5. Analytic Skin Weights & Joint Indices along Z
    z_vals = verts[:, 2]
    # Linear blend along Z from z = -height/2 (t=0) to z = +height/2 (t=1)
    t = np.clip((z_vals + (height / 2.0)) / height, 0.0, 1.0)

    if num_bones == 2:
        w0 = 1.0 - t
        w1 = t
        raw_weights = np.column_stack([w0, w1, np.zeros_like(w0), np.zeros_like(w0)]).astype(np.float32)
        raw_indices = np.zeros((len(verts), 4), dtype=np.uint16)
        raw_indices[:, 1] = 1

        # Skeleton
        skel = Skeleton(name="CylinderRig", root_name="Joint_0")
        skel.joints["Joint_0"] = JointNode(
            name="Joint_0",
            parent_name=None,
            rest_world_pos=np.array([0.0, 0.0, -height / 2.0], dtype=np.float32),
            children_names=["Joint_1"],
        )
        skel.joints["Joint_1"] = JointNode(
            name="Joint_1",
            parent_name="Joint_0",
            rest_world_pos=np.array([0.0, 0.0, height / 2.0], dtype=np.float32),
        )
        skel.bones["Bone_0_1"] = BoneEdge(
            name="Bone_0_1",
            parent_joint="Joint_0",
            child_joint="Joint_1",
            rest_length=height,
        )
    else:  # 3 bones
        # Joint 0 at -height/2, Joint 1 at 0, Joint 2 at +height/2
        w0 = np.clip(1.0 - 2.0 * t, 0.0, 1.0)
        w1 = np.clip(1.0 - 2.0 * np.abs(t - 0.5), 0.0, 1.0)
        w2 = np.clip(2.0 * t - 1.0, 0.0, 1.0)
        raw_weights = np.column_stack([w0, w1, w2, np.zeros_like(w0)]).astype(np.float32)
        raw_indices = np.zeros((len(verts), 4), dtype=np.uint16)
        raw_indices[:, 1] = 1
        raw_indices[:, 2] = 2

        skel = Skeleton(name="CylinderRig3", root_name="Joint_0")
        skel.joints["Joint_0"] = JointNode(
            name="Joint_0",
            parent_name=None,
            rest_world_pos=np.array([0.0, 0.0, -height / 2.0], dtype=np.float32),
            children_names=["Joint_1"],
        )
        skel.joints["Joint_1"] = JointNode(
            name="Joint_1",
            parent_name="Joint_0",
            rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            children_names=["Joint_2"],
        )
        skel.joints["Joint_2"] = JointNode(
            name="Joint_2",
            parent_name="Joint_1",
            rest_world_pos=np.array([0.0, 0.0, height / 2.0], dtype=np.float32),
        )
        skel.bones["Bone_0_1"] = BoneEdge(
            name="Bone_0_1",
            parent_joint="Joint_0",
            child_joint="Joint_1",
            rest_length=height / 2.0,
        )
        skel.bones["Bone_1_2"] = BoneEdge(
            name="Bone_1_2",
            parent_joint="Joint_1",
            child_joint="Joint_2",
            rest_length=height / 2.0,
        )

    weights, joint_indices = normalize_skin_weights(raw_weights, raw_indices, max_influences=4)

    if return_skinning:
        return verts, normals, faces_arr, weights, joint_indices

    return CylinderMesh(
        verts,
        normals,
        faces_arr,
        weights,
        joint_indices,
        skeleton=skel,
        radius=radius,
        height=height,
    )


# =============================================================================
# Capsule Mesh
# =============================================================================

class CapsuleMesh(tuple):
    """Subclass of tuple (vertices, normals, faces) with named attribute access."""

    vertices: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    radius: float
    length: float

    def __new__(
        cls,
        vertices: np.ndarray,
        normals: np.ndarray,
        faces: np.ndarray,
        radius: float = 0.15,
        length: float = 0.6,
    ) -> "CapsuleMesh":
        v = np.asarray(vertices, dtype=np.float32)
        n = np.asarray(normals, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int64)
        obj = super().__new__(cls, (v, n, f))
        obj.vertices = v
        obj.normals = n
        obj.faces = f
        obj.radius = float(radius)
        obj.length = float(length)
        return obj

    def compute_volume(self) -> float:
        """Computes enclosed volume of the capsule."""
        return compute_mesh_volume(self.vertices, self.faces)

    def is_watertight(self) -> bool:
        """Returns True if the capsule is a closed watertight 2-manifold."""
        return is_mesh_watertight(self.faces)


def create_capsule_mesh(
    radius: float = 0.15,
    length: float = 0.6,
    rings: int = 8,
    radial_segments: int = 12,
) -> CapsuleMesh:
    """
    Generates a watertight capsule mesh oriented along Z.
    Consists of a central cylindrical sleeve of length `length`, capped with
    two hemispheres of radius `radius` at -length/2 and +length/2.

    Parameters:
        radius: Capsule radius (default: 0.15)
        length: Cylindrical sleeve length (default: 0.6)
        rings: Number of latitude rings per hemisphere (default: 8)
        radial_segments: Number of longitudinal sectors (default: 12)

    Returns:
        CapsuleMesh: (vertices, normals, faces) tuple with named accessors
    """
    verts = []
    normals = []
    half_len = length / 2.0
    lat_steps = max(3, rings)
    rad_segs = max(4, radial_segments)

    # 1. Bottom Pole Vertex
    bottom_pole_idx = 0
    verts.append([0.0, 0.0, -half_len - radius])
    normals.append([0.0, 0.0, -1.0])

    # 2. Bottom Hemisphere Rings: phi from -pi/2 + dphi to -dphi
    for r in range(1, lat_steps):
        phi = -0.5 * math.pi + (0.5 * math.pi) * (r / lat_steps)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        z = -half_len + radius * sin_phi
        for s in range(rad_segs):
            theta = 2.0 * math.pi * s / rad_segs
            nx = cos_phi * math.cos(theta)
            ny = cos_phi * math.sin(theta)
            nz = sin_phi
            verts.append([radius * nx, radius * ny, z])
            normals.append([nx, ny, nz])

    # 3. Cylindrical Sleeve Ring at -half_len (phi = 0 for bottom hemisphere)
    for s in range(rad_segs):
        theta = 2.0 * math.pi * s / rad_segs
        nx = math.cos(theta)
        ny = math.sin(theta)
        verts.append([radius * nx, radius * ny, -half_len])
        normals.append([nx, ny, 0.0])

    # 4. Cylindrical Sleeve Ring at +half_len (phi = 0 for top hemisphere)
    for s in range(rad_segs):
        theta = 2.0 * math.pi * s / rad_segs
        nx = math.cos(theta)
        ny = math.sin(theta)
        verts.append([radius * nx, radius * ny, half_len])
        normals.append([nx, ny, 0.0])

    # 5. Top Hemisphere Rings: phi from +dphi to +pi/2 - dphi
    for r in range(1, lat_steps):
        phi = (0.5 * math.pi) * (r / lat_steps)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        z = half_len + radius * sin_phi
        for s in range(rad_segs):
            theta = 2.0 * math.pi * s / rad_segs
            nx = cos_phi * math.cos(theta)
            ny = cos_phi * math.sin(theta)
            nz = sin_phi
            verts.append([radius * nx, radius * ny, z])
            normals.append([nx, ny, nz])

    # 6. Top Pole Vertex
    top_pole_idx = len(verts)
    verts.append([0.0, 0.0, half_len + radius])
    normals.append([0.0, 0.0, 1.0])

    verts_arr = np.array(verts, dtype=np.float32)
    normals_arr = np.array(normals, dtype=np.float32)

    # 7. Triangular Faces
    faces = []
    # Fan from bottom pole to first latitude ring (vertex indices 1 .. rad_segs)
    for s in range(rad_segs):
        next_s = (s + 1) % rad_segs
        faces.append([bottom_pole_idx, 1 + next_s, 1 + s])

    # Quads between consecutive latitude rings
    num_rings = (lat_steps - 1) + 2 + (lat_steps - 1)
    for r in range(num_rings - 1):
        ring_start = 1 + r * rad_segs
        next_ring_start = ring_start + rad_segs
        for s in range(rad_segs):
            next_s = (s + 1) % rad_segs
            v0 = ring_start + s
            v1 = ring_start + next_s
            v2 = next_ring_start + s
            v3 = next_ring_start + next_s
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])

    # Fan from last latitude ring to top pole
    last_ring_start = 1 + (num_rings - 1) * rad_segs
    for s in range(rad_segs):
        next_s = (s + 1) % rad_segs
        faces.append([top_pole_idx, last_ring_start + s, last_ring_start + next_s])

    faces_arr = np.array(faces, dtype=np.int64)

    return CapsuleMesh(
        verts_arr,
        normals_arr,
        faces_arr,
        radius=radius,
        length=length,
    )
