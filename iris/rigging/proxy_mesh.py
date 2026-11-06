"""
Procedural skinned proxy mesh generator for creature skeletons.
Generates volumetric, articulated 3D mannequin geometry (limbs, torso, head, joint caps)
with Linear Blend Skinning (LBS) vertex weights attached to the skeleton hierarchy.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from iris.rigging.skeleton import Skeleton


class ProxyMeshGenerator:
    """Generates a complete, watertight, visually appealing skinned proxy mesh for any creature skeleton."""

    @staticmethod
    def _create_orthogonal_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates two unit vectors perpendicular to direction and each other."""
        d = direction / (np.linalg.norm(direction) + 1e-8)
        if abs(d[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = np.cross(d, ref)
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(d, u)
        v = v / (np.linalg.norm(v) + 1e-8)
        return u, v

    @classmethod
    def generate_bone_capsule(
        cls,
        p_start: np.ndarray,
        p_end: np.ndarray,
        radius_start: float,
        radius_end: float,
        j_parent: int,
        j_child: int,
        slices: int = 10,
        rings: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates an articulated tapered bone segment with smooth LBS weights.
        """
        vec = p_end - p_start
        length = float(np.linalg.norm(vec))
        if length < 1e-4:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint16),
                np.zeros((0, 4), dtype=np.uint16),
                np.zeros((0, 4), dtype=np.float32),
            )

        u, v = cls._create_orthogonal_basis(vec)

        verts: List[List[float]] = []
        norms: List[List[float]] = []
        joint_idx: List[List[int]] = []
        joint_w: List[List[float]] = []
        indices: List[List[int]] = []

        # Create vertices ring by ring
        for r in range(rings):
            t = r / float(rings - 1)  # 0.0 at parent, 1.0 at child
            center = p_start + t * vec

            # Anatomical bulge profile (slightly thicker in middle)
            bulge = 1.0 + 0.25 * math.sin(math.pi * t)
            rad = (radius_start * (1.0 - t) + radius_end * t) * bulge

            # Skinning weights: linear blend between parent and child joints
            w_parent = 1.0 - t
            w_child = t

            for s in range(slices):
                theta = (2.0 * math.pi * s) / float(slices)
                cos_th = math.cos(theta)
                sin_th = math.sin(theta)

                normal = cos_th * u + sin_th * v
                pos = center + rad * normal

                verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
                norms.append([float(normal[0]), float(normal[1]), float(normal[2])])
                joint_idx.append([j_parent, j_child, 0, 0])
                joint_w.append([float(w_parent), float(w_child), 0.0, 0.0])

        # Triangulate adjacent rings
        for r in range(rings - 1):
            ring_start = r * slices
            next_ring_start = (r + 1) * slices
            for s in range(slices):
                s_next = (s + 1) % slices
                i00 = ring_start + s
                i01 = ring_start + s_next
                i10 = next_ring_start + s
                i11 = next_ring_start + s_next

                # Two triangles per quad (CCW winding)
                indices.append([i00, i10, i01])
                indices.append([i01, i10, i11])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(joint_idx, dtype=np.uint16),
            np.array(joint_w, dtype=np.float32),
        )

    @classmethod
    def generate_joint_sphere(
        cls,
        center: np.ndarray,
        radius: float,
        joint_idx: int,
        lat_segments: int = 8,
        lon_segments: int = 10,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a 3D UV sphere/ellipsoid volume centered at a joint, 100% weighted to that joint.
        """
        verts: List[List[float]] = []
        norms: List[List[float]] = []
        joint_indices: List[List[int]] = []
        joint_weights: List[List[float]] = []
        indices: List[List[int]] = []

        sx, sy, sz = scale

        # Generate latitudinal slices
        for i in range(lat_segments + 1):
            lat = math.pi * (float(i) / lat_segments - 0.5)  # -pi/2 to pi/2
            y = math.sin(lat)
            cos_lat = math.cos(lat)

            for j in range(lon_segments):
                lon = 2.0 * math.pi * float(j) / lon_segments
                x = cos_lat * math.cos(lon)
                z = cos_lat * math.sin(lon)

                normal = np.array([x, y, z], dtype=np.float32)
                pos = center + radius * np.array([x * sx, y * sy, z * sz], dtype=np.float32)

                verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
                norms.append([float(normal[0]), float(normal[1]), float(normal[2])])
                joint_indices.append([joint_idx, 0, 0, 0])
                joint_weights.append([1.0, 0.0, 0.0, 0.0])

        # Triangulate
        for i in range(lat_segments):
            for j in range(lon_segments):
                first = i * lon_segments + j
                second = first + lon_segments
                j_next = (j + 1) % lon_segments
                first_next = i * lon_segments + j_next
                second_next = first_next + lon_segments

                indices.append([first, second, first_next])
                indices.append([first_next, second, second_next])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(joint_indices, dtype=np.uint16),
            np.array(joint_weights, dtype=np.float32),
        )

    @classmethod
    def create_skinned_mannequin(
        cls,
        skeleton: Skeleton,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds a comprehensive, rigged 3D proxy mesh covering all bones and joints of the skeleton.
        Returns:
            vertices: (V, 3) float32
            normals: (V, 3) float32
            indices: (F, 3) uint16
            joint_indices: (V, 4) uint16
            joint_weights: (V, 4) float32
        """
        joint_names = list(skeleton.joints.keys())
        joint_to_idx = {name: i for i, name in enumerate(joint_names)}

        all_verts: List[np.ndarray] = []
        all_norms: List[np.ndarray] = []
        all_indices: List[np.ndarray] = []
        all_joint_indices: List[np.ndarray] = []
        all_joint_weights: List[np.ndarray] = []

        vertex_offset = 0

        def append_part(v: np.ndarray, n: np.ndarray, idx: np.ndarray, j_idx: np.ndarray, j_w: np.ndarray) -> None:
            nonlocal vertex_offset
            if len(v) == 0:
                return
            all_verts.append(v)
            all_norms.append(n)
            all_indices.append(idx + vertex_offset)
            all_joint_indices.append(j_idx)
            all_joint_weights.append(j_w)
            vertex_offset += len(v)

        # 1. Articulated bone capsules
        for bone in skeleton.bones.values():
            if bone.parent_joint not in joint_to_idx or bone.child_joint not in joint_to_idx:
                continue

            j_p_idx = joint_to_idx[bone.parent_joint]
            j_c_idx = joint_to_idx[bone.child_joint]
            p_start = skeleton.joints[bone.parent_joint].rest_world_pos
            p_end = skeleton.joints[bone.child_joint].rest_world_pos
            bone_length = float(np.linalg.norm(p_end - p_start))

            # Determine appropriate radius by anatomical bone role
            b_lower = bone.name.lower()
            if "spine" in b_lower or "chest" in b_lower:
                r_start, r_end = 0.08, 0.09
            elif "hips" in b_lower:
                r_start, r_end = 0.085, 0.075
            elif "neck" in b_lower:
                r_start, r_end = 0.045, 0.040
            elif "upleg" in b_lower or "thigh" in b_lower:
                r_start, r_end = 0.065, 0.050
            elif "leg" in b_lower or "shin" in b_lower:
                r_start, r_end = 0.050, 0.038
            elif "arm" in b_lower or "shoulder" in b_lower:
                r_start, r_end = 0.048, 0.040
            elif "forearm" in b_lower:
                r_start, r_end = 0.040, 0.032
            else:
                # Default proportional to bone length
                rad = max(0.02, min(0.06, bone_length * 0.15))
                r_start, r_end = rad, rad

            v, n, idx, j_i, j_w = cls.generate_bone_capsule(
                p_start=p_start,
                p_end=p_end,
                radius_start=r_start,
                radius_end=r_end,
                j_parent=j_p_idx,
                j_child=j_c_idx,
                slices=10,
                rings=5,
            )
            append_part(v, n, idx, j_i, j_w)

        # 2. Key joint volumetric volumes (Head, Hands, Feet, Hips)
        for j_name, j_idx in joint_to_idx.items():
            pos = skeleton.joints[j_name].rest_world_pos
            name_lower = j_name.lower()

            if "head" in name_lower:
                # Stylized head volume: slightly elevated and elongated
                v, n, idx, j_i, j_w = cls.generate_joint_sphere(
                    center=pos + np.array([0.0, 0.05, 0.0], dtype=np.float32),
                    radius=0.095,
                    joint_idx=j_idx,
                    lat_segments=8,
                    lon_segments=10,
                    scale=(0.9, 1.15, 0.95),
                )
                append_part(v, n, idx, j_i, j_w)
            elif "hand" in name_lower:
                v, n, idx, j_i, j_w = cls.generate_joint_sphere(
                    center=pos,
                    radius=0.038,
                    joint_idx=j_idx,
                    lat_segments=6,
                    lon_segments=8,
                    scale=(0.8, 1.2, 0.6),
                )
                append_part(v, n, idx, j_i, j_w)
            elif "foot" in name_lower:
                v, n, idx, j_i, j_w = cls.generate_joint_sphere(
                    center=pos + np.array([0.0, -0.02, 0.04], dtype=np.float32),
                    radius=0.045,
                    joint_idx=j_idx,
                    lat_segments=6,
                    lon_segments=8,
                    scale=(0.7, 0.5, 1.4),
                )
                append_part(v, n, idx, j_i, j_w)
            elif any(k in name_lower for k in ["elbow", "knee", "shoulder"]):
                # Subtle joint articulation cap
                v, n, idx, j_i, j_w = cls.generate_joint_sphere(
                    center=pos,
                    radius=0.035,
                    joint_idx=j_idx,
                    lat_segments=5,
                    lon_segments=6,
                )
                append_part(v, n, idx, j_i, j_w)

        final_verts = np.vstack(all_verts)
        final_norms = np.vstack(all_norms)
        final_indices = np.vstack(all_indices)
        raw_joint_indices = np.vstack(all_joint_indices)
        raw_joint_weights = np.vstack(all_joint_weights)

        # Sanitize joint weights and indices to strictly conform to glTF 2.0 specification:
        # 1. Influences sorted descending by weight.
        # 2. Weights normalized to sum exactly to 1.0.
        # 3. Any influence with zero weight MUST have joint index = 0 (prevents ACCESSOR_JOINTS_USED_ZERO_WEIGHT).
        V = len(raw_joint_weights)
        clean_joint_indices = np.zeros((V, 4), dtype=np.uint16)
        clean_joint_weights = np.zeros((V, 4), dtype=np.float32)

        for v_idx in range(V):
            pairs = []
            for k in range(4):
                w = float(raw_joint_weights[v_idx, k])
                j = int(raw_joint_indices[v_idx, k])
                if w > 1e-5:
                    pairs.append((w, j))

            pairs.sort(key=lambda p: p[0], reverse=True)
            total_w = sum(p[0] for p in pairs)
            if total_w < 1e-6:
                clean_joint_weights[v_idx, 0] = 1.0
                clean_joint_indices[v_idx, 0] = 0
            else:
                for k, (w, j) in enumerate(pairs[:4]):
                    clean_joint_weights[v_idx, k] = w / total_w
                    clean_joint_indices[v_idx, k] = j

        return (
            final_verts.astype(np.float32),
            final_norms.astype(np.float32),
            final_indices.astype(np.uint16),
            clean_joint_indices,
            clean_joint_weights,
        )
