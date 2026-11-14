"""
Procedural anatomical stylized humanoid character mesh generator.
Generates an articulated, watertight, visually appealing 3D humanoid character mesh
(sculpted torso, deltoids, tapered limbs, anatomical head with visor, hands, feet)
with smooth Linear Blend Skinning (LBS) vertex weights attached to the skeleton hierarchy.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from iris.rigging.skeleton import Skeleton


class ProxyMeshGenerator:
    """Generates an anatomical stylized humanoid proxy character for any creature skeleton."""

    @staticmethod
    def _create_orthogonal_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates two unit vectors perpendicular to direction and each other."""
        d = direction / (np.linalg.norm(direction) + 1e-8)
        if abs(d[2]) < 0.85:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = np.cross(d, ref)
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(d, u)
        v = v / (np.linalg.norm(v) + 1e-8)
        return u.astype(np.float32), v.astype(np.float32)

    @classmethod
    def generate_loft_segment(
        cls,
        p_start: np.ndarray,
        p_end: np.ndarray,
        radius_x_start: float,
        radius_z_start: float,
        radius_x_end: float,
        radius_z_end: float,
        j_parent: int,
        j_child: int,
        slices: int = 12,
        rings: int = 6,
        bulge_center: float = 0.4,
        bulge_amount: float = 0.2,
        curve_offset_z: float = 0.0,
        weight_curve_power: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates an elliptical anatomical loft segment (e.g. thigh, calf, forearm, bicep)
        with smooth Hermite blend skinning weights.
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

        for r in range(rings):
            t = r / float(rings - 1)  # 0.0 at parent, 1.0 at child

            # Center position with optional anatomical curvature (e.g. calf back curve)
            center = p_start + t * vec
            if curve_offset_z != 0.0:
                center = center + v * (curve_offset_z * math.sin(math.pi * t))

            # Anatomical bulge (muscular contour)
            bulge = 1.0 + bulge_amount * math.sin(math.pi * (t ** (1.0 / max(0.2, bulge_center * 2.0))))
            rx = (radius_x_start * (1.0 - t) + radius_x_end * t) * bulge
            rz = (radius_z_start * (1.0 - t) + radius_z_end * t) * bulge

            # Smooth Hermite skinning weights
            smooth_t = t * t * (3.0 - 2.0 * t) if weight_curve_power == 1.0 else (t ** weight_curve_power)
            w_parent = max(0.0, min(1.0, 1.0 - smooth_t))
            w_child = max(0.0, min(1.0, smooth_t))

            for s in range(slices):
                theta = (2.0 * math.pi * s) / float(slices)
                cos_th = math.cos(theta)
                sin_th = math.sin(theta)

                normal = cos_th * u + sin_th * v
                pos = center + rx * cos_th * u + rz * sin_th * v

                verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
                norms.append([float(normal[0]), float(normal[1]), float(normal[2])])
                joint_idx.append([j_parent, j_child, 0, 0])
                joint_w.append([float(w_parent), float(w_child), 0.0, 0.0])

        for r in range(rings - 1):
            r0 = r * slices
            r1 = (r + 1) * slices
            for s in range(slices):
                s_next = (s + 1) % slices
                indices.append([r0 + s, r1 + s, r0 + s_next])
                indices.append([r0 + s_next, r1 + s, r1 + s_next])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(joint_idx, dtype=np.uint16),
            np.array(joint_w, dtype=np.float32),
        )

    @classmethod
    def generate_stylized_head(
        cls,
        center: np.ndarray,
        head_joint_idx: int,
        neck_joint_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a stylized humanoid head with cranium, jawline, and directional visor faceplate.
        """
        verts: List[List[float]] = []
        norms: List[List[float]] = []
        j_idx: List[List[int]] = []
        j_w: List[List[float]] = []
        indices: List[List[int]] = []

        # 1. Main Cranium (lat/lon ellipsoid, wider at top, tapering to jaw)
        lat_segs = 8
        lon_segs = 12
        r_head = 0.095

        for i in range(lat_segs + 1):
            lat = math.pi * (float(i) / lat_segs - 0.5)
            y = math.sin(lat)
            cos_lat = math.cos(lat)

            # Cranium taper: broader at top, tapered at jaw
            taper_x = 0.85 if y < 0 else 0.95
            taper_z = 0.80 if y < -0.3 else 1.05

            for j in range(lon_segs):
                lon = 2.0 * math.pi * float(j) / lon_segs
                x = cos_lat * math.sin(lon)
                z = cos_lat * math.cos(lon)

                # Push forward slightly for jaw
                jaw_offset = 0.02 if (y < -0.2 and z > 0) else 0.0

                pos = center + np.array(
                    [x * r_head * taper_x, (y * r_head * 1.15) + 0.04, (z * r_head * taper_z) + jaw_offset],
                    dtype=np.float32,
                )
                normal = np.array([x, y, z], dtype=np.float32)
                normal = normal / (np.linalg.norm(normal) + 1e-8)

                verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
                norms.append([float(normal[0]), float(normal[1]), float(normal[2])])

                # Blend slightly with neck at bottom of head
                w_neck = 0.25 if y < -0.4 else 0.0
                j_idx.append([head_joint_idx, neck_joint_idx, 0, 0])
                j_w.append([1.0 - w_neck, w_neck, 0.0, 0.0])

        for i in range(lat_segs):
            for j in range(lon_segs):
                first = i * lon_segs + j
                second = first + lon_segs
                j_next = (j + 1) % lon_segs
                indices.append([first, second, (i * lon_segs + j_next)])
                indices.append([(i * lon_segs + j_next), second, (second + 1 if j_next != 0 else second + 1 - lon_segs)])

        # 2. Sleek Visor Plate on Face (+Z face, angled, stylish)
        v_offset = len(verts)
        visor_pts = [
            # Top edge (temple left to temple right)
            [-0.055, 0.06, 0.088],
            [0.0, 0.065, 0.098],
            [0.055, 0.06, 0.088],
            # Mid line (cheek left, nose tip, cheek right)
            [-0.050, 0.01, 0.092],
            [0.0, 0.015, 0.106],
            [0.050, 0.01, 0.092],
            # Bottom line (mouth/chin angle)
            [-0.035, -0.03, 0.082],
            [0.0, -0.025, 0.092],
            [0.035, -0.03, 0.082],
        ]

        for p in visor_pts:
            pos = center + np.array(p, dtype=np.float32)
            normal = np.array([p[0] * 0.5, p[1] * 0.3, 0.95], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
            norms.append([float(normal[0]), float(normal[1]), float(normal[2])])
            j_idx.append([head_joint_idx, 0, 0, 0])
            j_w.append([1.0, 0.0, 0.0, 0.0])

        # Triangulate visor plate
        visor_tris = [
            # Top quads
            [0, 3, 1], [1, 3, 4],
            [1, 4, 2], [2, 4, 5],
            # Bottom quads
            [3, 6, 4], [4, 6, 7],
            [4, 7, 5], [5, 7, 8],
        ]
        for tri in visor_tris:
            indices.append([v_offset + tri[0], v_offset + tri[1], v_offset + tri[2]])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(j_idx, dtype=np.uint16),
            np.array(j_w, dtype=np.float32),
        )

    @classmethod
    def generate_torso_ribcage(
        cls,
        p_hips: np.ndarray,
        p_spine: np.ndarray,
        p_chest: np.ndarray,
        p_neck: np.ndarray,
        j_hips: int,
        j_spine: int,
        j_chest: int,
        j_neck: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a continuous, sculpted torso (pelvis, waist, ribcage, chest, clavicles)
        with smooth 4-joint blend skinning that eliminates pinching.
        """
        verts: List[List[float]] = []
        norms: List[List[float]] = []
        j_idx: List[List[int]] = []
        j_w: List[List[float]] = []
        indices: List[List[int]] = []

        slices = 16
        # Define 7 anatomical cross-sections along the torso height from pelvis to neck base
        # [y_pos, rx (width), rz (depth), z_center_offset, (w_hips, w_spine, w_chest, w_neck)]
        y_min = float(p_hips[1] - 0.08)
        y_max = float(p_chest[1] + 0.10)
        h_total = y_max - y_min

        sections = [
            # 0: Lower Pelvis / Groin
            (0.00, 0.115, 0.080, -0.010, (0.95, 0.05, 0.0, 0.0)),
            # 1: Pelvic Girdle / Iliac Crest
            (0.18, 0.138, 0.088, -0.005, (0.80, 0.20, 0.0, 0.0)),
            # 2: Waist / Navel
            (0.38, 0.112, 0.075, 0.005, (0.25, 0.65, 0.10, 0.0)),
            # 3: Lower Ribcage
            (0.55, 0.132, 0.088, 0.012, (0.05, 0.40, 0.55, 0.0)),
            # 4: Mid-Chest / Pectorals
            (0.72, 0.155, 0.098, 0.018, (0.0, 0.10, 0.85, 0.05)),
            # 5: Upper Chest / Clavicle Arch
            (0.88, 0.150, 0.090, 0.010, (0.0, 0.0, 0.85, 0.15)),
            # 6: Neck Base
            (1.00, 0.075, 0.065, 0.000, (0.0, 0.0, 0.50, 0.50)),
        ]

        for s_idx, (rel_y, rx, rz, cz, weights) in enumerate(sections):
            y_curr = y_min + rel_y * h_total
            for s in range(slices):
                theta = (2.0 * math.pi * s) / float(slices)
                cos_th = math.cos(theta)
                sin_th = math.sin(theta)

                # Anatomical chest contour: flatten back slightly, pronounce pectoral curvature in front
                adj_z = rz * (1.15 if sin_th > 0 else 0.90)
                px = rx * cos_th
                py = y_curr
                pz = adj_z * sin_th + cz

                normal = np.array([cos_th / rx, 0.0, sin_th / adj_z], dtype=np.float32)
                normal = normal / (np.linalg.norm(normal) + 1e-8)

                verts.append([float(px), float(py), float(pz)])
                norms.append([float(normal[0]), float(normal[1]), float(normal[2])])

                # 4-way blend across torso chain
                j_idx.append([j_hips, j_spine, j_chest, j_neck])
                j_w.append([float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3])])

        for r in range(len(sections) - 1):
            r0 = r * slices
            r1 = (r + 1) * slices
            for s in range(slices):
                s_next = (s + 1) % slices
                indices.append([r0 + s, r1 + s, r0 + s_next])
                indices.append([r0 + s_next, r1 + s, r1 + s_next])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(j_idx, dtype=np.uint16),
            np.array(j_w, dtype=np.float32),
        )

    @classmethod
    def generate_stylized_hand(
        cls,
        p_hand: np.ndarray,
        p_forearm: np.ndarray,
        is_left: bool,
        hand_joint_idx: int,
        forearm_joint_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates an anatomical humanoid hand with palm, thumb wedge, and fingers block.
        """
        verts: List[List[float]] = []
        norms: List[List[float]] = []
        j_idx: List[List[int]] = []
        j_w: List[List[float]] = []
        indices: List[List[int]] = []

        sign_x = -1.0 if is_left else 1.0

        # Hand local points (palm base at wrist, extending outward/downward)
        hand_box = [
            # Wrist connection
            [-0.020 * sign_x, 0.015, -0.015],
            [0.020 * sign_x, 0.015, -0.015],
            [0.020 * sign_x, 0.015, 0.015],
            [-0.020 * sign_x, 0.015, 0.015],
            # Palm knuckle ridge
            [-0.025 * sign_x, -0.045, -0.018],
            [0.025 * sign_x, -0.045, -0.018],
            [0.025 * sign_x, -0.045, 0.018],
            [-0.025 * sign_x, -0.045, 0.018],
            # Finger tips block
            [-0.022 * sign_x, -0.090, -0.012],
            [0.022 * sign_x, -0.090, -0.012],
            [0.022 * sign_x, -0.090, 0.012],
            [-0.022 * sign_x, -0.090, 0.012],
            # Thumb tip (angled outward and forward)
            [0.042 * sign_x, -0.035, 0.025],
        ]

        for p in hand_box:
            pos = p_hand + np.array(p, dtype=np.float32)
            normal = np.array([p[0], p[1], p[2]], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
            norms.append([float(normal[0]), float(normal[1]), float(normal[2])])
            # Wrist boundary blends with forearm
            w_fa = 0.35 if p[1] > 0.0 else 0.0
            j_idx.append([hand_joint_idx, forearm_joint_idx, 0, 0])
            j_w.append([1.0 - w_fa, w_fa, 0.0, 0.0])

        box_tris = [
            # Wrist to Palm quads
            [0, 4, 1], [1, 4, 5],
            [1, 5, 2], [2, 5, 6],
            [2, 6, 3], [3, 6, 7],
            [3, 7, 0], [0, 7, 4],
            # Palm to Fingers quads
            [4, 8, 5], [5, 8, 9],
            [5, 9, 6], [6, 9, 10],
            [6, 10, 7], [7, 10, 11],
            [7, 11, 4], [4, 11, 8],
            # Finger tip cap
            [8, 10, 9], [8, 11, 10],
            # Thumb triangles (connected to lateral palm points 5, 6, 9)
            [5, 12, 6], [6, 12, 10], [5, 9, 12],
        ]
        for tri in box_tris:
            indices.append([tri[0], tri[1], tri[2]])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(j_idx, dtype=np.uint16),
            np.array(j_w, dtype=np.float32),
        )

    @classmethod
    def generate_stylized_foot(
        cls,
        p_foot: np.ndarray,
        p_leg: np.ndarray,
        is_left: bool,
        foot_joint_idx: int,
        leg_joint_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates an anatomical humanoid foot/boot geometry with flat ground sole, heel cup, and toe box.
        """
        verts: List[List[float]] = []
        norms: List[List[float]] = []
        j_idx: List[List[int]] = []
        j_w: List[List[float]] = []
        indices: List[List[int]] = []

        sign_x = -1.0 if is_left else 1.0

        # Ankle connection & foot geometry relative to p_foot
        # Floor is at Y = 0.0 (p_foot is around Y=0.10)
        dy = -float(p_foot[1])  # Offset down to floor ground level

        foot_pts = [
            # Ankle ring
            [-0.035 * sign_x, 0.02, -0.025],
            [0.035 * sign_x, 0.02, -0.025],
            [0.035 * sign_x, 0.02, 0.025],
            [-0.035 * sign_x, 0.02, 0.025],
            # Heel back (sole level)
            [-0.038 * sign_x, dy + 0.02, -0.065],
            [0.038 * sign_x, dy + 0.02, -0.065],
            # Sole mid / instep arch
            [-0.042 * sign_x, dy + 0.00, 0.020],
            [0.042 * sign_x, dy + 0.00, 0.020],
            # Instep top bridge
            [-0.035 * sign_x, dy + 0.055, 0.030],
            [0.035 * sign_x, dy + 0.055, 0.030],
            # Toe box front (sole level)
            [-0.038 * sign_x, dy + 0.00, 0.125],
            [0.038 * sign_x, dy + 0.00, 0.125],
            # Toe box top
            [-0.032 * sign_x, dy + 0.032, 0.120],
            [0.032 * sign_x, dy + 0.032, 0.120],
        ]

        for p in foot_pts:
            pos = p_foot + np.array(p, dtype=np.float32)
            normal = np.array([p[0], max(-0.2, p[1]), max(0.1, p[2])], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
            norms.append([float(normal[0]), float(normal[1]), float(normal[2])])
            # Ankle top blends with leg (shin)
            w_leg = 0.35 if p[1] > 0.0 else 0.0
            j_idx.append([foot_joint_idx, leg_joint_idx, 0, 0])
            j_w.append([1.0 - w_leg, w_leg, 0.0, 0.0])

        foot_tris = [
            # Heel quads
            [0, 4, 1], [1, 4, 5],
            [0, 3, 4], [1, 5, 2],
            # Ankle to Instep top
            [3, 8, 2], [2, 8, 9],
            # Instep sides
            [4, 6, 8], [5, 9, 7],
            # Instep to Toe box
            [8, 12, 9], [9, 12, 13],
            [6, 10, 8], [8, 10, 12],
            [7, 9, 11], [9, 13, 11],
            # Toe front cap
            [10, 11, 12], [11, 13, 12],
            # Flat Ground Sole (bottom)
            [4, 7, 5], [4, 6, 7],
            [6, 11, 7], [6, 10, 11],
        ]
        for tri in foot_tris:
            indices.append([tri[0], tri[1], tri[2]])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(j_idx, dtype=np.uint16),
            np.array(j_w, dtype=np.float32),
        )

    @classmethod
    def generate_patella_knee_cap(
        cls,
        p_knee: np.ndarray,
        knee_joint_idx: int,
        thigh_joint_idx: int,
        shin_joint_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates an anatomical patella knee guard/cap covering the anterior knee joint (+Z)
        to prevent hollow crushing when the knee flexes.
        """
        verts: List[List[float]] = []
        norms: List[List[float]] = []
        j_idx: List[List[int]] = []
        j_w: List[List[float]] = []
        indices: List[List[int]] = []

        pts = [
            [-0.032, 0.040, 0.038],
            [0.032, 0.040, 0.038],
            [-0.035, 0.000, 0.048],
            [0.035, 0.000, 0.048],
            [-0.028, -0.040, 0.040],
            [0.028, -0.040, 0.040],
        ]

        for p in pts:
            pos = p_knee + np.array(p, dtype=np.float32)
            normal = np.array([p[0] * 0.5, p[1] * 0.3, 0.9], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
            norms.append([float(normal[0]), float(normal[1]), float(normal[2])])
            # Knee cap moves with knee, gently blended with thigh
            j_idx.append([knee_joint_idx, thigh_joint_idx, 0, 0])
            j_w.append([0.70, 0.30, 0.0, 0.0])

        tris = [
            [0, 2, 1], [1, 2, 3],
            [2, 4, 3], [3, 4, 5],
        ]
        for tri in tris:
            indices.append([tri[0], tri[1], tri[2]])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(j_idx, dtype=np.uint16),
            np.array(j_w, dtype=np.float32),
        )

    @classmethod
    def generate_deltoid_cap(
        cls,
        p_shoulder: np.ndarray,
        p_arm: np.ndarray,
        is_left: bool,
        shoulder_joint_idx: int,
        arm_joint_idx: int,
        chest_joint_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates an anatomical deltoid shoulder cap covering the shoulder articulation.
        """
        verts: List[List[float]] = []
        norms: List[List[float]] = []
        j_idx: List[List[int]] = []
        j_w: List[List[float]] = []
        indices: List[List[int]] = []

        sign_x = -1.0 if is_left else 1.0

        pts = [
            # Top clavicle ridge
            [-0.015 * sign_x, 0.045, -0.035],
            [0.045 * sign_x, 0.040, -0.035],
            [0.045 * sign_x, 0.040, 0.035],
            [-0.015 * sign_x, 0.045, 0.035],
            # Outer deltoid crest
            [0.075 * sign_x, 0.000, -0.040],
            [0.075 * sign_x, 0.000, 0.040],
            # Deltoid V-taper insertion
            [0.045 * sign_x, -0.060, 0.000],
        ]

        for p in pts:
            pos = p_arm + np.array(p, dtype=np.float32)
            normal = np.array([p[0], p[1], p[2]], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            verts.append([float(pos[0]), float(pos[1]), float(pos[2])])
            norms.append([float(normal[0]), float(normal[1]), float(normal[2])])
            # Deltoid blends between shoulder, chest, and upper arm
            j_idx.append([arm_joint_idx, shoulder_joint_idx, chest_joint_idx, 0])
            j_w.append([0.55, 0.30, 0.15, 0.0])

        tris = [
            [0, 4, 1], [3, 2, 5],
            [1, 4, 5], [1, 5, 2],
            [4, 6, 5],
            [0, 1, 2], [0, 2, 3],
        ]
        for tri in tris:
            indices.append([tri[0], tri[1], tri[2]])

        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(indices, dtype=np.uint16),
            np.array(j_idx, dtype=np.uint16),
            np.array(j_w, dtype=np.float32),
        )

    @classmethod
    def create_skinned_mannequin(
        cls,
        skeleton: Skeleton,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds a complete, watertight, anatomical stylized 3D humanoid character mesh
        covering all bones and joints of the skeleton.
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

        def get_pos(name: str) -> np.ndarray:
            return skeleton.joints[name].rest_world_pos

        def get_idx(name: str) -> int:
            return joint_to_idx.get(name, 0)

        # 1. Sculpted Continuous Torso (Pelvis, Waist, Chest, Ribcage, Clavicle arch)
        if all(k in joint_to_idx for k in ["Hips", "Spine", "Chest", "Neck"]):
            v, n, idx, ji, jw = cls.generate_torso_ribcage(
                p_hips=get_pos("Hips"),
                p_spine=get_pos("Spine"),
                p_chest=get_pos("Chest"),
                p_neck=get_pos("Neck"),
                j_hips=get_idx("Hips"),
                j_spine=get_idx("Spine"),
                j_chest=get_idx("Chest"),
                j_neck=get_idx("Neck"),
            )
            append_part(v, n, idx, ji, jw)

        # 2. Stylized Humanoid Head with Face Visor
        if "Head" in joint_to_idx and "Neck" in joint_to_idx:
            v, n, idx, ji, jw = cls.generate_stylized_head(
                center=get_pos("Head"),
                head_joint_idx=get_idx("Head"),
                neck_joint_idx=get_idx("Neck"),
            )
            append_part(v, n, idx, ji, jw)

        # 3. Neck Column
        if "Neck" in joint_to_idx and "Head" in joint_to_idx:
            v, n, idx, ji, jw = cls.generate_loft_segment(
                p_start=get_pos("Neck"),
                p_end=get_pos("Head") - np.array([0.0, 0.04, 0.0], dtype=np.float32),
                radius_x_start=0.045,
                radius_z_start=0.045,
                radius_x_end=0.042,
                radius_z_end=0.042,
                j_parent=get_idx("Neck"),
                j_child=get_idx("Head"),
                slices=10,
                rings=4,
            )
            append_part(v, n, idx, ji, jw)

        # 4. Upper Limbs: Deltoids, Biceps, Forearms, Hands
        for side, is_left in [("Left", True), ("Right", False)]:
            sh_name = f"{side}Shoulder"
            arm_name = f"{side}Arm"
            fa_name = f"{side}ForeArm"
            hand_name = f"{side}Hand"

            if arm_name in joint_to_idx and fa_name in joint_to_idx:
                # Deltoid shoulder cap
                if sh_name in joint_to_idx:
                    v, n, idx, ji, jw = cls.generate_deltoid_cap(
                        p_shoulder=get_pos(sh_name),
                        p_arm=get_pos(arm_name),
                        is_left=is_left,
                        shoulder_joint_idx=get_idx(sh_name),
                        arm_joint_idx=get_idx(arm_name),
                        chest_joint_idx=get_idx("Chest"),
                    )
                    append_part(v, n, idx, ji, jw)

                # Upper Arm (Bicep / Tricep)
                v, n, idx, ji, jw = cls.generate_loft_segment(
                    p_start=get_pos(arm_name),
                    p_end=get_pos(fa_name),
                    radius_x_start=0.044,
                    radius_z_start=0.044,
                    radius_x_end=0.034,
                    radius_z_end=0.034,
                    j_parent=get_idx(arm_name),
                    j_child=get_idx(fa_name),
                    slices=12,
                    rings=5,
                    bulge_center=0.45,
                    bulge_amount=0.18,
                )
                append_part(v, n, idx, ji, jw)

                # Forearm
                if hand_name in joint_to_idx:
                    v, n, idx, ji, jw = cls.generate_loft_segment(
                        p_start=get_pos(fa_name),
                        p_end=get_pos(hand_name),
                        radius_x_start=0.036,
                        radius_z_start=0.032,
                        radius_x_end=0.024,
                        radius_z_end=0.020,
                        j_parent=get_idx(fa_name),
                        j_child=get_idx(hand_name),
                        slices=12,
                        rings=5,
                        bulge_center=0.35,
                        bulge_amount=0.22,
                    )
                    append_part(v, n, idx, ji, jw)

                    # Stylized Hand
                    v, n, idx, ji, jw = cls.generate_stylized_hand(
                        p_hand=get_pos(hand_name),
                        p_forearm=get_pos(fa_name),
                        is_left=is_left,
                        hand_joint_idx=get_idx(hand_name),
                        forearm_joint_idx=get_idx(fa_name),
                    )
                    append_part(v, n, idx, ji, jw)

        # 5. Lower Limbs: Thighs, Knee Caps, Calves/Shins, Feet
        for side, is_left in [("Left", True), ("Right", False)]:
            upleg_name = f"{side}UpLeg"
            leg_name = f"{side}Leg"
            foot_name = f"{side}Foot"

            if upleg_name in joint_to_idx and leg_name in joint_to_idx:
                # Thigh (Quads / Hamstrings)
                v, n, idx, ji, jw = cls.generate_loft_segment(
                    p_start=get_pos(upleg_name),
                    p_end=get_pos(leg_name),
                    radius_x_start=0.068,
                    radius_z_start=0.072,
                    radius_x_end=0.046,
                    radius_z_end=0.046,
                    j_parent=get_idx(upleg_name),
                    j_child=get_idx(leg_name),
                    slices=12,
                    rings=6,
                    bulge_center=0.40,
                    bulge_amount=0.20,
                )
                append_part(v, n, idx, ji, jw)

                # Patella Knee Cap
                v, n, idx, ji, jw = cls.generate_patella_knee_cap(
                    p_knee=get_pos(leg_name),
                    knee_joint_idx=get_idx(leg_name),
                    thigh_joint_idx=get_idx(upleg_name),
                    shin_joint_idx=get_idx(leg_name),
                )
                append_part(v, n, idx, ji, jw)

                # Calf & Shin (Anatomical gastrocnemius bulge on back -Z)
                if foot_name in joint_to_idx:
                    v, n, idx, ji, jw = cls.generate_loft_segment(
                        p_start=get_pos(leg_name),
                        p_end=get_pos(foot_name),
                        radius_x_start=0.046,
                        radius_z_start=0.048,
                        radius_x_end=0.030,
                        radius_z_end=0.030,
                        j_parent=get_idx(leg_name),
                        j_child=get_idx(foot_name),
                        slices=12,
                        rings=6,
                        bulge_center=0.30,
                        bulge_amount=0.24,
                        curve_offset_z=-0.015,  # Calf curvature backward
                    )
                    append_part(v, n, idx, ji, jw)

                    # Stylized Foot
                    v, n, idx, ji, jw = cls.generate_stylized_foot(
                        p_foot=get_pos(foot_name),
                        p_leg=get_pos(leg_name),
                        is_left=is_left,
                        foot_joint_idx=get_idx(foot_name),
                        leg_joint_idx=get_idx(leg_name),
                    )
                    append_part(v, n, idx, ji, jw)

        if not all_verts:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint16),
                np.zeros((0, 4), dtype=np.uint16),
                np.zeros((0, 4), dtype=np.float32),
            )

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
            joint_map: Dict[int, float] = {}
            for k in range(4):
                w = float(raw_joint_weights[v_idx, k])
                j = int(raw_joint_indices[v_idx, k])
                if w > 1e-5:
                    joint_map[j] = joint_map.get(j, 0.0) + w

            sorted_pairs = sorted(joint_map.items(), key=lambda p: p[1], reverse=True)
            total_w = sum(p[1] for p in sorted_pairs[:4])
            if total_w < 1e-6:
                clean_joint_weights[v_idx, 0] = 1.0
                clean_joint_indices[v_idx, 0] = 0
            else:
                for k, (j, w) in enumerate(sorted_pairs[:4]):
                    clean_joint_weights[v_idx, k] = float(w / total_w)
                    clean_joint_indices[v_idx, k] = int(j)

        return (
            final_verts.astype(np.float32),
            final_norms.astype(np.float32),
            final_indices.astype(np.uint16),
            clean_joint_indices,
            clean_joint_weights,
        )
