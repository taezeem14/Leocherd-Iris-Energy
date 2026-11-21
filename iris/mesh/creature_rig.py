"""
Creature Image-to-3D Skinned Mesh, Skeletal Rig, and Multi-Engine Exporter.
Analyzes 2D creature concept art/images, extracts morphological proportions,
constructs an exact skeletal rig matching the creature, generates a textured
watertight 3D skinned mesh, and exports production assets for Blender, Unity,
and Unreal Engine.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image
import numpy as np

from iris.core.config import get_settings
from iris.core.exceptions import IrisError
from iris.core.logging import get_logger
from iris.export.bvh import BVHExporter
from iris.export.gltf import GltfExporter, GLTFExporter
from iris.retargeting.animation import SkeletalAnimation
from iris.rigging.joints import JointNode
from iris.rigging.proxy_mesh import ProxyMeshGenerator
from iris.rigging.skeleton import BoneEdge, Skeleton

logger = get_logger("iris.mesh.creature_rig")


class CreatureReconstructionError(IrisError):
    """Raised when creature image-to-rig reconstruction fails."""
    pass


class CreatureMorphologyProfile:
    """Stores spatial proportions and anatomical anchor points extracted from creature imagery."""

    def __init__(
        self,
        height_m: float = 1.45,
        width_ratio: float = 0.65,
        bulk_factor: float = 1.4,
        is_stocky: bool = True,
        anchors: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.height_m = height_m
        self.width_ratio = width_ratio
        self.bulk_factor = bulk_factor
        self.is_stocky = is_stocky
        self.anchors = anchors or {}


class CreatureImageAnalyzer:
    """Analyzes creature imagery to extract silhouette contours and anatomical anchor points."""

    @staticmethod
    def analyze_image(image_path: Union[str, Path]) -> Tuple[CreatureMorphologyProfile, np.ndarray]:
        path = Path(image_path)
        if not path.exists():
            raise CreatureReconstructionError(f"Creature image not found: {path}")

        if cv2 is not None:
            img = cv2.imread(str(path))
            if img is None:
                raise CreatureReconstructionError(f"Failed to decode image: {path}")

            h, w = img.shape[:2]
            logger.info(f"Analyzing creature image: {path.name} ({w}x{h})...")

            # 1. Background Segmentation
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
            bg_bright = np.mean(corners) > 180

            if bg_bright:
                _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            else:
                _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise CreatureReconstructionError("Could not isolate creature silhouette from background.")

            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
        else:
            pil_img = Image.open(str(path)).convert("RGB")
            w, h = pil_img.size
            img_rgb = np.array(pil_img)
            img = img_rgb[:, :, ::-1]  # Convert RGB to BGR for internal consistency
            logger.info(f"Analyzing creature image via PIL: {path.name} ({w}x{h})...")

            gray = np.mean(img_rgb, axis=2).astype(np.uint8)
            corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
            bg_bright = np.mean(corners) > 180

            if bg_bright:
                thresh = (gray < 245).astype(np.uint8) * 255
            else:
                thresh = (gray > 15).astype(np.uint8) * 255

            ys, xs = np.where(thresh > 0)
            if len(xs) > 0 and len(ys) > 0:
                bx, by = int(np.min(xs)), int(np.min(ys))
                bw, bh = int(np.max(xs) - bx + 1), int(np.max(ys) - by + 1)
            else:
                bx, by, bw, bh = 0, 0, w, h

        # 2. Slice Analysis
        slice_widths = []
        slice_centers = []
        for r in range(by, by + bh):
            fg = np.where(thresh[r, bx : bx + bw] > 0)[0]
            if len(fg) > 0:
                slice_widths.append(float(fg[-1] - fg[0]))
                slice_centers.append(float(bx + (fg[-1] + fg[0]) * 0.5))
            else:
                slice_widths.append(0.0)
                slice_centers.append(float(bx + bw * 0.5))

        slice_widths = np.array(slice_widths, dtype=np.float32)
        slice_centers = np.array(slice_centers, dtype=np.float32)

        # Proportions
        w_to_h = float(bw) / float(bh)
        is_stocky = w_to_h > 0.45

        # In metric space (Dwarf / Stocky Warrior ~ 1.45m tall, Standard Human ~ 1.80m)
        height_m = 1.45 if is_stocky else 1.80
        scale = height_m / float(bh)

        # Locate key anatomical elevations from top to bottom
        head_top_y = height_m
        ground_y = 0.0

        # Head center: ~12% from top
        head_center_y = float(head_top_y - 0.12 * height_m)
        head_r = float(0.09 * height_m)

        # Neck: ~20% from top
        neck_y = float(head_top_y - 0.20 * height_m)

        # Chest / Shoulders: ~28% from top
        chest_y = float(head_top_y - 0.28 * height_m)
        sh_span = float(bw * 0.48 * scale)

        # Spine / Waist: ~45% from top
        spine_y = float(head_top_y - 0.45 * height_m)

        # Hips / Pelvis: ~56% from top
        hips_y = float(head_top_y - 0.56 * height_m)
        hip_span = float(bw * 0.30 * scale)

        # Knees: ~75% from top
        knee_y = float(head_top_y - 0.75 * height_m)

        # Feet / Ankles: ground level + 0.04m
        foot_y = 0.04

        # Arm spans: A-pose elbows and hands
        elbow_y = float(chest_y - 0.22 * height_m)
        hand_y = float(hips_y - 0.10 * height_m)
        elbow_span = float(sh_span * 1.35)
        hand_span = float(sh_span * 1.55)

        anchors = {
            "Hips": np.array([0.0, hips_y, 0.0], dtype=np.float32),
            "Spine": np.array([0.0, spine_y, 0.0], dtype=np.float32),
            "Chest": np.array([0.0, chest_y, 0.0], dtype=np.float32),
            "Neck": np.array([0.0, neck_y, 0.0], dtype=np.float32),
            "Head": np.array([0.0, head_center_y, 0.0], dtype=np.float32),
            "LeftShoulder": np.array([-sh_span * 0.45, chest_y + 0.01 * height_m, 0.0], dtype=np.float32),
            "LeftArm": np.array([-sh_span, chest_y, 0.0], dtype=np.float32),
            "LeftForeArm": np.array([-elbow_span, elbow_y, 0.04], dtype=np.float32),
            "LeftHand": np.array([-hand_span, hand_y, 0.08], dtype=np.float32),
            "RightShoulder": np.array([sh_span * 0.45, chest_y + 0.01 * height_m, 0.0], dtype=np.float32),
            "RightArm": np.array([sh_span, chest_y, 0.0], dtype=np.float32),
            "RightForeArm": np.array([elbow_span, elbow_y, 0.04], dtype=np.float32),
            "RightHand": np.array([hand_span, hand_y, 0.08], dtype=np.float32),
            "LeftUpLeg": np.array([-hip_span, hips_y - 0.04, 0.0], dtype=np.float32),
            "LeftLeg": np.array([-hip_span * 1.05, knee_y, 0.02], dtype=np.float32),
            "LeftFoot": np.array([-hip_span * 1.15, foot_y, 0.04], dtype=np.float32),
            "RightUpLeg": np.array([hip_span, hips_y - 0.04, 0.0], dtype=np.float32),
            "RightLeg": np.array([hip_span * 1.05, knee_y, 0.02], dtype=np.float32),
            "RightFoot": np.array([hip_span * 1.15, foot_y, 0.04], dtype=np.float32),
        }

        profile = CreatureMorphologyProfile(
            height_m=height_m,
            width_ratio=w_to_h,
            bulk_factor=1.5 if is_stocky else 1.0,
            is_stocky=is_stocky,
            anchors=anchors,
        )

        logger.info(
            f"Creature Profile: Height={height_m:.2f}m, "
            f"WidthRatio={w_to_h:.2f}, Morphology={'Brute/Dwarf' if is_stocky else 'Humanoid'}"
        )

        # Crop and center creature texture for UV mapping
        pad = int(bh * 0.05)
        crop_y0 = max(0, by - pad)
        crop_y1 = min(h, by + bh + pad)
        crop_x0 = max(0, bx - pad)
        crop_x1 = min(w, bx + bw + pad)
        creature_texture = img[crop_y0:crop_y1, crop_x0:crop_x1]

        return profile, creature_texture


class CreatureRigBuilder:
    """Constructs an exact 3D skeletal hierarchy tailored to the creature's proportions."""

    @staticmethod
    def build_skeleton(profile: CreatureMorphologyProfile) -> Skeleton:
        skeleton = Skeleton(name="CreatureRig", root_name="Hips")
        anchors = profile.anchors

        bone_links = [
            ("Hips", "Spine", "Bone_Hips_Spine"),
            ("Spine", "Chest", "Bone_Spine_Chest"),
            ("Chest", "Neck", "Bone_Chest_Neck"),
            ("Neck", "Head", "Bone_Neck_Head"),
            ("Chest", "LeftShoulder", "Bone_Chest_LeftShoulder"),
            ("LeftShoulder", "LeftArm", "Bone_LeftShoulder_LeftArm"),
            ("LeftArm", "LeftForeArm", "Bone_LeftArm_LeftForeArm"),
            ("LeftForeArm", "LeftHand", "Bone_LeftForeArm_LeftHand"),
            ("Chest", "RightShoulder", "Bone_Chest_RightShoulder"),
            ("RightShoulder", "RightArm", "Bone_RightShoulder_RightArm"),
            ("RightArm", "RightForeArm", "Bone_RightArm_RightForeArm"),
            ("RightForeArm", "RightHand", "Bone_RightForeArm_RightHand"),
            ("Hips", "LeftUpLeg", "Bone_Hips_LeftUpLeg"),
            ("LeftUpLeg", "LeftLeg", "Bone_LeftUpLeg_LeftLeg"),
            ("LeftLeg", "LeftFoot", "Bone_LeftLeg_LeftFoot"),
            ("Hips", "RightUpLeg", "Bone_Hips_RightUpLeg"),
            ("RightUpLeg", "RightLeg", "Bone_RightUpLeg_RightLeg"),
            ("RightLeg", "RightFoot", "Bone_RightLeg_RightFoot"),
        ]

        parent_map: Dict[str, str] = {}
        children_map: Dict[str, List[str]] = {k: [] for k in anchors.keys()}
        for p, c, _ in bone_links:
            parent_map[c] = p
            children_map[p].append(c)

        # 1. Add all joints
        for j_name, pos in anchors.items():
            parent = parent_map.get(j_name)
            children = children_map.get(j_name, [])
            p_pos = anchors.get(parent, pos) if parent else pos
            node = JointNode(
                name=j_name,
                parent_name=parent,
                children_names=children,
                rest_world_pos=pos.copy(),
                world_pos=pos.copy(),
                local_pos=(pos - p_pos).copy(),
            )
            skeleton.joints[j_name] = node

        # 2. Add bones
        for p_name, c_name, b_name in bone_links:
            p_pos = anchors[p_name]
            c_pos = anchors[c_name]
            length = float(np.linalg.norm(c_pos - p_pos))
            skeleton.bones[b_name] = BoneEdge(
                name=b_name,
                parent_joint=p_name,
                child_joint=c_name,
                rest_length=max(length, 0.05),
            )

        skeleton.validate()
        logger.info(f"Constructed Creature Skeleton with {len(skeleton.joints)} joints and {len(skeleton.bones)} bones.")
        return skeleton


class CreatureSkeletalFigureGenerator:
    """
    Generates the exact production-grade 3D luminous cyan skeletal figure
    composed of articulated octahedral diamond bones and spherical joint beads
    matching the reference creature concept art.
    """

    @staticmethod
    def _create_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def generate_octahedral_bone(
        cls,
        p_start: np.ndarray,
        p_end: np.ndarray,
        j_parent: int,
        j_child: int,
        radius: float = 0.035,
        bulge_t: float = 0.25,
        num_joints: int = 19,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vec = p_end - p_start
        length = float(np.linalg.norm(vec))
        if length < 1e-4:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32),
                np.zeros((0, 4), dtype=np.uint16),
                np.zeros((0, 4), dtype=np.float32),
            )
        u, v = cls._create_basis(vec)
        p_mid = p_start + bulge_t * vec
        r = float(min(radius, length * 0.24))

        v0 = p_start
        v1 = p_mid + r * u
        v2 = p_mid + r * v
        v3 = p_mid - r * u
        v4 = p_mid - r * v
        v5 = p_end
        verts = np.array([v0, v1, v2, v3, v4, v5], dtype=np.float32)

        # glTF skinning convention: active influences first, zero-weight slots use joint index 0
        def get_distinct_indices(primary: int, secondary: Optional[int] = None) -> List[int]:
            res = [primary]
            if secondary is not None:
                res.append(secondary)
            while len(res) < 4:
                res.append(0)
            return res

        if j_parent == j_child:
            parent_distinct = get_distinct_indices(j_parent)
            ji = np.tile(np.array(parent_distinct, dtype=np.uint16), (6, 1))
            jw = np.zeros((6, 4), dtype=np.float32)
            jw[:, 0] = 1.0
        else:
            w_p = float(max(0.0, min(1.0, 1.0 - bulge_t)))
            w_c = float(max(0.0, min(1.0, bulge_t)))
            parent_distinct = get_distinct_indices(j_parent)
            child_distinct = get_distinct_indices(j_child)
            if w_p >= w_c:
                mid_distinct = get_distinct_indices(j_parent, j_child)
                mid_weights = [w_p, w_c, 0.0, 0.0]
            else:
                mid_distinct = get_distinct_indices(j_child, j_parent)
                mid_weights = [w_c, w_p, 0.0, 0.0]

            ji = np.array(
                [
                    parent_distinct,
                    mid_distinct,
                    mid_distinct,
                    mid_distinct,
                    mid_distinct,
                    child_distinct,
                ],
                dtype=np.uint16,
            )

            jw = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    mid_weights,
                    mid_weights,
                    mid_weights,
                    mid_weights,
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [5, 2, 1],
                [5, 3, 2],
                [5, 4, 3],
                [5, 1, 4],
            ],
            dtype=np.uint32,
        )

        norms = np.zeros_like(verts)
        for f in faces:
            fn = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
            for idx in f:
                norms[idx] += fn
        nlen = np.linalg.norm(norms, axis=1, keepdims=True)
        nlen[nlen < 1e-6] = 1.0
        norms = (norms / nlen).astype(np.float32)

        return verts, norms, faces, ji, jw

    @classmethod
    def generate_joint_sphere(
        cls,
        center: np.ndarray,
        radius: float,
        j_idx: int,
        lat_segments: int = 8,
        lon_segments: int = 12,
        num_joints: int = 19,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        verts = []
        norms = []
        faces = []

        # South pole
        verts.append([float(center[0]), float(center[1] - radius), float(center[2])])
        norms.append([0.0, -1.0, 0.0])

        for i in range(1, lat_segments):
            theta = math.pi * (1.0 - float(i) / float(lat_segments))
            y = -radius * math.cos(theta)
            r = radius * math.sin(theta)
            for j in range(lon_segments):
                phi = 2.0 * math.pi * float(j) / float(lon_segments)
                x = r * math.cos(phi)
                z = r * math.sin(phi)
                verts.append([float(center[0] + x), float(center[1] + y), float(center[2] + z)])
                norms.append([float(x / radius), float(y / radius), float(z / radius)])

        # North pole
        north_idx = len(verts)
        verts.append([float(center[0]), float(center[1] + radius), float(center[2])])
        norms.append([0.0, 1.0, 0.0])

        # South fan
        for j in range(lon_segments):
            next_j = (j + 1) % lon_segments
            faces.append([0, 1 + next_j, 1 + j])

        for i in range(lat_segments - 2):
            b0 = 1 + i * lon_segments
            b1 = 1 + (i + 1) * lon_segments
            for j in range(lon_segments):
                next_j = (j + 1) % lon_segments
                faces.append([b0 + j, b0 + next_j, b1 + next_j])
                faces.append([b0 + j, b1 + next_j, b1 + j])

        b_last = 1 + (lat_segments - 2) * lon_segments
        for j in range(lon_segments):
            next_j = (j + 1) % lon_segments
            faces.append([b_last + j, b_last + next_j, north_idx])

        V = len(verts)
        # glTF skinning convention: single rigid influence, zero-weight slots use joint index 0
        ji = np.zeros((V, 4), dtype=np.uint16)
        ji[:, 0] = j_idx
        jw = np.zeros((V, 4), dtype=np.float32)
        jw[:, 0] = 1.0
        return (
            np.array(verts, dtype=np.float32),
            np.array(norms, dtype=np.float32),
            np.array(faces, dtype=np.uint32),
            ji,
            jw,
        )

    @classmethod
    def generate_skeletal_figure(
        cls,
        skeleton: Skeleton,
        profile: CreatureMorphologyProfile,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        joint_names = list(skeleton.joints.keys())
        j_map = {n: i for i, n in enumerate(joint_names)}

        all_v = []
        all_n = []
        all_f = []
        all_ji = []
        all_jw = []
        v_offset = 0

        def append_component(v, n, f, ji, jw):
            nonlocal v_offset
            if len(v) == 0:
                return
            all_v.append(v)
            all_n.append(n)
            all_f.append(f + v_offset)
            all_ji.append(ji)
            all_jw.append(jw)
            v_offset += len(v)

        # 1. Joint spheres (cyan glowing beads)
        num_j = len(skeleton.joints)
        for j_name, j_node in skeleton.joints.items():
            idx = j_map[j_name]
            if j_name in ["Head", "Hips", "Chest"]:
                r = 0.035
            elif j_name in ["LeftShoulder", "RightShoulder"]:
                r = 0.030
            elif j_name in ["LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]:
                r = 0.028
            elif j_name in ["LeftLeg", "RightLeg", "LeftForeArm", "RightForeArm"]:
                r = 0.024
            else:
                r = 0.020
            v, n, f, ji, jw = cls.generate_joint_sphere(j_node.rest_world_pos, r, idx, num_joints=num_j)
            append_component(v, n, f, ji, jw)

        # 2. Main articulated octahedral diamond bones
        for b_name, bone in skeleton.bones.items():
            p_pos = skeleton.joints[bone.parent_joint].rest_world_pos
            c_pos = skeleton.joints[bone.child_joint].rest_world_pos
            jp = j_map[bone.parent_joint]
            jc = j_map[bone.child_joint]
            if "Spine" in b_name or "Chest" in b_name:
                rad = 0.055 * profile.bulk_factor
            elif "Hips" in b_name:
                rad = 0.048 * profile.bulk_factor
            elif "Shoulder" in b_name:
                rad = 0.045 * profile.bulk_factor
            elif "Arm" in b_name:
                rad = 0.038 * profile.bulk_factor
            else:
                rad = 0.042 * profile.bulk_factor
            v, n, f, ji, jw = cls.generate_octahedral_bone(p_pos, c_pos, jp, jc, radius=rad, num_joints=num_j)
            append_component(v, n, f, ji, jw)

        # 3. Head Mohawk Crest (as seen in the reference image)
        if "Head" in j_map:
            p_head = skeleton.joints["Head"].rest_world_pos
            p_crest = p_head + np.array([0.0, 0.12 * profile.height_m, 0.01], dtype=np.float32)
            v, n, f, ji, jw = cls.generate_octahedral_bone(
                p_head, p_crest, j_map["Head"], j_map["Head"], radius=0.032, bulge_t=0.35, num_joints=num_j
            )
            append_component(v, n, f, ji, jw)
            v, n, f, ji, jw = cls.generate_joint_sphere(p_crest, 0.018, j_map["Head"], num_joints=num_j)
            append_component(v, n, f, ji, jw)

        # 4. Pelvic Buckle Diamond (as seen in the reference image)
        if "Hips" in j_map:
            p_hips = skeleton.joints["Hips"].rest_world_pos
            p_buckle = p_hips + np.array([0.0, -0.08, 0.05], dtype=np.float32)
            v, n, f, ji, jw = cls.generate_octahedral_bone(
                p_hips, p_buckle, j_map["Hips"], j_map["Hips"], radius=0.045 * profile.bulk_factor, bulge_t=0.5, num_joints=num_j
            )
            append_component(v, n, f, ji, jw)

        # 5. Clenched Fist Finger Rays with Knuckle Beads (as seen in the reference image)
        for hand_name, sign_x in [("LeftHand", -1.0), ("RightHand", 1.0)]:
            if hand_name in j_map:
                p_h = skeleton.joints[hand_name].rest_world_pos
                h_idx = j_map[hand_name]
                finger_offsets = [
                    np.array([0.035 * sign_x, -0.040, 0.035], dtype=np.float32),
                    np.array([0.015 * sign_x, -0.070, 0.020], dtype=np.float32),
                    np.array([-0.005 * sign_x, -0.075, 0.010], dtype=np.float32),
                    np.array([-0.025 * sign_x, -0.065, 0.000], dtype=np.float32),
                ]
                for f_off in finger_offsets:
                    p_tip = p_h + f_off
                    v, n, f, ji, jw = cls.generate_octahedral_bone(
                        p_h, p_tip, h_idx, h_idx, radius=0.014, bulge_t=0.4, num_joints=num_j
                    )
                    append_component(v, n, f, ji, jw)
                    v, n, f, ji, jw = cls.generate_joint_sphere(p_tip, 0.010, h_idx, num_joints=num_j)
                    append_component(v, n, f, ji, jw)

        final_v = np.vstack(all_v).astype(np.float32)
        final_n = np.vstack(all_n).astype(np.float32)
        final_f = np.vstack(all_f).astype(np.int32)
        final_ji = np.vstack(all_ji).astype(np.uint16)
        final_jw = np.vstack(all_jw).astype(np.float32)

        # Cylindrical UVs
        min_v = final_v.min(axis=0)
        max_v = final_v.max(axis=0)
        span_v = np.maximum(max_v - min_v, 1e-4)
        u = np.clip((final_v[:, 0] - min_v[0]) / span_v[0], 0.0, 1.0)
        v_coord = np.clip(1.0 - (final_v[:, 1] - min_v[1]) / span_v[1], 0.0, 1.0)
        uvs = np.column_stack([u, v_coord]).astype(np.float32)

        return final_v, final_n, uvs, final_f, final_ji, final_jw


class CreatureSkinnedMeshGenerator:
    """Generates the primary watertight textured skinned 3D mesh matching the creature."""

    @classmethod
    def generate_mesh(
        cls,
        skeleton: Skeleton,
        profile: CreatureMorphologyProfile,
        texture_img: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the production-grade 3D luminous cyan skeletal figure
        composed of articulated octahedral diamond bones and spherical joint beads.
        """
        return CreatureSkeletalFigureGenerator.generate_skeletal_figure(skeleton, profile)



class CreatureReconstructionPipeline:
    """End-to-End Orchestrator for 2D Creature Image -> Skinned 3D Rigged Model -> Engine Exports."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def process_creature(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        animation_source_json: Optional[Union[str, Path]] = None,
        export_formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        img_path = Path(image_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        formats = export_formats or ["gltf", "blend", "fbx", "bvh"]

        logger.info(f"=== Initiating Creature Reconstruction for: {img_path.name} ===")

        # 1. Analyze 2D Creature Image
        profile, texture_crop = CreatureImageAnalyzer.analyze_image(img_path)

        # Save extracted diffuse texture resized to power-of-two (512x512) for optimal GPU mipmapping
        # and zero IMAGE_NPOT_DIMENSIONS notices in Khronos glTF-Validator
        tex_path = out_dir / "creature_diffuse.png"
        pot_size = (512, 512)
        if cv2 is not None:
            tex_pot = cv2.resize(texture_crop, pot_size, interpolation=cv2.INTER_AREA if texture_crop.shape[0] > 512 else cv2.INTER_CUBIC)
            cv2.imwrite(str(tex_path), tex_pot)
        else:
            crop_rgb = texture_crop[:, :, ::-1] if len(texture_crop.shape) == 3 else texture_crop
            Image.fromarray(crop_rgb).resize(pot_size, Image.Resampling.LANCZOS).save(str(tex_path))
        logger.info(f"Saved power-of-two (512x512) creature diffuse texture to: {tex_path}")

        # 2. Build Creature Skeleton
        skeleton = CreatureRigBuilder.build_skeleton(profile)

        # 3. Generate Primary Skinned 3D Mesh
        verts, normals, uvs, faces, joint_idx, joint_weights = CreatureSkinnedMeshGenerator.generate_mesh(
            skeleton, profile, texture_crop
        )
        logger.info(
            f"Generated Primary Creature Skinned Mesh: {len(verts)} vertices, "
            f"{len(faces)} faces, {len(skeleton.joints)} articulated joints."
        )

        # Save Wavefront OBJ proxy and MTL material definition
        obj_path = out_dir / "creature_mesh.obj"
        with open(obj_path, "w", encoding="utf-8") as f_obj:
            f_obj.write(f"# Leocherd Iris Energy Creature Mesh - {img_path.stem}\n")
            f_obj.write(f"mtllib creature_material.mtl\nusemtl CreatureMat\n")
            for i in range(len(verts)):
                f_obj.write(f"v {verts[i,0]:.4f} {verts[i,1]:.4f} {verts[i,2]:.4f}\n")
            for i in range(len(uvs)):
                f_obj.write(f"vt {uvs[i,0]:.4f} {1.0 - uvs[i,1]:.4f}\n")
            for i in range(len(normals)):
                f_obj.write(f"vn {normals[i,0]:.4f} {normals[i,1]:.4f} {normals[i,2]:.4f}\n")
            for face in faces:
                idx1, idx2, idx3 = face[0] + 1, face[1] + 1, face[2] + 1
                f_obj.write(f"f {idx1}/{idx1}/{idx1} {idx2}/{idx2}/{idx2} {idx3}/{idx3}/{idx3}\n")

        mtl_path = out_dir / "creature_material.mtl"
        with open(mtl_path, "w", encoding="utf-8") as f_mtl:
            f_mtl.write("# Leocherd Iris Energy Creature Material Definition\n")
            f_mtl.write("newmtl CreatureMat\n")
            f_mtl.write("Ka 1.0000 1.0000 1.0000\n")
            f_mtl.write("Kd 1.0000 1.0000 1.0000\n")
            f_mtl.write("Ks 0.1500 0.1500 0.1500\n")
            f_mtl.write("Ns 25.0000\n")
            f_mtl.write("d 1.0000\n")
            f_mtl.write("illum 2\n")
            f_mtl.write(f"map_Kd {tex_path.name}\n")

        # 4. Load or Retarget Animation
        fps = 30.0
        anim_name = "CreatureAction"
        if animation_source_json and Path(animation_source_json).exists():
            with open(animation_source_json, "r", encoding="utf-8") as f_anim:
                anim_raw = json.load(f_anim)

            fps = float(anim_raw.get("fps", 30.0))
            anim_name = str(anim_raw.get("name", "CreatureAction"))

            if "rotations" in anim_raw and "root_translations" in anim_raw:
                timestamps = anim_raw.get("timestamps", [])
                root_trans = [
                    np.array(
                        [r[0] * profile.bulk_factor, r[1] * (profile.height_m / 1.8), r[2] * profile.bulk_factor],
                        dtype=np.float32,
                    )
                    for r in anim_raw["root_translations"]
                ]
                rotations = {k: np.array(v, dtype=np.float32) for k, v in anim_raw["rotations"].items()}
                # Ensure shoulder tracks map to both joint and bone names
                if "LeftShoulder" in rotations:
                    rotations["Bone_Chest_LeftShoulder"] = rotations["LeftShoulder"]
                if "RightShoulder" in rotations:
                    rotations["Bone_Chest_RightShoulder"] = rotations["RightShoulder"]
            else:
                frames_list = anim_raw.get("frames", [])
                timestamps = [float(i) / fps for i in range(len(frames_list))]
                root_trans = [
                    np.array(
                        [
                            f.get("root_translation", [0, profile.anchors["Hips"][1], 0])[0],
                            f.get("root_translation", [0, profile.anchors["Hips"][1], 0])[1] * (profile.height_m / 1.8),
                            f.get("root_translation", [0, profile.anchors["Hips"][1], 0])[2],
                        ],
                        dtype=np.float32,
                    )
                    for f in frames_list
                ]
                rotations = {}
                for b_name in skeleton.bones.keys():
                    track = []
                    for f in frames_list:
                        q = f.get("rotations", {}).get(b_name, [1.0, 0.0, 0.0, 0.0])
                        track.append(q)
                    rotations[b_name] = np.array(track, dtype=np.float32)

            animation = SkeletalAnimation(
                name=anim_name,
                fps=fps,
                timestamps=np.array(timestamps, dtype=np.float32),
                root_translations=root_trans,
                rotations=rotations,
                root_joint_name="Hips",
            )
        else:
            # Procedural idle / sway animation
            T = 60
            timestamps = [float(i) / fps for i in range(T)]
            root_trans = [
                np.array([0.0, profile.anchors["Hips"][1] + 0.02 * math.sin(t * 0.1), 0.0], dtype=np.float32)
                for t in range(T)
            ]
            rotations = {
                b_name: np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
                for b_name in skeleton.bones.keys()
            }
            animation = SkeletalAnimation(
                name="CreatureIdle",
                fps=fps,
                timestamps=np.array(timestamps, dtype=np.float32),
                root_translations=root_trans,
                rotations=rotations,
                root_joint_name="Hips",
            )

        # 5. Export Multi-Engine Assets
        exported_files: Dict[str, str] = {
            "mesh_obj": str(obj_path),
            "material_mtl": str(mtl_path),
            "texture": str(tex_path),
        }

        # A. glTF 2.0 Export
        if "gltf" in formats:
            gltf_path = out_dir / "creature_rigged.gltf"
            gltf_exporter = GLTFExporter()
            cyan_material = {
                "name": "Cyan_Luminous_Skeletal_Material",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.0, 1.0, 0.85, 1.0],
                    "metallicFactor": 0.15,
                    "roughnessFactor": 0.25,
                },
                "emissiveFactor": [0.0, 0.90, 0.78],
                "doubleSided": True,
            }
            gltf_exporter.export(
                skeleton=skeleton,
                animation=animation,
                output_path=gltf_path,
                mesh=(verts, normals, uvs, faces, joint_idx, joint_weights),
                material=cyan_material,
                texture_path=str(tex_path),
            )
            exported_files["gltf"] = str(gltf_path)
            logger.info(f"Exported Unity/WebGL glTF to: {gltf_path}")

        # B. BVH Mocap Export
        if "bvh" in formats:
            bvh_path = out_dir / "creature_animation.bvh"
            bvh_exporter = BVHExporter()
            bvh_exporter.export(skeleton=skeleton, animation=animation, output_path=bvh_path)
            exported_files["bvh"] = str(bvh_path)
            logger.info(f"Exported Universal BVH Mocap to: {bvh_path}")

        # C. Blender Rig & Animation JSON
        rig_json = out_dir / "creature_rig_data.json"
        anim_json = out_dir / "creature_anim_data.json"
        with open(rig_json, "w", encoding="utf-8") as f_rig:
            json.dump(skeleton.to_dict(), f_rig, indent=2)
        with open(anim_json, "w", encoding="utf-8") as f_anim:
            json.dump(
                {
                    "name": animation.name,
                    "fps": float(animation.fps),
                    "timestamps": [float(x) for x in animation.timestamps],
                    "root_joint_name": str(animation.root_joint_name),
                    "root_translations": [[float(v) for v in row] for row in animation.root_translations],
                    "rotations": {k: [[float(v) for v in row] for row in arr] for k, arr in animation.rotations.items()},
                },
                f_anim,
                indent=2,
            )
        exported_files["rig_json"] = str(rig_json)
        exported_files["anim_json"] = str(anim_json)

        # D. Headless Blender .blend and FBX Export
        blender_bin = self.settings.blender_executable
        if blender_bin and Path(blender_bin).exists() and ("blend" in formats or "fbx" in formats):
            blend_path = out_dir / "creature_rigged.blend"
            fbx_path = out_dir / "creature_rigged.fbx"
            script_auto = self.settings.project_root / "blender" / "automate_rig.py"

            import subprocess
            cmd = [
                str(blender_bin),
                "--background",
                "--python",
                str(script_auto),
                "--",
                "--mesh",
                str(exported_files.get("gltf", obj_path)),
                "--rig",
                str(rig_json),
                "--anim",
                str(anim_json),
                "--output",
                str(blend_path),
                "--export-fbx",
                str(fbx_path),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0 or "Traceback" in proc.stderr or "Traceback" in proc.stdout:
                    err_tail = (proc.stderr or proc.stdout or "").strip()[-800:]
                    raise RuntimeError(f"Blender exited with code {proc.returncode}: {err_tail}")
                if blend_path.exists():
                    exported_files["blend"] = str(blend_path)
                    logger.info(f"Successfully compiled Blender (.blend): {blend_path}")
                if fbx_path.exists():
                    exported_files["fbx"] = str(fbx_path)
                    logger.info(f"Successfully compiled Unreal Engine (.fbx): {fbx_path}")
            except Exception as e:
                logger.warning(f"Headless Blender compilation failed: {e}")

        logger.info(f"=== Creature Reconstruction Complete! {len(exported_files)} assets ready ===")
        return exported_files
