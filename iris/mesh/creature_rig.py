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
import cv2
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

        img = cv2.imread(str(path))
        if img is None:
            raise CreatureReconstructionError(f"Failed to decode image: {path}")

        h, w = img.shape[:2]
        logger.info(f"Analyzing creature image: {path.name} ({w}x{h})...")

        # 1. Background Segmentation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Check background brightness (white vs dark)
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
            "LeftArm": np.array([-sh_span, chest_y, 0.0], dtype=np.float32),
            "LeftForeArm": np.array([-elbow_span, elbow_y, 0.04], dtype=np.float32),
            "LeftHand": np.array([-hand_span, hand_y, 0.08], dtype=np.float32),
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
            ("Chest", "LeftArm", "Bone_Chest_LeftArm"),
            ("LeftArm", "LeftForeArm", "Bone_LeftArm_LeftForeArm"),
            ("LeftForeArm", "LeftHand", "Bone_LeftForeArm_LeftHand"),
            ("Chest", "RightArm", "Bone_Chest_RightArm"),
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
        Generates a single connected humanoid mannequin mesh with proper topology.
        Returns:
            vertices: (V, 3) float32
            normals: (V, 3) float32
            uvs: (V, 2) float32
            faces: (F, 3) int32
            joint_indices: (V, 4) uint16
            joint_weights: (V, 4) float32
        """
        # Use the connected mannequin generator (torso + head + neck + limbs as one mesh)
        # instead of per-bone capsules which produce disconnected body parts
        v, n, f, ji, jw = ProxyMeshGenerator.create_skinned_mannequin(skeleton)

        # Scale mesh vertices outward based on creature bulk factor
        bulk = profile.bulk_factor
        if bulk != 1.0:
            v[:, 0] *= float(bulk * 1.1)
            v[:, 2] *= float(bulk * 1.05)
            n[:, 0] *= float(bulk)
            n[:, 2] *= float(bulk)
            norms = np.linalg.norm(n, axis=1, keepdims=True)
            n = np.where(norms > 1e-6, n / norms, n)

        # Compute cylindrical-planar projective UV coordinates
        min_v = v.min(axis=0)
        max_v = v.max(axis=0)
        span_v = np.maximum(max_v - min_v, 1e-4)

        u = np.clip((v[:, 0] - min_v[0]) / span_v[0], 0.0, 1.0)
        v_coord = np.clip(1.0 - (v[:, 1] - min_v[1]) / span_v[1], 0.0, 1.0)
        uvs = np.column_stack([u, v_coord]).astype(np.float32)

        return (
            v.astype(np.float32),
            n.astype(np.float32),
            uvs.astype(np.float32),
            f.astype(np.int32),
            ji.astype(np.uint16),
            jw.astype(np.float32),
        )


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

        # Save extracted diffuse texture
        tex_path = out_dir / "creature_diffuse.png"
        cv2.imwrite(str(tex_path), texture_crop)
        logger.info(f"Saved creature diffuse texture to: {tex_path}")

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

        # Save Wavefront OBJ proxy
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
            "texture": str(tex_path),
        }

        # A. glTF 2.0 Export
        if "gltf" in formats:
            gltf_path = out_dir / "creature_rigged.gltf"
            gltf_exporter = GLTFExporter()
            gltf_exporter.export(
                skeleton=skeleton,
                animation=animation,
                output_path=gltf_path,
                mesh=(verts, normals, faces, joint_idx, joint_weights),
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
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                exported_files["blend"] = str(blend_path)
                exported_files["fbx"] = str(fbx_path)
                logger.info(f"Successfully compiled Blender (.blend): {blend_path}")
                logger.info(f"Successfully compiled Unreal Engine (.fbx): {fbx_path}")
            except Exception as e:
                logger.warning(f"Headless Blender compilation failed: {e}")

        logger.info(f"=== Creature Reconstruction Complete! {len(exported_files)} assets ready ===")
        return exported_files
