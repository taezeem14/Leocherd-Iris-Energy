"""
Standard humanoid biological morphology plugin.
"""

from __future__ import annotations

import numpy as np
from iris.morphology.base import BaseMorphology, JointLimit


class HumanMorphology(BaseMorphology):
    """Industry-standard 19-joint humanoid rig compatible with Mixamo, Blender Rigify, and glTF."""

    def __init__(self) -> None:
        super().__init__(name="human", root_joint="Hips")

    def _initialize_topology(self) -> None:
        # Standard anatomical rest pose (T-Pose in meters, +Y up, +X right, +Z forward)
        joints_t_pose = {
            "Hips": [0.0, 1.00, 0.0],
            "Spine": [0.0, 1.20, 0.0],
            "Chest": [0.0, 1.45, 0.0],
            "Neck": [0.0, 1.58, 0.0],
            "Head": [0.0, 1.72, 0.0],
            "LeftShoulder": [-0.10, 1.45, 0.0],
            "LeftArm": [-0.20, 1.45, 0.0],
            "LeftForeArm": [-0.48, 1.45, 0.0],
            "LeftHand": [-0.75, 1.45, 0.0],
            "RightShoulder": [0.10, 1.45, 0.0],
            "RightArm": [0.20, 1.45, 0.0],
            "RightForeArm": [0.48, 1.45, 0.0],
            "RightHand": [0.75, 1.45, 0.0],
            "LeftUpLeg": [-0.10, 0.95, 0.0],
            "LeftLeg": [-0.10, 0.50, 0.0],
            "LeftFoot": [-0.10, 0.10, 0.05],
            "RightUpLeg": [0.10, 0.95, 0.0],
            "RightLeg": [0.10, 0.50, 0.0],
            "RightFoot": [0.10, 0.10, 0.05],
        }

        for j_name, pos in joints_t_pose.items():
            self.add_joint(j_name, np.array(pos, dtype=np.float32))

        # Core Spine Chain
        self.add_bone("Bone_Hips_Spine", "Hips", "Spine")
        self.add_bone("Bone_Spine_Chest", "Spine", "Chest")
        self.add_bone("Bone_Chest_Neck", "Chest", "Neck")
        self.add_bone("Bone_Neck_Head", "Neck", "Head")

        # Left Arm Chain
        self.add_bone("Bone_Chest_LeftShoulder", "Chest", "LeftShoulder")
        self.add_bone("Bone_LeftShoulder_LeftArm", "LeftShoulder", "LeftArm")
        self.add_bone("Bone_LeftArm_LeftForeArm", "LeftArm", "LeftForeArm")
        self.add_bone("Bone_LeftForeArm_LeftHand", "LeftForeArm", "LeftHand")

        # Right Arm Chain
        self.add_bone("Bone_Chest_RightShoulder", "Chest", "RightShoulder")
        self.add_bone("Bone_RightShoulder_RightArm", "RightShoulder", "RightArm")
        self.add_bone("Bone_RightArm_RightForeArm", "RightArm", "RightForeArm")
        self.add_bone("Bone_RightForeArm_RightHand", "RightForeArm", "RightHand")

        # Left Leg Chain
        self.add_bone("Bone_Hips_LeftUpLeg", "Hips", "LeftUpLeg")
        self.add_bone("Bone_LeftUpLeg_LeftLeg", "LeftUpLeg", "LeftLeg")
        self.add_bone("Bone_LeftLeg_LeftFoot", "LeftLeg", "LeftFoot")

        # Right Leg Chain
        self.add_bone("Bone_Hips_RightUpLeg", "Hips", "RightUpLeg")
        self.add_bone("Bone_RightUpLeg_RightLeg", "RightUpLeg", "RightLeg")
        self.add_bone("Bone_RightLeg_RightFoot", "RightLeg", "RightFoot")

    @property
    def relaxed_pose(self) -> Dict[str, np.ndarray]:
        """Anatomical relaxed standing pose with arms resting at sides and upright posture."""
        pose = {k: v.copy() for k, v in self._rest_pose.items()}
        # Natural relaxed arms hanging down along sides
        pose["LeftForeArm"] = np.array([-0.22, 1.17, 0.0], dtype=np.float32)
        pose["LeftHand"] = np.array([-0.22, 0.90, 0.0], dtype=np.float32)
        pose["RightForeArm"] = np.array([0.22, 1.17, 0.0], dtype=np.float32)
        pose["RightHand"] = np.array([0.22, 0.90, 0.0], dtype=np.float32)
        # Feet firmly on ground plane
        pose["LeftFoot"] = np.array([-0.10, 0.10, 0.04], dtype=np.float32)
        pose["RightFoot"] = np.array([0.10, 0.10, 0.04], dtype=np.float32)
        return pose
