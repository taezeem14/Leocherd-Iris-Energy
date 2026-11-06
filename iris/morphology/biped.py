"""
Generalized biped morphology with variable proportions and optional tail.
"""

from __future__ import annotations

import numpy as np
from iris.morphology.base import BaseMorphology


class BipedMorphology(BaseMorphology):
    """Generalized biped (birds, theropods, anthropomorphic creatures)."""

    def __init__(self, with_tail: bool = True) -> None:
        self.with_tail = with_tail
        super().__init__(name="biped", root_joint="Hips")

    def _initialize_topology(self) -> None:
        joints = {
            "Hips": [0.0, 0.90, -0.10],
            "Spine": [0.0, 1.05, 0.05],
            "Chest": [0.0, 1.25, 0.20],
            "Neck": [0.0, 1.40, 0.35],
            "Head": [0.0, 1.50, 0.45],
            "LeftWing_Base": [-0.15, 1.25, 0.20],
            "LeftWing_Mid": [-0.45, 1.20, 0.10],
            "LeftWing_Tip": [-0.75, 1.10, -0.05],
            "RightWing_Base": [0.15, 1.25, 0.20],
            "RightWing_Mid": [0.45, 1.20, 0.10],
            "RightWing_Tip": [0.75, 1.10, -0.05],
            "LeftThigh": [-0.12, 0.85, -0.10],
            "LeftKnee": [-0.12, 0.45, 0.05],
            "LeftAnkle": [-0.12, 0.15, -0.10],
            "LeftFoot": [-0.12, 0.05, 0.05],
            "RightThigh": [0.12, 0.85, -0.10],
            "RightKnee": [0.12, 0.45, 0.05],
            "RightAnkle": [0.12, 0.15, -0.10],
            "RightFoot": [0.12, 0.05, 0.05],
        }
        if self.with_tail:
            joints.update(
                {
                    "Tail_1": [0.0, 0.85, -0.30],
                    "Tail_2": [0.0, 0.75, -0.60],
                    "Tail_3": [0.0, 0.60, -0.90],
                }
            )

        for name, pos in joints.items():
            self.add_joint(name, np.array(pos, dtype=np.float32))

        self.add_bone("Bone_Hips_Spine", "Hips", "Spine")
        self.add_bone("Bone_Spine_Chest", "Spine", "Chest")
        self.add_bone("Bone_Chest_Neck", "Chest", "Neck")
        self.add_bone("Bone_Neck_Head", "Neck", "Head")

        self.add_bone("Bone_Chest_LWBase", "Chest", "LeftWing_Base")
        self.add_bone("Bone_LWBase_LWMid", "LeftWing_Base", "LeftWing_Mid")
        self.add_bone("Bone_LWMid_LWTip", "LeftWing_Mid", "LeftWing_Tip")

        self.add_bone("Bone_Chest_RWBase", "Chest", "RightWing_Base")
        self.add_bone("Bone_RWBase_RWMid", "RightWing_Base", "RightWing_Mid")
        self.add_bone("Bone_RWMid_RWTip", "RightWing_Mid", "RightWing_Tip")

        self.add_bone("Bone_Hips_LThigh", "Hips", "LeftThigh")
        self.add_bone("Bone_LThigh_LKnee", "LeftThigh", "LeftKnee")
        self.add_bone("Bone_LKnee_LAnkle", "LeftKnee", "LeftAnkle")
        self.add_bone("Bone_LAnkle_LFoot", "LeftAnkle", "LeftFoot")

        self.add_bone("Bone_Hips_RThigh", "Hips", "RightThigh")
        self.add_bone("Bone_RThigh_RKnee", "RightThigh", "RightKnee")
        self.add_bone("Bone_RKnee_RAnkle", "RightKnee", "RightAnkle")
        self.add_bone("Bone_RAnkle_RFoot", "RightAnkle", "RightFoot")

        if self.with_tail:
            self.add_bone("Bone_Hips_Tail1", "Hips", "Tail_1")
            self.add_bone("Bone_Tail1_Tail2", "Tail_1", "Tail_2")
            self.add_bone("Bone_Tail2_Tail3", "Tail_2", "Tail_3")
