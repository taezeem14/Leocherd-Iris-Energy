"""
Quadruped animal morphology plugin (mammals, felines, canines).
"""

from __future__ import annotations

import numpy as np
from iris.morphology.base import BaseMorphology


class QuadrupedMorphology(BaseMorphology):
    """Quadruped creature skeleton: horizontal spine, scapula, 4 limb chains, tail."""

    def __init__(self) -> None:
        super().__init__(name="quadruped", root_joint="Pelvis")

    def _initialize_topology(self) -> None:
        joints_rest = {
            "Pelvis": [0.0, 0.70, -0.40],
            "Spine_Lumbar": [0.0, 0.72, -0.20],
            "Spine_Thoracic": [0.0, 0.75, 0.05],
            "Neck": [0.0, 0.85, 0.35],
            "Head": [0.0, 0.95, 0.55],
            "Tail_Base": [0.0, 0.68, -0.55],
            "Tail_Mid": [0.0, 0.55, -0.75],
            "Tail_Tip": [0.0, 0.40, -0.90],
            # Forelimbs (Front)
            "FrontLeft_Scapula": [-0.15, 0.72, 0.10],
            "FrontLeft_Knee": [-0.15, 0.45, 0.10],
            "FrontLeft_Ankle": [-0.15, 0.20, 0.10],
            "FrontLeft_Paw": [-0.15, 0.05, 0.15],
            "FrontRight_Scapula": [0.15, 0.72, 0.10],
            "FrontRight_Knee": [0.15, 0.45, 0.10],
            "FrontRight_Ankle": [0.15, 0.20, 0.10],
            "FrontRight_Paw": [0.15, 0.05, 0.15],
            # Hindlimbs (Back)
            "BackLeft_Hip": [-0.15, 0.68, -0.38],
            "BackLeft_Knee": [-0.15, 0.42, -0.45],
            "BackLeft_Hock": [-0.15, 0.22, -0.40],
            "BackLeft_Paw": [-0.15, 0.05, -0.35],
            "BackRight_Hip": [0.15, 0.68, -0.38],
            "BackRight_Knee": [0.15, 0.42, -0.45],
            "BackRight_Hock": [0.15, 0.22, -0.40],
            "BackRight_Paw": [0.15, 0.05, -0.35],
        }

        for name, pos in joints_rest.items():
            self.add_joint(name, np.array(pos, dtype=np.float32))

        # Spine & Head
        self.add_bone("Bone_Pelvis_Lumbar", "Pelvis", "Spine_Lumbar")
        self.add_bone("Bone_Lumbar_Thoracic", "Spine_Lumbar", "Spine_Thoracic")
        self.add_bone("Bone_Thoracic_Neck", "Spine_Thoracic", "Neck")
        self.add_bone("Bone_Neck_Head", "Neck", "Head")

        # Tail Chain
        self.add_bone("Bone_Pelvis_TailBase", "Pelvis", "Tail_Base")
        self.add_bone("Bone_TailBase_TailMid", "Tail_Base", "Tail_Mid")
        self.add_bone("Bone_TailMid_TailTip", "Tail_Mid", "Tail_Tip")

        # Front Limbs
        self.add_bone("Bone_Thoracic_FLScapula", "Spine_Thoracic", "FrontLeft_Scapula")
        self.add_bone("Bone_FLScapula_FLKnee", "FrontLeft_Scapula", "FrontLeft_Knee")
        self.add_bone("Bone_FLKnee_FLAnkle", "FrontLeft_Knee", "FrontLeft_Ankle")
        self.add_bone("Bone_FLAnkle_FLPaw", "FrontLeft_Ankle", "FrontLeft_Paw")

        self.add_bone("Bone_Thoracic_FRScapula", "Spine_Thoracic", "FrontRight_Scapula")
        self.add_bone("Bone_FRScapula_FRKnee", "FrontRight_Scapula", "FrontRight_Knee")
        self.add_bone("Bone_FRKnee_FRAnkle", "FrontRight_Knee", "FrontRight_Ankle")
        self.add_bone("Bone_FRAnkle_FRPaw", "FrontRight_Ankle", "FrontRight_Paw")

        # Hind Limbs
        self.add_bone("Bone_Pelvis_BLHip", "Pelvis", "BackLeft_Hip")
        self.add_bone("Bone_BLHip_BLKnee", "BackLeft_Hip", "BackLeft_Knee")
        self.add_bone("Bone_BLKnee_BLHock", "BackLeft_Knee", "BackLeft_Hock")
        self.add_bone("Bone_BLHock_BLPaw", "BackLeft_Hock", "BackLeft_Paw")

        self.add_bone("Bone_Pelvis_BRHip", "Pelvis", "BackRight_Hip")
        self.add_bone("Bone_BRHip_BRKnee", "BackRight_Hip", "BackRight_Knee")
        self.add_bone("Bone_BRKnee_BRHock", "BackRight_Knee", "BackRight_Hock")
        self.add_bone("Bone_BRHock_BRPaw", "BackRight_Hock", "BackRight_Paw")
