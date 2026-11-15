"""
Retargeting joint mapping tables and coordinate alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np

from iris.rigging.orientation import QuaternionMath

if TYPE_CHECKING:
    from iris.rigging.skeleton import Skeleton


def compute_rest_pose_offsets(
    source_skel: Skeleton,
    target_skel: Skeleton,
    joint_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """
    Computes rotational rest-pose offsets for each mapped joint pair between source and target skeletons.

    For each mapped pair (src, tgt), determines the rest bone direction vectors and calculates
    the minimal-arc rotation quaternion that aligns the source rest orientation with the target.
    """
    offsets: Dict[str, np.ndarray] = {}

    for src_name, tgt_name in joint_map.items():
        if src_name not in source_skel.joints or tgt_name not in target_skel.joints:
            offsets[src_name] = QuaternionMath.identity()
            continue

        src_node = source_skel.joints[src_name]
        tgt_node = target_skel.joints[tgt_name]

        # Determine source rest vector (child direction or parent direction)
        if len(src_node.children_names) > 0:
            child_name = src_node.children_names[0]
            v_src = source_skel.joints[child_name].rest_world_pos - src_node.rest_world_pos
        elif src_node.parent_name is not None:
            v_src = src_node.rest_world_pos - source_skel.joints[src_node.parent_name].rest_world_pos
        else:
            v_src = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Determine target rest vector
        if len(tgt_node.children_names) > 0:
            child_name = tgt_node.children_names[0]
            v_tgt = target_skel.joints[child_name].rest_world_pos - tgt_node.rest_world_pos
        elif tgt_node.parent_name is not None:
            v_tgt = tgt_node.rest_world_pos - target_skel.joints[tgt_node.parent_name].rest_world_pos
        else:
            v_tgt = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        norm_src = float(np.linalg.norm(v_src))
        norm_tgt = float(np.linalg.norm(v_tgt))

        if norm_src > 1e-4 and norm_tgt > 1e-4:
            offset_q = QuaternionMath.rotation_between_vectors(v_src, v_tgt)
        else:
            offset_q = QuaternionMath.identity()

        offsets[src_name] = offset_q

    return offsets


@dataclass
class RigMapping:
    """Defines name correspondence and rotational offset between source and target rigs."""

    source_rig_name: str
    target_rig_name: str
    joint_map: Dict[str, str] = field(default_factory=dict)
    # Orientation offsets (quaternion [w, x, y, z])
    rotational_offsets: Dict[str, np.ndarray] = field(default_factory=dict)

    def map_joint(self, source_name: str) -> Optional[str]:
        return self.joint_map.get(source_name)

    def get_offset(self, source_name: str) -> np.ndarray:
        return self.rotational_offsets.get(source_name, QuaternionMath.identity())

    @classmethod
    def create_human_to_quadruped_mapping(cls) -> RigMapping:
        """Constructs canonical topology mapping from Humanoid to Quadruped."""
        joint_map = {
            # Spine & Axial
            "Hips": "Pelvis",
            "Spine": "Spine_Lumbar",
            "Chest": "Spine_Thoracic",
            "Neck": "Neck",
            "Head": "Head",
            # Front Limbs (Arms -> Front legs)
            "LeftShoulder": "FrontLeft_Scapula",
            "LeftArm": "FrontLeft_Knee",
            "LeftForeArm": "FrontLeft_Ankle",
            "LeftHand": "FrontLeft_Paw",
            "RightShoulder": "FrontRight_Scapula",
            "RightArm": "FrontRight_Knee",
            "RightForeArm": "FrontRight_Ankle",
            "RightHand": "FrontRight_Paw",
            # Hind Limbs (Legs -> Hind legs)
            "LeftUpLeg": "BackLeft_Hip",
            "LeftLeg": "BackLeft_Knee",
            "LeftFoot": "BackLeft_Paw",
            "RightUpLeg": "BackRight_Hip",
            "RightLeg": "BackRight_Knee",
            "RightFoot": "BackRight_Paw",
        }
        return cls(
            source_rig_name="human",
            target_rig_name="quadruped",
            joint_map=joint_map,
        )

    @classmethod
    def create_quadruped_to_human_mapping(cls) -> RigMapping:
        """Constructs inverse topology mapping from Quadruped to Humanoid."""
        joint_map = {
            "Pelvis": "Hips",
            "Spine_Lumbar": "Spine",
            "Spine_Thoracic": "Chest",
            "Neck": "Neck",
            "Head": "Head",
            "FrontLeft_Scapula": "LeftShoulder",
            "FrontLeft_Knee": "LeftArm",
            "FrontLeft_Ankle": "LeftForeArm",
            "FrontLeft_Paw": "LeftHand",
            "FrontRight_Scapula": "RightShoulder",
            "FrontRight_Knee": "RightArm",
            "FrontRight_Ankle": "RightForeArm",
            "FrontRight_Paw": "RightHand",
            "BackLeft_Hip": "LeftUpLeg",
            "BackLeft_Knee": "LeftLeg",
            "BackLeft_Paw": "LeftFoot",
            "BackRight_Hip": "RightUpLeg",
            "BackRight_Knee": "RightLeg",
            "BackRight_Paw": "RightFoot",
        }
        return cls(
            source_rig_name="quadruped",
            target_rig_name="human",
            joint_map=joint_map,
        )

    @classmethod
    def create_human_to_serpentine_mapping(cls) -> RigMapping:
        """Constructs topology mapping from Humanoid to Serpentine vertebral chain."""
        joint_map = {
            "Hips": "Vertebra_0",
            "Spine": "Vertebra_3",
            "Chest": "Vertebra_6",
            "Neck": "Vertebra_9",
            "Head": "Vertebra_11",
        }
        return cls(
            source_rig_name="human",
            target_rig_name="serpentine",
            joint_map=joint_map,
        )

    @classmethod
    def create_human_to_tentacular_mapping(cls) -> RigMapping:
        """Constructs topology mapping from Humanoid to Tentacular radial appendages."""
        joint_map = {
            "Hips": "CentralBody",
            # Limbs map to radial tentacles
            "LeftArm": "Tentacle_0_Seg_0",
            "LeftForeArm": "Tentacle_0_Seg_1",
            "LeftHand": "Tentacle_0_Seg_2",
            "RightArm": "Tentacle_1_Seg_0",
            "RightForeArm": "Tentacle_1_Seg_1",
            "RightHand": "Tentacle_1_Seg_2",
            "LeftUpLeg": "Tentacle_2_Seg_0",
            "LeftLeg": "Tentacle_2_Seg_1",
            "LeftFoot": "Tentacle_2_Seg_2",
            "RightUpLeg": "Tentacle_3_Seg_0",
            "RightLeg": "Tentacle_3_Seg_1",
            "RightFoot": "Tentacle_3_Seg_2",
            # Spine and Head map to remaining tentacles
            "Spine": "Tentacle_4_Seg_0",
            "Chest": "Tentacle_4_Seg_1",
            "Head": "Tentacle_4_Seg_2",
            "Neck": "Tentacle_5_Seg_0",
        }
        return cls(
            source_rig_name="human",
            target_rig_name="tentacular",
            joint_map=joint_map,
        )

    @classmethod
    def create_human_to_avian_mapping(cls) -> RigMapping:
        """Constructs topology mapping from Humanoid to Avian / Biped."""
        joint_map = {
            "Hips": "Hips",
            "Spine": "Spine",
            "Chest": "Chest",
            "Neck": "Neck",
            "Head": "Head",
            "LeftArm": "LeftWing_Base",
            "LeftForeArm": "LeftWing_Mid",
            "LeftHand": "LeftWing_Tip",
            "RightArm": "RightWing_Base",
            "RightForeArm": "RightWing_Mid",
            "RightHand": "RightWing_Tip",
            "LeftUpLeg": "LeftThigh",
            "LeftLeg": "LeftKnee",
            "LeftFoot": "LeftFoot",
            "RightUpLeg": "RightThigh",
            "RightLeg": "RightKnee",
            "RightFoot": "RightFoot",
        }
        return cls(
            source_rig_name="human",
            target_rig_name="avian",
            joint_map=joint_map,
        )

    @classmethod
    def build_default_mapping(cls, source_morphology: str, target_morphology: str) -> RigMapping:
        """
        Resolves the appropriate pre-configured RigMapping for the given source and target
        morphology names.
        """
        src = source_morphology.lower()
        tgt = target_morphology.lower()

        if src in ("human", "biped") and tgt == "quadruped":
            return cls.create_human_to_quadruped_mapping()
        elif src == "quadruped" and tgt in ("human", "biped"):
            return cls.create_quadruped_to_human_mapping()
        elif src in ("human", "biped") and tgt == "serpentine":
            return cls.create_human_to_serpentine_mapping()
        elif src in ("human", "biped") and tgt == "tentacular":
            return cls.create_human_to_tentacular_mapping()
        elif src in ("human", "biped") and tgt in ("avian", "bird"):
            return cls.create_human_to_avian_mapping()
        elif src == tgt:
            # Identity mapping
            return cls(source_rig_name=src, target_rig_name=tgt)
        else:
            return cls(source_rig_name=src, target_rig_name=tgt)


create_human_to_quadruped_mapping = RigMapping.create_human_to_quadruped_mapping
create_quadruped_to_human_mapping = RigMapping.create_quadruped_to_human_mapping
create_human_to_serpentine_mapping = RigMapping.create_human_to_serpentine_mapping
create_human_to_tentacular_mapping = RigMapping.create_human_to_tentacular_mapping
create_human_to_avian_mapping = RigMapping.create_human_to_avian_mapping
build_default_mapping = RigMapping.build_default_mapping
