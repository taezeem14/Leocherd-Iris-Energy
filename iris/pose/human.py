"""
Humanoid anatomical mappings, virtual joint generation, and kinematic depth constraints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from iris.pose.base import Joint3D, PoseFrame

# Standard 33 MediaPipe BlazePose landmark indices
BLAZEPOSE_KEYPOINTS = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# Standard COCO 17 keypoint indices
COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Standard humanoid rig bone hierarchy (Mixamo / Blender convention)
HUMANOID_RIG_MAPPING = {
    "Hips": ("left_hip", "right_hip"),  # Midpoint
    "Spine": "virtual_spine",
    "Chest": ("left_shoulder", "right_shoulder"),  # Midpoint
    "Neck": "virtual_neck",
    "Head": "nose",
    "LeftShoulder": "left_shoulder",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    "RightShoulder": "right_shoulder",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
}


class HumanPoseSynthesizer:
    """Derives standard articulated 3D humanoid skeleton joints from raw landmark detections."""

    @staticmethod
    def synthesize_rig_joints(raw_joints: Dict[str, Joint3D]) -> Dict[str, Joint3D]:
        """
        Calculates virtual anatomical centers (Hips root, Spine, Chest, Neck)
        and formats coordinate space with metric 3D convention.
        """
        rig: Dict[str, Joint3D] = {}

        def get_pos(name: str) -> np.ndarray:
            if name in raw_joints:
                return raw_joints[name].to_numpy()
            return np.zeros(3, dtype=np.float32)

        def get_conf(name: str) -> float:
            if name in raw_joints:
                return raw_joints[name].confidence
            return 0.5

        # 1. Hips (Root) - midpoint between left and right hip
        lh_pos = get_pos("left_hip")
        rh_pos = get_pos("right_hip")
        hips_pos = (lh_pos + rh_pos) * 0.5
        hips_conf = min(get_conf("left_hip"), get_conf("right_hip"))
        rig["Hips"] = Joint3D(
            name="Hips",
            x=float(hips_pos[0]),
            y=float(hips_pos[1]),
            z=float(hips_pos[2]),
            confidence=hips_conf,
            visibility=1.0,
        )

        # 2. Chest - midpoint between left and right shoulder
        ls_pos = get_pos("left_shoulder")
        rs_pos = get_pos("right_shoulder")
        chest_pos = (ls_pos + rs_pos) * 0.5
        chest_conf = min(get_conf("left_shoulder"), get_conf("right_shoulder"))
        rig["Chest"] = Joint3D(
            name="Chest",
            x=float(chest_pos[0]),
            y=float(chest_pos[1]),
            z=float(chest_pos[2]),
            confidence=chest_conf,
            visibility=1.0,
        )

        # 3. Spine - positioned midway between Hips and Chest
        spine_pos = hips_pos * 0.5 + chest_pos * 0.5
        rig["Spine"] = Joint3D(
            name="Spine",
            x=float(spine_pos[0]),
            y=float(spine_pos[1]),
            z=float(spine_pos[2]),
            confidence=min(hips_conf, chest_conf),
            visibility=1.0,
        )

        # 4. Head & Neck
        head_ref = get_pos("nose") if "nose" in raw_joints else chest_pos + np.array([0, 0.25, 0], dtype=np.float32)
        neck_pos = chest_pos * 0.7 + head_ref * 0.3
        rig["Neck"] = Joint3D(
            name="Neck",
            x=float(neck_pos[0]),
            y=float(neck_pos[1]),
            z=float(neck_pos[2]),
            confidence=min(chest_conf, get_conf("nose")),
            visibility=1.0,
        )
        rig["Head"] = Joint3D(
            name="Head",
            x=float(head_ref[0]),
            y=float(head_ref[1]),
            z=float(head_ref[2]),
            confidence=get_conf("nose"),
            visibility=1.0,
        )

        # 5. Upper Limbs
        rig["LeftShoulder"] = Joint3D(
            name="LeftShoulder",
            x=float(ls_pos[0]),
            y=float(ls_pos[1]),
            z=float(ls_pos[2]),
            confidence=get_conf("left_shoulder"),
        )
        rig["LeftArm"] = Joint3D(
            name="LeftArm",
            x=float(ls_pos[0]),
            y=float(ls_pos[1]),
            z=float(ls_pos[2]),
            confidence=get_conf("left_shoulder"),
        )
        le_pos = get_pos("left_elbow")
        rig["LeftForeArm"] = Joint3D(
            name="LeftForeArm",
            x=float(le_pos[0]),
            y=float(le_pos[1]),
            z=float(le_pos[2]),
            confidence=get_conf("left_elbow"),
        )
        lw_pos = get_pos("left_wrist")
        rig["LeftHand"] = Joint3D(
            name="LeftHand",
            x=float(lw_pos[0]),
            y=float(lw_pos[1]),
            z=float(lw_pos[2]),
            confidence=get_conf("left_wrist"),
        )

        rig["RightShoulder"] = Joint3D(
            name="RightShoulder",
            x=float(rs_pos[0]),
            y=float(rs_pos[1]),
            z=float(rs_pos[2]),
            confidence=get_conf("right_shoulder"),
        )
        rig["RightArm"] = Joint3D(
            name="RightArm",
            x=float(rs_pos[0]),
            y=float(rs_pos[1]),
            z=float(rs_pos[2]),
            confidence=get_conf("right_shoulder"),
        )
        re_pos = get_pos("right_elbow")
        rig["RightForeArm"] = Joint3D(
            name="RightForeArm",
            x=float(re_pos[0]),
            y=float(re_pos[1]),
            z=float(re_pos[2]),
            confidence=get_conf("right_elbow"),
        )
        rw_pos = get_pos("right_wrist")
        rig["RightHand"] = Joint3D(
            name="RightHand",
            x=float(rw_pos[0]),
            y=float(rw_pos[1]),
            z=float(rw_pos[2]),
            confidence=get_conf("right_wrist"),
        )

        # 6. Lower Limbs
        rig["LeftUpLeg"] = Joint3D(
            name="LeftUpLeg",
            x=float(lh_pos[0]),
            y=float(lh_pos[1]),
            z=float(lh_pos[2]),
            confidence=get_conf("left_hip"),
        )
        lk_pos = get_pos("left_knee")
        rig["LeftLeg"] = Joint3D(
            name="LeftLeg",
            x=float(lk_pos[0]),
            y=float(lk_pos[1]),
            z=float(lk_pos[2]),
            confidence=get_conf("left_knee"),
        )
        la_pos = get_pos("left_ankle")
        rig["LeftFoot"] = Joint3D(
            name="LeftFoot",
            x=float(la_pos[0]),
            y=float(la_pos[1]),
            z=float(la_pos[2]),
            confidence=get_conf("left_ankle"),
        )

        rig["RightUpLeg"] = Joint3D(
            name="RightUpLeg",
            x=float(rh_pos[0]),
            y=float(rh_pos[1]),
            z=float(rh_pos[2]),
            confidence=get_conf("right_hip"),
        )
        rk_pos = get_pos("right_knee")
        rig["RightLeg"] = Joint3D(
            name="RightLeg",
            x=float(rk_pos[0]),
            y=float(rk_pos[1]),
            z=float(rk_pos[2]),
            confidence=get_conf("right_knee"),
        )
        ra_pos = get_pos("right_ankle")
        rig["RightFoot"] = Joint3D(
            name="RightFoot",
            x=float(ra_pos[0]),
            y=float(ra_pos[1]),
            z=float(ra_pos[2]),
            confidence=get_conf("right_ankle"),
        )

        return rig
