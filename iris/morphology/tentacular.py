"""
Radial tentacular morphology plugin (cephalopods, jellyfish, eldritch appendages).
"""

from __future__ import annotations

import math
import numpy as np
from iris.morphology.base import BaseMorphology


class TentacularMorphology(BaseMorphology):
    """Central hub body with M radial articulated tentacle chains of length K."""

    def __init__(self, tentacle_count: int = 6, segments_per_tentacle: int = 4) -> None:
        self.tentacle_count = max(3, tentacle_count)
        self.segments_per_tentacle = max(2, segments_per_tentacle)
        super().__init__(name="tentacular", root_joint="CentralBody")

    def _initialize_topology(self) -> None:
        # Central Root
        self.add_joint("CentralBody", np.array([0.0, 1.0, 0.0], dtype=np.float32))

        angle_step = (2.0 * math.pi) / self.tentacle_count
        seg_len = 0.25

        for t_idx in range(self.tentacle_count):
            angle = t_idx * angle_step
            dir_x = math.cos(angle)
            dir_z = math.sin(angle)

            parent_joint = "CentralBody"
            for s_idx in range(self.segments_per_tentacle):
                dist = (s_idx + 1) * seg_len
                joint_name = f"Tentacle_{t_idx}_Seg_{s_idx}"
                pos = [
                    dir_x * dist,
                    1.0 - (s_idx * 0.1),
                    dir_z * dist,
                ]
                self.add_joint(joint_name, np.array(pos, dtype=np.float32))
                self.add_bone(f"Bone_T{t_idx}_S{s_idx}", parent_joint, joint_name)
                parent_joint = joint_name
