"""
Serpentine articulated spine morphology plugin (snakes, eels, worms).
"""

from __future__ import annotations

import numpy as np
from iris.morphology.base import BaseMorphology


class SerpentineMorphology(BaseMorphology):
    """Continuous N-joint articulated vertebral chain."""

    def __init__(self, segment_count: int = 12, total_length: float = 2.0) -> None:
        self.segment_count = max(4, segment_count)
        self.total_length = total_length
        super().__init__(name="serpentine", root_joint="Vertebra_0")

    def _initialize_topology(self) -> None:
        step = self.total_length / (self.segment_count - 1)
        for i in range(self.segment_count):
            name = f"Vertebra_{i}"
            pos = [0.0, 0.05, float(i * step)]
            self.add_joint(name, np.array(pos, dtype=np.float32))

        for i in range(self.segment_count - 1):
            parent = f"Vertebra_{i}"
            child = f"Vertebra_{i+1}"
            self.add_bone(f"Bone_Vert_{i}_{i+1}", parent, child)
