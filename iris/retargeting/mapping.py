"""
Retargeting joint mapping tables and coordinate alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from iris.rigging.orientation import QuaternionMath


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
