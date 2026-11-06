"""
User-defined arbitrary graph morphology plugin from JSON or dictionary schema.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from iris.core.exceptions import MorphologyError
from iris.morphology.base import BaseMorphology, BoneDefinition


class CustomMorphology(BaseMorphology):
    """Dynamic morphology instantiated from arbitrary user definitions."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.raw_data = data
        name = data.get("name", "custom")
        root = data.get("root_joint")
        if not root:
            raise MorphologyError("Custom morphology must define a 'root_joint'.")
        super().__init__(name=name, root_joint=root)

    def _initialize_topology(self) -> None:
        joints_dict = self.raw_data.get("joints", {})
        if isinstance(joints_dict, list):
            # Form: [{"name": "A", "position": [x,y,z]}]
            for j in joints_dict:
                self.add_joint(j["name"], np.array(j.get("position", [0, 0, 0]), dtype=np.float32))
        elif isinstance(joints_dict, dict):
            # Form: {"A": [x,y,z]}
            for j_name, pos in joints_dict.items():
                self.add_joint(j_name, np.array(pos, dtype=np.float32))

        bones_list = self.raw_data.get("bones", [])
        for b in bones_list:
            b_name = b.get("name", f"Bone_{b['parent']}_{b['child']}")
            self.add_bone(b_name, b["parent"], b["child"])
