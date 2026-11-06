"""
Abstract base class and schemas for biological and procedural morphology plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel, Field

from iris.core.exceptions import MorphologyError


class JointLimit(BaseModel):
    """Rotational degrees of freedom limits (degrees)."""

    min_x: float = -180.0
    max_x: float = 180.0
    min_y: float = -180.0
    max_y: float = 180.0
    min_z: float = -180.0
    max_z: float = 180.0


class BoneDefinition(BaseModel):
    """Directed connection from parent joint to child joint."""

    name: str
    parent: str
    child: str
    rest_length: float = Field(default=0.25, gt=0.0)
    limits: Optional[JointLimit] = None


class BaseMorphology(ABC):
    """Defines skeletal topology, parent-child links, and rest proportions for an anatomical family."""

    def __init__(self, name: str, root_joint: str) -> None:
        self.name = name
        self.root_joint = root_joint
        self._joints: Set[str] = set()
        self._bones: Dict[str, BoneDefinition] = {}
        self._parent_map: Dict[str, str] = {}
        self._child_map: Dict[str, List[str]] = {}
        self._rest_pose: Dict[str, np.ndarray] = {}
        self._initialize_topology()
        self.validate()

    @abstractmethod
    def _initialize_topology(self) -> None:
        """Subclasses populate joints, bones, and rest pose positions."""
        pass

    def add_joint(self, name: str, rest_pos: np.ndarray) -> None:
        self._joints.add(name)
        self._rest_pose[name] = np.array(rest_pos, dtype=np.float32)
        if name not in self._child_map:
            self._child_map[name] = []

    def add_bone(self, name: str, parent: str, child: str, limits: Optional[JointLimit] = None) -> None:
        if parent not in self._joints:
            raise MorphologyError(f"Cannot add bone '{name}': parent joint '{parent}' not defined.")
        if child not in self._joints:
            raise MorphologyError(f"Cannot add bone '{name}': child joint '{child}' not defined.")

        rest_len = float(np.linalg.norm(self._rest_pose[child] - self._rest_pose[parent]))
        if rest_len <= 1e-5:
            rest_len = 0.1  # Avoid zero-length bone

        bone = BoneDefinition(name=name, parent=parent, child=child, rest_length=rest_len, limits=limits)
        self._bones[name] = bone
        self._parent_map[child] = parent
        self._child_map.setdefault(parent, []).append(child)

    @property
    def joints(self) -> List[str]:
        return sorted(list(self._joints))

    @property
    def bones(self) -> Dict[str, BoneDefinition]:
        return self._bones

    @property
    def rest_pose(self) -> Dict[str, np.ndarray]:
        return self._rest_pose

    def get_parent(self, joint_name: str) -> Optional[str]:
        return self._parent_map.get(joint_name)

    def get_children(self, joint_name: str) -> List[str]:
        return self._child_map.get(joint_name, [])

    def validate(self) -> None:
        """Validates acyclic tree structure, root connectivity, and uniqueness."""
        if self.root_joint not in self._joints:
            raise MorphologyError(f"Root joint '{self.root_joint}' not found in defined joints.")

        # Check for cycles via DFS
        visited = set()
        recursion_stack = set()

        def dfs(node: str) -> None:
            visited.add(node)
            recursion_stack.add(node)
            for child in self.get_children(node):
                if child not in visited:
                    dfs(child)
                elif child in recursion_stack:
                    raise MorphologyError(f"Cycle detected in skeletal hierarchy involving '{child}'.")
            recursion_stack.remove(node)

        dfs(self.root_joint)

        # Check all joints are reachable
        unreachable = self._joints - visited
        if unreachable:
            raise MorphologyError(f"Unreachable disconnected joints found: {unreachable}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "root_joint": self.root_joint,
            "joints": self.joints,
            "bones": [b.model_dump() for b in self._bones.values()],
            "rest_pose": {k: v.tolist() for k, v in self._rest_pose.items()},
        }
