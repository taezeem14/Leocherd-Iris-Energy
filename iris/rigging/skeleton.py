"""
Graph-based 3D skeleton architecture, forward kinematics, and kinematic regularization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import numpy as np

from iris.core.exceptions import SkeletonError
from iris.morphology.base import BaseMorphology
from iris.pose.base import PoseFrame
from iris.rigging.joints import BoneEdge, JointNode
from iris.rigging.orientation import QuaternionMath


class Skeleton:
    """Directed acyclic tree representing an articulated 3D character rig."""

    def __init__(self, name: str, root_name: str) -> None:
        self.name = name
        self.root_name = root_name
        self.joints: Dict[str, JointNode] = {}
        self.bones: Dict[str, BoneEdge] = {}

    @classmethod
    def from_morphology(cls, morphology: BaseMorphology) -> "Skeleton":
        skel = cls(name=morphology.name, root_name=morphology.root_joint)

        # 1. Create JointNodes
        for j_name in morphology.joints:
            parent = morphology.get_parent(j_name)
            children = morphology.get_children(j_name)
            rest_pos = morphology.rest_pose.get(j_name, np.zeros(3, dtype=np.float32))

            node = JointNode(
                name=j_name,
                parent_name=parent,
                children_names=children,
                rest_world_pos=rest_pos.copy(),
                world_pos=rest_pos.copy(),
                local_pos=rest_pos.copy() if parent is None else (rest_pos - morphology.rest_pose[parent]),
            )
            skel.joints[j_name] = node

        # 2. Create BoneEdges
        for b_name, b_def in morphology.bones.items():
            edge = BoneEdge(
                name=b_name,
                parent_joint=b_def.parent,
                child_joint=b_def.child,
                rest_length=b_def.rest_length,
            )
            skel.bones[b_name] = edge

        skel.validate()
        return skel

    def validate(self) -> None:
        if self.root_name not in self.joints:
            raise SkeletonError(f"Root joint '{self.root_name}' does not exist in skeleton joints.")

        # Check for cycles
        visited = set()
        stack = set()

        def dfs(name: str) -> None:
            visited.add(name)
            stack.add(name)
            node = self.joints[name]
            for child in node.children_names:
                if child not in visited:
                    dfs(child)
                elif child in stack:
                    raise SkeletonError(f"Cycle detected in skeleton hierarchy at joint '{child}'.")
            stack.remove(name)

        dfs(self.root_name)

        # Check zero-length rest bones
        for b in self.bones.values():
            if b.rest_length <= 1e-6:
                raise SkeletonError(f"Bone '{b.name}' has zero or invalid rest length ({b.rest_length}).")

    def update_from_pose(self, pose: PoseFrame, enforce_bone_lengths: bool = True) -> None:
        """
        Applies observed world positions from a PoseFrame, regularizes bone lengths,
        and computes local quaternions.
        """
        # 1. Set raw world positions for matching joints
        for j_name, node in self.joints.items():
            if j_name in pose.joints:
                node.world_pos = pose.joints[j_name].to_numpy()

        # 2. Kinematic bone length enforcement (top-down traversal from root)
        if enforce_bone_lengths:
            self._enforce_lengths_recursive(self.root_name)

        # 3. Compute bone orientations and hierarchical local rotations
        self._compute_rotations()

    def _enforce_lengths_recursive(self, current_joint: str) -> None:
        node = self.joints[current_joint]
        for child_name in node.children_names:
            child_node = self.joints[child_name]

            # Find connecting bone
            bone = None
            for b in self.bones.values():
                if b.parent_joint == current_joint and b.child_joint == child_name:
                    bone = b
                    break

            if bone:
                cur_vec = child_node.world_pos - node.world_pos
                cur_norm = np.linalg.norm(cur_vec)
                if cur_norm > 1e-6:
                    unit_dir = cur_vec / cur_norm
                else:
                    # Fallback to rest direction
                    rest_vec = child_node.rest_world_pos - node.rest_world_pos
                    unit_dir = QuaternionMath.normalize_vector(rest_vec)

                # Clamp child position strictly to anatomical rest length
                child_node.world_pos = node.world_pos + unit_dir * bone.rest_length

            self._enforce_lengths_recursive(child_name)

    def _compute_rotations(self) -> None:
        """Calculates world and local rotations for all joints."""
        # Top-down calculation
        self._compute_joint_rotation_recursive(self.root_name)

    def _compute_joint_rotation_recursive(self, joint_name: str) -> None:
        node = self.joints[joint_name]

        if node.children_names:
            # Primary bone direction to first child
            primary_child = self.joints[node.children_names[0]]
            rest_dir = QuaternionMath.normalize_vector(primary_child.rest_world_pos - node.rest_world_pos)
            cur_dir = QuaternionMath.normalize_vector(primary_child.world_pos - node.world_pos)
            node.world_rotation = QuaternionMath.rotation_between_vectors(rest_dir, cur_dir)
        else:
            # End effector inherits parent world rotation
            if node.parent_name:
                node.world_rotation = self.joints[node.parent_name].world_rotation.copy()
            else:
                node.world_rotation = QuaternionMath.identity()

        # Compute local rotation: q_local = q_parent^-1 * q_world
        if node.parent_name:
            parent_node = self.joints[node.parent_name]
            q_parent_inv = QuaternionMath.conjugate(parent_node.world_rotation)
            node.local_rotation = QuaternionMath.multiply(q_parent_inv, node.world_rotation)
            node.local_pos = node.world_pos - parent_node.world_pos
        else:
            node.local_rotation = node.world_rotation.copy()
            node.local_pos = node.world_pos.copy()

        # Recurse
        for child_name in node.children_names:
            self._compute_joint_rotation_recursive(child_name)

    def reset_to_rest(self) -> None:
        """Resets all joints to their canonical rest positions, orientations, and local offsets."""
        for node in self.joints.values():
            node.world_pos = node.rest_world_pos.copy()
            if node.parent_name is not None:
                parent_rest = self.joints[node.parent_name].rest_world_pos
                node.local_pos = (node.rest_world_pos - parent_rest).copy()
            else:
                node.local_pos = node.rest_world_pos.copy()
            node.world_rotation = QuaternionMath.identity()
            node.local_rotation = QuaternionMath.identity()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "root_name": self.root_name,
            "joints": {
                name: {
                    "parent": node.parent_name,
                    "children": node.children_names,
                    "rest_world_pos": node.rest_world_pos.tolist(),
                    "world_pos": node.world_pos.tolist(),
                    "local_pos": node.local_pos.tolist(),
                    "world_rotation": node.world_rotation.tolist(),  # [w, x, y, z]
                    "local_rotation": node.local_rotation.tolist(),  # [w, x, y, z]
                }
                for name, node in self.joints.items()
            },
            "bones": [
                {
                    "name": b.name,
                    "parent": b.parent_joint,
                    "child": b.child_joint,
                    "rest_length": float(b.rest_length),
                }
                for b in self.bones.values()
            ],
        }
