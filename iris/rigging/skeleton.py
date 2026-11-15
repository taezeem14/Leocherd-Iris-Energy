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
        self.symmetry_plane: Optional[Tuple[np.ndarray, float]] = None
        self.symmetry_pairs: Dict[str, str] = {}
        self.symmetry_bone_pairs: Dict[str, str] = {}
        self.midline_joints: List[str] = []
        self.midline_bones: List[str] = []

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

    def validate_hierarchy(self) -> None:
        """Validates acyclic tree structure, root connectivity, and bone lengths."""
        self.validate()

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

                # Anatomical biomechanical constraints for legs:
                # 1. Shin (knee to foot) must extend downward (-Y)
                if "foot" in child_name.lower() and "leg" in current_joint.lower():
                    if unit_dir[1] > -0.10:
                        unit_dir[1] = -0.90
                        unit_dir = QuaternionMath.normalize_vector(unit_dir)

                # 2. Thigh (hips to knee) must extend downward (-Y)
                elif "leg" in child_name.lower() and "upleg" in current_joint.lower():
                    if unit_dir[1] > 0.10:
                        unit_dir[1] = -0.85
                        unit_dir = QuaternionMath.normalize_vector(unit_dir)

                # Clamp child position strictly to anatomical rest length
                child_node.world_pos = node.world_pos + unit_dir * bone.rest_length

            self._enforce_lengths_recursive(child_name)

    def _compute_rotations(self) -> None:
        """Calculates world and local rotations for all joints."""
        # Top-down calculation
        self._compute_joint_rotation_recursive(self.root_name)

    def _compute_generalized_root_orientation(self) -> Optional[np.ndarray]:
        """
        Computes a generalized orthonormal 3-axis basis (v_up, v_fwd, v_lat)
        for arbitrary creature roots (horizontal quadrupeds, linear serpentines,
        radial tentaculars, unlabelled discovered skeletons) without roll/twist ambiguity.
        """
        node = self.joints[self.root_name]
        if not node.children_names:
            return QuaternionMath.identity()

        # 1. Determine primary longitudinal child
        long_child = None
        trunk_keywords = ["spine", "lumbar", "thoracic", "chest", "neck", "vert", "trunk", "body", "head"]
        for c in node.children_names:
            c_low = c.lower()
            if any(kw in c_low for kw in trunk_keywords):
                long_child = c
                break

        if long_child is None:
            # Pick child with deepest subtree
            def _get_depth(j_name: str) -> int:
                j = self.joints[j_name]
                if not j.children_names:
                    return 1
                return 1 + max(_get_depth(ch) for ch in j.children_names)

            depths = {c: _get_depth(c) for c in node.children_names}
            max_depth = max(depths.values())
            deep_children = [c for c, d in depths.items() if d == max_depth]
            if len(deep_children) == 1:
                long_child = deep_children[0]
            else:
                # Tie-breaker: largest rest distance from root
                long_child = max(
                    deep_children,
                    key=lambda c: float(np.linalg.norm(self.joints[c].rest_world_pos - node.rest_world_pos)),
                )

        child_node = self.joints[long_child]
        v_long_rest = child_node.rest_world_pos - node.rest_world_pos
        v_long_cur = child_node.world_pos - node.world_pos
        if np.linalg.norm(v_long_rest) < 1e-6 or np.linalg.norm(v_long_cur) < 1e-6:
            return None

        # 2. Determine secondary lateral axis
        v_lat_rest = None
        v_lat_cur = None

        # 2a. Check symmetry pairs among children of root
        if hasattr(self, "symmetry_pairs") and self.symmetry_pairs:
            for c in node.children_names:
                partner = self.symmetry_pairs.get(c)
                if partner in node.children_names and partner != c:
                    if self.joints[partner].rest_world_pos[0] > self.joints[c].rest_world_pos[0]:
                        cL, cR = c, partner
                    else:
                        cL, cR = partner, c
                    v_lat_rest = self.joints[cR].rest_world_pos - self.joints[cL].rest_world_pos
                    v_lat_cur = self.joints[cR].world_pos - self.joints[cL].world_pos
                    break

        # 2b. Check name pattern among children of root (e.g. BackLeft_Hip and BackRight_Hip)
        if v_lat_rest is None:
            left_candidates = [c for c in node.children_names if "left" in c.lower() or c.startswith("FL") or c.startswith("BL")]
            right_candidates = [c for c in node.children_names if "right" in c.lower() or c.startswith("FR") or c.startswith("BR")]
            if left_candidates and right_candidates:
                cL, cR = left_candidates[0], right_candidates[0]
                v_lat_rest = self.joints[cR].rest_world_pos - self.joints[cL].rest_world_pos
                v_lat_cur = self.joints[cR].world_pos - self.joints[cL].world_pos

        # 2c. Check symmetry pairs anywhere in the skeleton
        if v_lat_rest is None and hasattr(self, "symmetry_pairs") and self.symmetry_pairs:
            valid_pairs = []
            for jL, jR in self.symmetry_pairs.items():
                if jL in self.joints and jR in self.joints and jL != jR:
                    sep = float(np.linalg.norm(self.joints[jR].rest_world_pos - self.joints[jL].rest_world_pos))
                    valid_pairs.append((sep, jL, jR))
            if valid_pairs:
                valid_pairs.sort(key=lambda item: item[0], reverse=True)
                _, jL, jR = valid_pairs[0]
                if self.joints[jR].rest_world_pos[0] < self.joints[jL].rest_world_pos[0]:
                    jL, jR = jR, jL
                v_lat_rest = self.joints[jR].rest_world_pos - self.joints[jL].rest_world_pos
                v_lat_cur = self.joints[jR].world_pos - self.joints[jL].world_pos

        # 2d. Check curvature along chain (e.g. Serpentine: root -> c1 -> c2)
        if v_lat_rest is None and child_node.children_names:
            c2_name = child_node.children_names[0]
            c2_node = self.joints[c2_name]
            u1_cur = child_node.world_pos - node.world_pos
            u2_cur = c2_node.world_pos - child_node.world_pos
            norm_cur = np.cross(u1_cur, u2_cur)
            if np.linalg.norm(norm_cur) > 1e-4:
                v_lat_cur = np.cross(norm_cur, v_long_cur)
            u1_rest = child_node.rest_world_pos - node.rest_world_pos
            u2_rest = c2_node.rest_world_pos - child_node.rest_world_pos
            norm_rest = np.cross(u1_rest, u2_rest)
            if np.linalg.norm(norm_rest) > 1e-4:
                v_lat_rest = np.cross(norm_rest, v_long_rest)
            else:
                v_lat_rest = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 2e. Check radial children (e.g. Tentacular hub with multiple branches)
        if v_lat_rest is None and len(node.children_names) >= 2:
            other_children = [c for c in node.children_names if c != long_child]
            other_node = self.joints[other_children[0]]
            v_other_rest = other_node.rest_world_pos - node.rest_world_pos
            v_other_cur = other_node.world_pos - node.world_pos
            u_long_rest = v_long_rest / np.linalg.norm(v_long_rest)
            u_long_cur = v_long_cur / np.linalg.norm(v_long_cur)
            v_lat_rest = v_other_rest - np.dot(v_other_rest, u_long_rest) * u_long_rest
            v_lat_cur = v_other_cur - np.dot(v_other_cur, u_long_cur) * u_long_cur

        # 2f. Ultimate fallback for lateral
        if v_lat_rest is None or np.linalg.norm(v_lat_rest) < 1e-6 or v_lat_cur is None or np.linalg.norm(v_lat_cur) < 1e-6:
            v_lat_rest = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            v_lat_cur = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 3. Determine Upright vs Horizontal Posture
        v_long_unit = v_long_rest / (np.linalg.norm(v_long_rest) + 1e-8)
        is_upright = abs(float(v_long_unit[1])) > 0.7071

        if is_upright:
            y_rest = v_long_rest
            z_rest = np.cross(v_lat_rest, y_rest)
            b_rest = QuaternionMath.build_orthonormal_basis(y_rest, z_rest)

            y_cur = v_long_cur
            z_cur = np.cross(v_lat_cur, y_cur)
            b_cur = QuaternionMath.build_orthonormal_basis(y_cur, z_cur)
        else:
            z_rest = v_long_rest
            y_rest = np.cross(z_rest, v_lat_rest)
            b_rest = QuaternionMath.build_orthonormal_basis(y_rest, z_rest)

            z_cur = v_long_cur
            y_cur = np.cross(z_cur, v_lat_cur)
            b_cur = QuaternionMath.build_orthonormal_basis(y_cur, z_cur)

        return QuaternionMath.rotation_between_bases(b_rest, b_cur)

    def _compute_joint_rotation_recursive(self, joint_name: str) -> None:
        node = self.joints[joint_name]
        q_world = None
        name_lower = joint_name.lower()

        # Specialized anatomical basis calculation for major segments to prevent roll/twist distortion
        if name_lower == "hips" and "Spine" in self.joints and "LeftUpLeg" in self.joints and "RightUpLeg" in self.joints:
            p_spine = self.joints["Spine"]
            p_l = self.joints["LeftUpLeg"]
            p_r = self.joints["RightUpLeg"]
            y_rest = p_spine.rest_world_pos - node.rest_world_pos
            lat_rest = p_r.rest_world_pos - p_l.rest_world_pos  # Lateral pointing strictly Right (+X)
            z_rest = np.cross(lat_rest, y_rest)                 # Right x Up = Forward (+Z)
            b_rest = QuaternionMath.build_orthonormal_basis(y_rest, z_rest)

            y_cur = p_spine.world_pos - node.world_pos
            lat_cur = p_r.world_pos - p_l.world_pos
            z_cur = np.cross(lat_cur, y_cur)
            b_cur = QuaternionMath.build_orthonormal_basis(y_cur, z_cur)
            q_world = QuaternionMath.rotation_between_bases(b_rest, b_cur)

        elif joint_name == self.root_name:
            q_world = self._compute_generalized_root_orientation()

        elif name_lower == "chest" and "Neck" in self.joints and "LeftShoulder" in self.joints and "RightShoulder" in self.joints:
            p_neck = self.joints["Neck"]
            p_l = self.joints["LeftShoulder"]
            p_r = self.joints["RightShoulder"]
            y_rest = p_neck.rest_world_pos - node.rest_world_pos
            lat_rest = p_r.rest_world_pos - p_l.rest_world_pos  # Right (+X)
            z_rest = np.cross(lat_rest, y_rest)                 # Forward (+Z)
            b_rest = QuaternionMath.build_orthonormal_basis(y_rest, z_rest)

            y_cur = p_neck.world_pos - node.world_pos
            lat_cur = p_r.world_pos - p_l.world_pos
            z_cur = np.cross(lat_cur, y_cur)
            b_cur = QuaternionMath.build_orthonormal_basis(y_cur, z_cur)
            q_world = QuaternionMath.rotation_between_bases(b_rest, b_cur)

        elif name_lower == "neck" and "Head" in self.joints:
            p_head = self.joints["Head"]
            y_rest = p_head.rest_world_pos - node.rest_world_pos
            if "LeftShoulder" in self.joints and "RightShoulder" in self.joints:
                lat_rest = self.joints["RightShoulder"].rest_world_pos - self.joints["LeftShoulder"].rest_world_pos
                lat_cur = self.joints["RightShoulder"].world_pos - self.joints["LeftShoulder"].world_pos
            else:
                lat_rest = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                lat_cur = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            z_rest = np.cross(lat_rest, y_rest)
            b_rest = QuaternionMath.build_orthonormal_basis(y_rest, z_rest)

            y_cur = p_head.world_pos - node.world_pos
            z_cur = np.cross(lat_cur, y_cur)
            b_cur = QuaternionMath.build_orthonormal_basis(y_cur, z_cur)
            q_world = QuaternionMath.rotation_between_bases(b_rest, b_cur)

        elif name_lower == "head" and "Neck" in self.joints:
            p_neck = self.joints["Neck"]
            y_rest = node.rest_world_pos - p_neck.rest_world_pos
            if "LeftShoulder" in self.joints and "RightShoulder" in self.joints:
                lat_rest = self.joints["RightShoulder"].rest_world_pos - self.joints["LeftShoulder"].rest_world_pos
                lat_cur = self.joints["RightShoulder"].world_pos - self.joints["LeftShoulder"].world_pos
            else:
                lat_rest = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                lat_cur = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            z_rest = np.cross(lat_rest, y_rest)
            b_rest = QuaternionMath.build_orthonormal_basis(y_rest, z_rest)

            y_cur = node.world_pos - p_neck.world_pos
            z_cur = np.cross(lat_cur, y_cur)
            b_cur = QuaternionMath.build_orthonormal_basis(y_cur, z_cur)
            q_world = QuaternionMath.rotation_between_bases(b_rest, b_cur)

        if q_world is not None:
            node.world_rotation = q_world
        elif node.children_names:
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

            # Anatomical biomechanical limits to prevent distortion/unnatural twisting:
            if name_lower == "neck":
                node.local_rotation = QuaternionMath.clamp_rotation_angle(node.local_rotation, 0.65)  # ~37 deg max cervical excursion
                node.world_rotation = QuaternionMath.multiply(parent_node.world_rotation, node.local_rotation)
            elif name_lower == "head":
                node.local_rotation = QuaternionMath.clamp_rotation_angle(node.local_rotation, 0.55)  # ~31 deg max head-on-neck
                node.world_rotation = QuaternionMath.multiply(parent_node.world_rotation, node.local_rotation)
            elif "spine" in name_lower:
                node.local_rotation = QuaternionMath.clamp_rotation_angle(node.local_rotation, 0.52)  # ~30 deg spine flexion
                node.world_rotation = QuaternionMath.multiply(parent_node.world_rotation, node.local_rotation)
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
