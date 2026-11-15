"""
Production Inverse Kinematics (IK) Engine for Leocherd Iris Energy.

Includes:
- TwoBoneIKSolver: Analytical, closed-form 2-bone IK solver for limbs using Law of Cosines.
- FABRIKSolver: Forward And Backward Reaching Inverse Kinematics solver for N-segment chains
  with per-joint angular constraint cones.
- IKChain: Skeleton-integrated kinematic chain management.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from iris.core.exceptions import SkeletonError
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton


@dataclass
class TwoBoneIKResult:
    """Resolved 3D positions and local rotation quaternions for a 2-bone kinematic limb."""

    positions: Tuple[np.ndarray, np.ndarray, np.ndarray]
    rotations: Tuple[np.ndarray, np.ndarray, np.ndarray]

    @property
    def p_root(self) -> np.ndarray:
        return self.positions[0]

    @property
    def p_mid(self) -> np.ndarray:
        return self.positions[1]

    @property
    def p_end(self) -> np.ndarray:
        return self.positions[2]

    @property
    def q_root(self) -> np.ndarray:
        return self.rotations[0]

    @property
    def q_mid(self) -> np.ndarray:
        return self.rotations[1]

    @property
    def q_end(self) -> np.ndarray:
        return self.rotations[2]

    def __iter__(self):
        yield self.positions
        yield self.rotations

    def __getitem__(self, index: int) -> Any:
        return (self.positions, self.rotations)[index]

    def __len__(self) -> int:
        return 2


class TwoBoneIKSolver:
    """
    Analytical, closed-form 2-bone Inverse Kinematics solver for limbs
    (e.g. UpperArm -> ForeArm -> Hand, Thigh -> Shin -> Foot, Quadruped Shoulder -> Knee -> Paw).

    Solves the triangle law of cosines given p_root, p_target, p_pole, and bone lengths l1, l2.
    Handles reach clamping gracefully when distance > l1 + l2 or distance < |l1 - l2|.
    """

    def __init__(self, l1: Optional[float] = None, l2: Optional[float] = None) -> None:
        self.l1 = float(l1) if l1 is not None else None
        self.l2 = float(l2) if l2 is not None else None

    def solve(
        self,
        p_root: np.ndarray,
        p_target: np.ndarray,
        p_pole: np.ndarray,
        l1: Optional[float] = None,
        l2: Optional[float] = None,
        rest_positions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        parent_rotation: Optional[np.ndarray] = None,
        pole_is_vector: bool = False,
    ) -> TwoBoneIKResult:
        """
        Solves analytical 2-bone IK.

        Args:
            p_root: Root joint 3D position [x, y, z].
            p_target: Desired target position for the end-effector [x, y, z].
            p_pole: Pole vector or pole target position directing the bend plane.
            l1: Length of bone 1 (root to mid). Defaults to self.l1.
            l2: Length of bone 2 (mid to end). Defaults to self.l2.
            rest_positions: Optional canonical rest positions (r_root, r_mid, r_end).
            parent_rotation: Optional world rotation quaternion of root's parent joint.
            pole_is_vector: If True, p_pole is treated directly as a direction vector.

        Returns:
            TwoBoneIKResult containing:
                positions: (p_root, p_mid, p_end)
                rotations: (q_root_local, q_mid_local, q_end_local)
        """
        p_root_arr = np.asarray(p_root, dtype=np.float32)
        p_target_arr = np.asarray(p_target, dtype=np.float32)
        p_pole_arr = np.asarray(p_pole, dtype=np.float32)

        # Resolve bone lengths
        len1 = float(l1 if l1 is not None else (self.l1 if self.l1 is not None else 0.0))
        len2 = float(l2 if l2 is not None else (self.l2 if self.l2 is not None else 0.0))
        if (len1 <= 1e-6 or len2 <= 1e-6) and rest_positions is not None:
            r0, r1, r2 = rest_positions
            len1 = float(np.linalg.norm(r1 - r0))
            len2 = float(np.linalg.norm(r2 - r1))

        if len1 <= 1e-6 or len2 <= 1e-6:
            raise ValueError(f"Invalid bone lengths: l1={len1}, l2={len2}. Both must be positive.")

        # Vector from root to target
        d_vec = p_target_arr - p_root_arr
        dist = float(np.linalg.norm(d_vec))

        # Handle zero distance target
        if dist < 1e-8:
            d_hat = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            dist = 1e-8
        else:
            d_hat = d_vec / dist

        # Reach clamping limits
        max_reach = len1 + len2
        min_reach = abs(len1 - len2)

        # Compute bend direction from pole vector / pole position
        n_bend = self._compute_bend_normal(p_root_arr, d_hat, p_pole_arr, pole_is_vector)

        if dist >= max_reach:
            # Over-extension clamping: limb is fully extended straight toward target
            p_mid = p_root_arr + len1 * d_hat
            p_end = p_root_arr + max_reach * d_hat
        elif dist <= min_reach:
            # Under-extension clamping: limb is folded completely
            p_mid = p_root_arr + len1 * d_hat
            p_end = p_root_arr + min_reach * d_hat
        else:
            # Triangle Law of Cosines
            # c = dist, a = len1, b = len2
            # cos(alpha) at root: (a^2 + c^2 - b^2) / (2 * a * c)
            cos_alpha = (len1 * len1 + dist * dist - len2 * len2) / (2.0 * len1 * dist)
            cos_alpha = max(-1.0, min(1.0, cos_alpha))
            sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha))

            p_mid = p_root_arr + len1 * (cos_alpha * d_hat + sin_alpha * n_bend)
            p_end = p_target_arr.copy()

        rotations = self._compute_rotations(
            p_root=p_root_arr,
            p_mid=p_mid,
            p_end=p_end,
            d_hat=d_hat,
            n_bend=n_bend,
            rest_positions=rest_positions,
            parent_rotation=parent_rotation,
        )

        return TwoBoneIKResult(
            positions=(p_root_arr.copy(), p_mid.astype(np.float32), p_end.astype(np.float32)),
            rotations=rotations,
        )

    def _compute_bend_normal(
        self,
        p_root: np.ndarray,
        d_hat: np.ndarray,
        p_pole: np.ndarray,
        pole_is_vector: bool = False,
    ) -> np.ndarray:
        if pole_is_vector:
            v_perp = p_pole - float(np.dot(p_pole, d_hat)) * d_hat
            norm = float(np.linalg.norm(v_perp))
            if norm > 1e-6:
                return (v_perp / norm).astype(np.float32)

        # Treat p_pole as a world position: vector from root to pole
        v_from_root = p_pole - p_root
        v_perp = v_from_root - float(np.dot(v_from_root, d_hat)) * d_hat
        norm = float(np.linalg.norm(v_perp))

        # If pole position was collinear with root->target line, check if p_pole itself was a direction vector
        if norm < 1e-5:
            v_perp_dir = p_pole - float(np.dot(p_pole, d_hat)) * d_hat
            norm_dir = float(np.linalg.norm(v_perp_dir))
            if norm_dir > 1e-5:
                return (v_perp_dir / norm_dir).astype(np.float32)

        if norm > 1e-5:
            return (v_perp / norm).astype(np.float32)

        # Fallback to an orthogonal unit vector
        fallback = (
            np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if abs(d_hat[1]) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        fallback_perp = fallback - float(np.dot(fallback, d_hat)) * d_hat
        return QuaternionMath.normalize_vector(fallback_perp)

    def _compute_rotations(
        self,
        p_root: np.ndarray,
        p_mid: np.ndarray,
        p_end: np.ndarray,
        d_hat: np.ndarray,
        n_bend: np.ndarray,
        rest_positions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        parent_rotation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u1_cur = QuaternionMath.normalize_vector(p_mid - p_root)
        u2_cur = QuaternionMath.normalize_vector(p_end - p_mid)

        if rest_positions is not None:
            r0, r1, r2 = rest_positions
            u1_rest = QuaternionMath.normalize_vector(r1 - r0)
            u2_rest = QuaternionMath.normalize_vector(r2 - r1)
        else:
            u1_rest = d_hat.copy()
            u2_rest = d_hat.copy()

        q1_world = QuaternionMath.rotation_between_vectors(u1_rest, u1_cur)
        q2_world = QuaternionMath.rotation_between_vectors(u2_rest, u2_cur)

        if parent_rotation is not None:
            q_parent_inv = QuaternionMath.conjugate(parent_rotation)
            q_root_local = QuaternionMath.multiply(q_parent_inv, q1_world)
        else:
            q_root_local = q1_world.copy()

        q1_inv = QuaternionMath.conjugate(q1_world)
        q_mid_local = QuaternionMath.multiply(q1_inv, q2_world)
        q_end_local = QuaternionMath.identity()

        return (q_root_local, q_mid_local, q_end_local)


class FABRIKResult(np.ndarray):
    """
    Subclass of np.ndarray containing solved joint positions (shape: (N+1, 3)).
    Provides solver convergence metadata: iterations, converged, and error.
    """

    iterations: int
    converged: bool
    error: float

    def __new__(
        cls,
        positions: Union[List[np.ndarray], np.ndarray],
        iterations: int = 0,
        converged: bool = True,
        error: float = 0.0,
    ) -> FABRIKResult:
        obj = np.asarray(positions, dtype=np.float32).view(cls)
        obj.iterations = int(iterations)
        obj.converged = bool(converged)
        obj.error = float(error)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.iterations = getattr(obj, "iterations", 0)
        self.converged = getattr(obj, "converged", False)
        self.error = getattr(obj, "error", 0.0)

    @property
    def positions(self) -> np.ndarray:
        return np.asarray(self)


class FABRIKSolver:
    """
    Forward And Backward Reaching Inverse Kinematics (FABRIK) solver.

    Unsupervised multi-joint chain IK solver for arbitrary N-segment kinematic chains
    (spines, necks, tails, tentacles, multi-segmented limbs).
    Solves via alternating forward and backward reaching passes with exact bone length
    preservation and optional per-joint angular constraint cones.
    """

    def __init__(
        self,
        tol: float = 1e-3,
        max_iterations: int = 15,
        angular_limits: Optional[Union[float, List[Optional[float]], Dict[int, float]]] = None,
    ) -> None:
        self.tol = tol
        self.max_iterations = max_iterations
        self.angular_limits = angular_limits

    def solve(
        self,
        joint_positions: Union[List[np.ndarray], np.ndarray],
        target_pos: np.ndarray,
        bone_lengths: Optional[Union[List[float], np.ndarray]] = None,
        angular_limits: Optional[Union[float, List[Optional[float]], Dict[int, float]]] = None,
        tol: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> FABRIKResult:
        """
        Solves FABRIK on an articulated multi-joint chain.

        Args:
            joint_positions: Initial joint positions array of shape (N+1, 3).
            target_pos: Target 3D coordinates for the end-effector [x, y, z].
            bone_lengths: Optional explicit segment lengths. If None, derived from joint_positions.
            angular_limits: Optional maximum bend angle limits (radians) relative to parent bone.
            tol: Convergence tolerance in meters (defaults to self.tol).
            max_iterations: Maximum iteration count (defaults to self.max_iterations).

        Returns:
            FABRIKResult containing solved positions, iterations count, convergence flag, and final error.
        """
        pts = np.asarray(joint_positions, dtype=np.float32).copy()
        target = np.asarray(target_pos, dtype=np.float32)
        n_joints = len(pts)
        if n_joints < 2:
            raise ValueError(f"FABRIK chain requires at least 2 joints, got {n_joints}.")

        n_segments = n_joints - 1

        # Derive or validate bone lengths
        if bone_lengths is not None:
            lengths = np.asarray(bone_lengths, dtype=np.float32)
            if len(lengths) != n_segments:
                raise ValueError(f"Expected {n_segments} bone lengths, got {len(lengths)}.")
        else:
            diffs = pts[1:] - pts[:-1]
            lengths = np.linalg.norm(diffs, axis=1).astype(np.float32)

        total_length = float(np.sum(lengths))
        p0_orig = pts[0].copy()

        # Check reachability
        dist_to_target = float(np.linalg.norm(target - p0_orig))
        if dist_to_target > total_length:
            # Beyond full extension: stretch straight toward target
            if dist_to_target < 1e-8:
                u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                u = (target - p0_orig) / dist_to_target

            for i in range(n_segments):
                pts[i + 1] = pts[i] + lengths[i] * u

            return FABRIKResult(
                pts,
                iterations=1,
                converged=False,
                error=dist_to_target - total_length,
            )

        curr_tol = float(tol if tol is not None else self.tol)
        curr_max_iter = int(max_iterations if max_iterations is not None else self.max_iterations)
        active_limits = angular_limits if angular_limits is not None else self.angular_limits

        initial_err = float(np.linalg.norm(pts[-1] - target))
        if initial_err <= curr_tol:
            return FABRIKResult(pts, iterations=0, converged=True, error=initial_err)

        last_err = initial_err
        for iteration in range(1, curr_max_iter + 1):
            # --- Forward Pass (end-effector to root) ---
            pts[-1] = target
            for i in range(n_segments - 1, -1, -1):
                r_vec = pts[i] - pts[i + 1]
                r_len = float(np.linalg.norm(r_vec))
                if r_len > 1e-8:
                    pts[i] = pts[i + 1] + (lengths[i] / r_len) * r_vec
                else:
                    pts[i] = pts[i + 1] + np.array([0.0, lengths[i], 0.0], dtype=np.float32)

            # --- Backward Pass (root to end-effector) ---
            pts[0] = p0_orig
            for i in range(n_segments):
                r_vec = pts[i + 1] - pts[i]
                r_len = float(np.linalg.norm(r_vec))
                if r_len > 1e-8:
                    u = r_vec / r_len
                else:
                    u = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                # Angular cone constraint relative to parent bone segment
                if i >= 1 and active_limits is not None:
                    lim = self._get_joint_limit(i, active_limits)
                    if lim is not None and lim > 0.0:
                        parent_vec = pts[i] - pts[i - 1]
                        parent_len = float(np.linalg.norm(parent_vec))
                        if parent_len > 1e-8:
                            u_parent = parent_vec / parent_len
                            cos_theta = float(np.clip(np.dot(u_parent, u), -1.0, 1.0))
                            cos_lim = math.cos(lim)
                            if cos_theta < cos_lim:
                                # Direction exceeds angular cone: clamp onto cone boundary
                                v_perp = u - cos_theta * u_parent
                                v_perp_len = float(np.linalg.norm(v_perp))
                                if v_perp_len > 1e-6:
                                    n_perp = v_perp / v_perp_len
                                else:
                                    fallback = (
                                        np.array([0.0, 1.0, 0.0], dtype=np.float32)
                                        if abs(u_parent[1]) < 0.9
                                        else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                                    )
                                    alt_perp = fallback - float(np.dot(fallback, u_parent)) * u_parent
                                    n_perp = alt_perp / float(np.linalg.norm(alt_perp))

                                u = cos_lim * u_parent + math.sin(lim) * n_perp

                pts[i + 1] = pts[i] + lengths[i] * u

            last_err = float(np.linalg.norm(pts[-1] - target))
            if last_err <= curr_tol:
                return FABRIKResult(pts, iterations=iteration, converged=True, error=last_err)

        return FABRIKResult(pts, iterations=curr_max_iter, converged=(last_err <= curr_tol), error=last_err)

    @staticmethod
    def _get_joint_limit(
        joint_idx: int,
        limits: Union[float, List[Optional[float]], Dict[int, float]],
    ) -> Optional[float]:
        if isinstance(limits, (int, float)):
            return float(limits)
        elif isinstance(limits, (list, tuple)):
            if joint_idx < len(limits):
                val = limits[joint_idx]
                return float(val) if val is not None else None
            return None
        elif isinstance(limits, dict):
            val = limits.get(joint_idx)
            return float(val) if val is not None else None
        return None


class IKChain:
    """
    Represents an articulated kinematic chain within a Skeleton.
    Binds a joint sequence to an analytical TwoBone or multi-segment FABRIK solver.
    """

    def __init__(
        self,
        skeleton: Skeleton,
        joint_names: Optional[List[str]] = None,
        root_joint: Optional[str] = None,
        end_effector: Optional[str] = None,
        solver_type: str = "auto",
        angular_limits: Optional[Union[float, List[Optional[float]], Dict[int, float]]] = None,
        tol: float = 1e-3,
        max_iterations: int = 15,
    ) -> None:
        self.skeleton = skeleton

        # Resolve chain joint sequence
        if joint_names is not None and len(joint_names) > 0:
            self.joint_names = list(joint_names)
            for j in self.joint_names:
                if j not in self.skeleton.joints:
                    raise SkeletonError(f"Joint '{j}' does not exist in skeleton joints.")
            self.root_joint = root_joint or self.joint_names[0]
            self.end_effector = end_effector or self.joint_names[-1]
        else:
            if root_joint is None or end_effector is None:
                raise SkeletonError("Either 'joint_names' or both 'root_joint' and 'end_effector' must be provided.")
            self.root_joint = root_joint
            self.end_effector = end_effector
            path = self._find_chain_path(self.root_joint, self.end_effector)
            if path is None:
                raise SkeletonError(f"No kinematic path from '{root_joint}' to '{end_effector}' in skeleton.")
            self.joint_names = path

        # Determine solver type
        normalized_solver = solver_type.lower()
        if normalized_solver == "auto":
            self.solver_type = "two_bone" if len(self.joint_names) == 3 else "fabrik"
        elif normalized_solver in ("two_bone", "two-bone", "twobone"):
            self.solver_type = "two_bone"
        elif normalized_solver == "fabrik":
            self.solver_type = "fabrik"
        else:
            raise SkeletonError(f"Unsupported IK solver type '{solver_type}'. Must be 'two_bone' or 'fabrik'.")

        # Cache bone rest lengths
        self.bone_lengths: List[float] = []
        for i in range(len(self.joint_names) - 1):
            p_name = self.joint_names[i]
            c_name = self.joint_names[i + 1]
            b_len = None
            for bone in self.skeleton.bones.values():
                if (bone.parent_joint == p_name and bone.child_joint == c_name) or (
                    bone.parent_joint == c_name and bone.child_joint == p_name
                ):
                    b_len = float(bone.rest_length)
                    break
            if b_len is None:
                p_rest = self.skeleton.joints[p_name].rest_world_pos
                c_rest = self.skeleton.joints[c_name].rest_world_pos
                b_len = float(np.linalg.norm(c_rest - p_rest))
            self.bone_lengths.append(b_len)

        # Instantiate solvers
        self.two_bone_solver = TwoBoneIKSolver(
            l1=self.bone_lengths[0] if len(self.bone_lengths) >= 2 else None,
            l2=self.bone_lengths[1] if len(self.bone_lengths) >= 2 else None,
        )
        self.fabrik_solver = FABRIKSolver(
            tol=tol,
            max_iterations=max_iterations,
            angular_limits=angular_limits,
        )

    def _find_chain_path(self, current: str, target: str) -> Optional[List[str]]:
        if current == target:
            return [current]
        node = self.skeleton.joints.get(current)
        if not node:
            return None
        for child in node.children_names:
            sub = self._find_chain_path(child, target)
            if sub is not None:
                return [current] + sub
        return None

    def solve(
        self,
        target_pos: np.ndarray,
        pole_pos: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Solves IK for the kinematic chain and updates skeleton joint transforms.

        Args:
            target_pos: Desired end-effector position [x, y, z].
            pole_pos: Optional pole vector or target position directing limb bend plane.

        Returns:
            Dict mapping joint names in chain to their updated world positions.
        """
        target = np.asarray(target_pos, dtype=np.float32)

        if self.solver_type == "two_bone":
            if len(self.joint_names) != 3:
                raise SkeletonError(f"TwoBoneIKSolver requires exactly 3 joints, got {len(self.joint_names)}.")

            root_node = self.skeleton.joints[self.joint_names[0]]
            mid_node = self.skeleton.joints[self.joint_names[1]]
            end_node = self.skeleton.joints[self.joint_names[2]]

            # If pole_pos is not provided, deduce from rest geometry or current geometry
            if pole_pos is None:
                rest_vec = mid_node.rest_world_pos - root_node.rest_world_pos
                d_rest = end_node.rest_world_pos - root_node.rest_world_pos
                d_norm = float(np.linalg.norm(d_rest))
                if d_norm > 1e-5:
                    d_unit = d_rest / d_norm
                    perp = rest_vec - float(np.dot(rest_vec, d_unit)) * d_unit
                    if np.linalg.norm(perp) > 1e-4:
                        pole = root_node.world_pos + perp
                    else:
                        pole = root_node.world_pos + np.array([0.0, 0.0, 1.0], dtype=np.float32)
                else:
                    pole = root_node.world_pos + np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                pole = np.asarray(pole_pos, dtype=np.float32)

            rest_positions = (
                root_node.rest_world_pos,
                mid_node.rest_world_pos,
                end_node.rest_world_pos,
            )

            parent_rot = None
            if root_node.parent_name:
                parent_rot = self.skeleton.joints[root_node.parent_name].world_rotation

            res = self.two_bone_solver.solve(
                p_root=root_node.world_pos,
                p_target=target,
                p_pole=pole,
                l1=self.bone_lengths[0],
                l2=self.bone_lengths[1],
                rest_positions=rest_positions,
                parent_rotation=parent_rot,
            )

            for i, name in enumerate(self.joint_names):
                node = self.skeleton.joints[name]
                node.world_pos = res.positions[i].copy()
                node.world_rotation = res.rotations[i].copy()
                if node.parent_name:
                    p_node = self.skeleton.joints[node.parent_name]
                    node.local_pos = node.world_pos - p_node.world_pos
                    node.local_rotation = QuaternionMath.multiply(
                        QuaternionMath.conjugate(p_node.world_rotation),
                        node.world_rotation,
                    )
                else:
                    node.local_pos = node.world_pos.copy()
                    node.local_rotation = node.world_rotation.copy()

        elif self.solver_type == "fabrik":
            current_positions = np.array(
                [self.skeleton.joints[j].world_pos for j in self.joint_names],
                dtype=np.float32,
            )
            res_fabrik = self.fabrik_solver.solve(
                joint_positions=current_positions,
                target_pos=target,
                bone_lengths=self.bone_lengths,
            )

            for i, name in enumerate(self.joint_names):
                node = self.skeleton.joints[name]
                node.world_pos = res_fabrik.positions[i].copy()

            # Update orientations along the chain
            for i in range(len(self.joint_names) - 1):
                node = self.skeleton.joints[self.joint_names[i]]
                child_node = self.skeleton.joints[self.joint_names[i + 1]]
                rest_dir = QuaternionMath.normalize_vector(child_node.rest_world_pos - node.rest_world_pos)
                cur_dir = QuaternionMath.normalize_vector(child_node.world_pos - node.world_pos)
                node.world_rotation = QuaternionMath.rotation_between_vectors(rest_dir, cur_dir)
                if node.parent_name:
                    p_node = self.skeleton.joints[node.parent_name]
                    node.local_pos = node.world_pos - p_node.world_pos
                    node.local_rotation = QuaternionMath.multiply(
                        QuaternionMath.conjugate(p_node.world_rotation),
                        node.world_rotation,
                    )
                else:
                    node.local_pos = node.world_pos.copy()
                    node.local_rotation = node.world_rotation.copy()

            # End effector inherits parent world rotation
            end_node = self.skeleton.joints[self.joint_names[-1]]
            if end_node.parent_name:
                p_node = self.skeleton.joints[end_node.parent_name]
                end_node.world_rotation = p_node.world_rotation.copy()
                end_node.local_pos = end_node.world_pos - p_node.world_pos
                end_node.local_rotation = QuaternionMath.identity()
            else:
                end_node.local_pos = end_node.world_pos.copy()
                end_node.local_rotation = end_node.world_rotation.copy()

        return {name: self.skeleton.joints[name].world_pos.copy() for name in self.joint_names}
