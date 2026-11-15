"""
Unsupervised Automatic Skeleton & Kinematic Hierarchy Discovery Engine.

Discovers articulated 3D kinematic hierarchies and skeleton topologies from arbitrary
unlabelled 3D point trajectory sequences using pairwise distance variance analysis,
kinematic centroid acceleration minimization, and minimum spanning trees.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from iris.core.exceptions import SkeletonError
from iris.core.logging import get_logger
from iris.pose.base import PoseFrame
from iris.rigging.joints import Bone, BoneEdge, JointNode
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton

logger = get_logger("iris.rigging.skeleton_discovery")


class AutomaticSkeletonDiscovery:
    """
    Unsupervised kinematic tree and skeleton structure discovery from motion trajectories.

    Pipeline:
      1. Pairwise rigid-distance variance analysis:
         Computes distance matrix D_ij(t) = ||P_i(t) - P_j(t)||, mean distance mu_ij,
         and temporal distance variance sigma^2_ij.
      2. Root Joint Identification:
         Finds the point closest to the kinematic center of mass with minimal variance
         in acceleration relative to global centroid.
      3. Kinematic Tree Inference via Minimum Spanning Tree (MST):
         Builds an undirected weighted graph where edge weight W_ij = mu_ij + lambda * sigma^2_ij
         (penalizing non-rigid pairs) and computes the MST.
      4. Directed Kinematic Hierarchy:
         Performs Breadth-First Search (BFS) starting from the identified Root Joint to establish
         parent-child relationships and bone directions.
      5. Output Articulated Skeleton:
         Constructs and validates a fully functional Skeleton instance populated with JointNode
         and Bone objects, rest lengths, and canonical rest orientations.
    """

    def __init__(
        self,
        lambda_param: float = 100.0,
        dist_weight: float = 0.5,
        acc_weight: float = 0.5,
        rest_frame_idx: int = 0,
    ) -> None:
        """
        Args:
            lambda_param: Rigidity penalty coefficient weighting pairwise distance variance.
            dist_weight: Weight for distance-to-centroid in root joint identification.
            acc_weight: Weight for acceleration-variance in root joint identification.
            rest_frame_idx: Frame index from trajectory used as reference for canonical rest pose.
        """
        self.lambda_param = float(lambda_param)
        self.dist_weight = float(dist_weight)
        self.acc_weight = float(acc_weight)
        self.rest_frame_idx = int(rest_frame_idx)

        # Cached state from discovery run
        self.point_names: List[str] = []
        self.mean_distance_matrix: Optional[np.ndarray] = None
        self.variance_distance_matrix: Optional[np.ndarray] = None
        self.edge_weights: Optional[np.ndarray] = None
        self.root_index: int = 0
        self.root_name: str = ""
        self.mst_edges: List[Tuple[int, int, float]] = []
        self.skeleton: Optional[Skeleton] = None

    def _parse_trajectory(
        self,
        trajectory: Union[np.ndarray, Dict[str, np.ndarray], List[PoseFrame]],
        point_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Normalizes input trajectory representations into a standard (T, N, 3) float32 array."""
        if isinstance(trajectory, list) and len(trajectory) > 0 and isinstance(trajectory[0], PoseFrame):
            names = point_names or trajectory[0].joint_names
            T = len(trajectory)
            N = len(names)
            coords = np.zeros((T, N, 3), dtype=np.float32)
            for t, frame in enumerate(trajectory):
                for i, name in enumerate(names):
                    joint = frame.get_joint(name)
                    if joint is not None:
                        coords[t, i] = joint.to_numpy()
                    elif t > 0:
                        coords[t, i] = coords[t - 1, i]
            return coords, list(names)

        elif isinstance(trajectory, dict):
            names = point_names or list(trajectory.keys())
            if not names:
                raise SkeletonError("Empty trajectory dictionary provided.")
            arrays = []
            for name in names:
                if name not in trajectory:
                    raise SkeletonError(f"Point '{name}' missing from trajectory dict.")
                arr = np.asarray(trajectory[name], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]  # shape: (1, 3)
                elif arr.ndim != 2 or arr.shape[-1] != 3:
                    raise SkeletonError(f"Trajectory for '{name}' must have shape (T, 3), got {arr.shape}.")
                arrays.append(arr)

            min_T = min(a.shape[0] for a in arrays)
            coords = np.stack([a[:min_T] for a in arrays], axis=1)  # shape: (T, N, 3)
            return coords, list(names)

        elif isinstance(trajectory, np.ndarray):
            arr = np.asarray(trajectory, dtype=np.float32)
            if arr.ndim == 2:
                # (N, 3) -> single frame
                arr = arr[np.newaxis, ...]
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise SkeletonError(f"Expected trajectory array of shape (T, N, 3), got {arr.shape}.")

            T, N, _ = arr.shape
            names = point_names if point_names is not None else [f"Joint_{i}" for i in range(N)]
            if len(names) != N:
                raise SkeletonError(f"Point names count ({len(names)}) does not match trajectory joints ({N}).")
            return arr, list(names)

        else:
            raise SkeletonError(f"Unsupported trajectory type: {type(trajectory)}")

    def compute_pairwise_distance_variance(
        self, trajectory: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step a: Pairwise Rigid-Distance Variance Analysis.
        Computes pairwise Euclidean distance matrix across all frames:
            D_ij(t) = ||P_i(t) - P_j(t)||
        Mean distance:
            mu_ij = Mean_t(D_ij(t))
        Distance variance:
            sigma^2_ij = Var_t(D_ij(t))

        Args:
            trajectory: 3D point trajectory of shape (T, N, 3).

        Returns:
            Tuple of (mean_distance_matrix, variance_distance_matrix) of shape (N, N).
        """
        T, N, _ = trajectory.shape
        if T == 0 or N == 0:
            raise SkeletonError("Trajectory has zero frames or zero joints.")

        # Compute pairwise distance tensor: shape (T, N, N)
        diff = trajectory[:, :, np.newaxis, :] - trajectory[:, np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        mean_dist = np.mean(dist, axis=0).astype(np.float32)
        if T > 1:
            var_dist = np.var(dist, axis=0).astype(np.float32)
        else:
            var_dist = np.zeros((N, N), dtype=np.float32)

        # Enforce exact symmetry and zero diagonal
        mean_dist = 0.5 * (mean_dist + mean_dist.T)
        var_dist = 0.5 * (var_dist + var_dist.T)
        np.fill_diagonal(mean_dist, 0.0)
        np.fill_diagonal(var_dist, 0.0)

        return mean_dist, var_dist

    def identify_root_joint(
        self,
        trajectory: np.ndarray,
        point_names: List[str],
        root_name: Optional[str] = None,
    ) -> Tuple[int, str]:
        """
        Step b: Root Joint Identification.
        Finds the point closest to the kinematic center of mass with minimal variance
        in acceleration relative to global centroid.

        Args:
            trajectory: 3D point trajectory of shape (T, N, 3).
            point_names: List of joint names.
            root_name: Optional explicit root joint name override.

        Returns:
            Tuple of (root_index, root_name).
        """
        T, N, _ = trajectory.shape
        if root_name is not None:
            if root_name in point_names:
                idx = point_names.index(root_name)
                return idx, root_name
            else:
                logger.warning(f"Specified root_name '{root_name}' not in point names. Reverting to automatic.")

        if N <= 1:
            return 0, point_names[0]

        # 1. Global kinematic centroid: shape (T, 1, 3)
        centroid = np.mean(trajectory, axis=1, keepdims=True)

        # Relative coordinates to centroid: shape (T, N, 3)
        rel_pos = trajectory - centroid

        # Mean distance to centroid across frames: shape (N,)
        dist_to_centroid = np.mean(np.linalg.norm(rel_pos, axis=-1), axis=0)

        # 2. Acceleration relative to global centroid: shape (N,)
        if T >= 3:
            # Second central/backward differences of relative position: shape (T-2, N, 3)
            acc = rel_pos[2:] - 2.0 * rel_pos[1:-1] + rel_pos[:-2]
            acc_mag = np.linalg.norm(acc, axis=-1)  # shape: (T-2, N)
            acc_var = np.var(acc_mag, axis=0)  # shape: (N,)
        else:
            acc_var = np.zeros(N, dtype=np.float32)

        # Normalize metrics to [0, 1] range
        dist_ptp = float(np.ptp(dist_to_centroid))
        if dist_ptp > 1e-6:
            norm_dist = (dist_to_centroid - np.min(dist_to_centroid)) / dist_ptp
        else:
            norm_dist = np.zeros(N, dtype=np.float32)

        acc_ptp = float(np.ptp(acc_var))
        if acc_ptp > 1e-6:
            norm_acc = (acc_var - np.min(acc_var)) / acc_ptp
        else:
            norm_acc = np.zeros(N, dtype=np.float32)

        scores = self.dist_weight * norm_dist + self.acc_weight * norm_acc
        best_idx = int(np.argmin(scores))

        return best_idx, point_names[best_idx]

    def infer_kinematic_tree(
        self,
        mean_dist: np.ndarray,
        var_dist: np.ndarray,
        lambda_param: Optional[float] = None,
    ) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
        """
        Step c: Kinematic Tree Inference via Minimum Spanning Tree (MST).
        Builds undirected weighted graph where edge weight W_ij = mu_ij + lambda * sigma^2_ij
        (penalizing non-rigid pairs) and computes the MST using Kruskal's algorithm.

        Args:
            mean_dist: Mean pairwise distance matrix (N, N).
            var_dist: Variance pairwise distance matrix (N, N).
            lambda_param: Optional override for rigidity penalty.

        Returns:
            Tuple of (mst_edges, edge_weight_matrix) where mst_edges is list of (u, v, weight).
        """
        N = mean_dist.shape[0]
        if N <= 1:
            return [], np.zeros((N, N), dtype=np.float32)

        lam = self.lambda_param if lambda_param is None else float(lambda_param)

        # Edge weight W_ij = mu_ij + lambda * sigma^2_ij
        edge_weights = mean_dist + lam * var_dist
        np.fill_diagonal(edge_weights, 0.0)

        # Collect candidate edges (i < j)
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                edges.append((float(edge_weights[i, j]), i, j))

        # Sort edges ascending by weight
        edges.sort(key=lambda item: item[0])

        # Disjoint-Set (Union-Find) with path compression and rank optimization
        parent = list(range(N))
        rank = [0] * N

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        mst_edges: List[Tuple[int, int, float]] = []
        for weight, u, v in edges:
            if union(u, v):
                mst_edges.append((u, v, weight))
                if len(mst_edges) == N - 1:
                    break

        return mst_edges, edge_weights

    def build_hierarchy(
        self,
        mst_edges: List[Tuple[int, int, float]],
        root_idx: int,
        point_names: List[str],
        trajectory: np.ndarray,
        mean_dist: np.ndarray,
        creature_name: str = "discovered_creature",
        rest_frame_idx: Optional[int] = None,
    ) -> Skeleton:
        """
        Step d & e: Directed Kinematic Hierarchy & Articulated Skeleton Construction.
        Performs BFS starting from Root Joint to establish parent-child relationships,
        bone directions, rest lengths, and canonical rest orientations.

        Args:
            mst_edges: Undirected MST edges (u, v, weight).
            root_idx: Root joint index.
            point_names: List of joint names.
            trajectory: (T, N, 3) coordinates.
            mean_dist: (N, N) mean distances.
            creature_name: Name for the discovered skeleton.
            rest_frame_idx: Frame index used for canonical bind pose geometry.

        Returns:
            Articulated Skeleton instance.
        """
        N = len(point_names)
        root_name = point_names[root_idx]
        frame_idx = self.rest_frame_idx if rest_frame_idx is None else int(rest_frame_idx)
        frame_idx = max(0, min(frame_idx, trajectory.shape[0] - 1))
        ref_frame = trajectory[frame_idx]  # shape: (N, 3)

        # 1. Build adjacency list from undirected MST edges
        adj: Dict[int, List[int]] = {i: [] for i in range(N)}
        for u, v, _ in mst_edges:
            adj[u].append(v)
            adj[v].append(u)

        # 2. Breadth-First Search (BFS) to establish directed tree rooted at root_idx
        parent_map: Dict[int, Optional[int]] = {root_idx: None}
        children_map: Dict[int, List[int]] = {i: [] for i in range(N)}
        visited: Set[int] = {root_idx}
        queue: deque[int] = deque([root_idx])
        bfs_order: List[int] = []

        while queue:
            curr = queue.popleft()
            bfs_order.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = curr
                    children_map[curr].append(neighbor)
                    queue.append(neighbor)

        # Handle any disconnected components (safety check)
        for i in range(N):
            if i not in visited:
                parent_map[i] = root_idx
                children_map[root_idx].append(i)
                visited.add(i)
                bfs_order.append(i)

        # 3. Compute mathematically consistent rest positions along bone directions
        rest_world_positions = np.zeros((N, 3), dtype=np.float32)
        rest_world_positions[root_idx] = ref_frame[root_idx].copy()

        for curr in bfs_order:
            for child in children_map[curr]:
                bone_len = float(mean_dist[curr, child])
                if bone_len <= 1e-6:
                    bone_len = 0.1

                direction = ref_frame[child] - ref_frame[curr]
                dir_norm = float(np.linalg.norm(direction))
                if dir_norm > 1e-6:
                    unit_dir = direction / dir_norm
                else:
                    unit_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                rest_world_positions[child] = rest_world_positions[curr] + unit_dir * bone_len

        # 4. Construct Skeleton
        skel = Skeleton(name=creature_name, root_name=root_name)

        # 5. Populate JointNodes
        for i, name in enumerate(point_names):
            p_idx = parent_map[i]
            p_name = point_names[p_idx] if p_idx is not None else None
            c_names = [point_names[c] for c in children_map[i]]
            rest_pos = rest_world_positions[i].astype(np.float32)
            cur_pos = ref_frame[i].astype(np.float32)

            node = JointNode(
                name=name,
                parent_name=p_name,
                children_names=c_names,
                rest_world_pos=rest_pos.copy(),
                world_pos=cur_pos.copy(),
                local_pos=rest_pos.copy() if p_idx is None else (rest_pos - rest_world_positions[p_idx]),
                world_rotation=QuaternionMath.identity(),
                local_rotation=QuaternionMath.identity(),
            )

            # Matrices
            mat_world = np.eye(4, dtype=np.float32)
            mat_world[:3, 3] = rest_pos
            node.world_matrix = mat_world

            mat_local = np.eye(4, dtype=np.float32)
            mat_local[:3, 3] = node.local_pos
            node.local_matrix = mat_local

            skel.joints[name] = node

        # 6. Populate BoneEdges
        for curr in bfs_order:
            for child in children_map[curr]:
                p_name = point_names[curr]
                c_name = point_names[child]
                b_name = f"Bone_{p_name}_{c_name}"
                bone_len = float(mean_dist[curr, child])
                if bone_len <= 1e-6:
                    bone_len = 0.1

                bone = BoneEdge(
                    name=b_name,
                    parent_joint=p_name,
                    child_joint=c_name,
                    rest_length=bone_len,
                )
                skel.bones[b_name] = bone

        # 7. Validate constructed skeletal hierarchy
        skel.validate_hierarchy()

        return skel

    def discover(
        self,
        trajectory: Union[np.ndarray, Dict[str, np.ndarray], List[PoseFrame]],
        point_names: Optional[List[str]] = None,
        creature_name: str = "discovered_creature",
        root_name: Optional[str] = None,
        lambda_param: Optional[float] = None,
    ) -> Skeleton:
        """
        Executes end-to-end unsupervised automatic skeleton discovery.

        Args:
            trajectory: 3D point trajectory (array, dict, or PoseFrame list).
            point_names: Optional point/joint names.
            creature_name: Name assigned to the discovered Skeleton.
            root_name: Optional manual root joint name.
            lambda_param: Optional override for rigidity penalty parameter.

        Returns:
            Articulated, validated Skeleton instance.
        """
        coords, names = self._parse_trajectory(trajectory, point_names)
        self.point_names = names

        logger.info(
            f"Discovering skeleton '{creature_name}' from trajectory: "
            f"{coords.shape[0]} frames, {coords.shape[1]} joints."
        )

        # Step a: Pairwise distance variance
        mean_dist, var_dist = self.compute_pairwise_distance_variance(coords)
        self.mean_distance_matrix = mean_dist
        self.variance_distance_matrix = var_dist

        # Step b: Root identification
        root_idx, identified_root = self.identify_root_joint(coords, names, root_name=root_name)
        self.root_index = root_idx
        self.root_name = identified_root

        # Step c: MST kinematic tree inference
        mst_edges, weights = self.infer_kinematic_tree(mean_dist, var_dist, lambda_param=lambda_param)
        self.mst_edges = mst_edges
        self.edge_weights = weights

        # Step d & e: Directed hierarchy & Skeleton construction
        skeleton = self.build_hierarchy(
            mst_edges=mst_edges,
            root_idx=root_idx,
            point_names=names,
            trajectory=coords,
            mean_dist=mean_dist,
            creature_name=creature_name,
        )
        self.skeleton = skeleton

        logger.info(
            f"Successfully discovered skeleton '{skeleton.name}': "
            f"Root='{skeleton.root_name}', {len(skeleton.joints)} joints, {len(skeleton.bones)} bones."
        )

        return skeleton

    @classmethod
    def discover_skeleton(
        cls,
        trajectory: Union[np.ndarray, Dict[str, np.ndarray], List[PoseFrame]],
        point_names: Optional[List[str]] = None,
        creature_name: str = "discovered_creature",
        lambda_param: float = 100.0,
        root_name: Optional[str] = None,
    ) -> Skeleton:
        """Classmethod convenience entrypoint for one-shot discovery."""
        instance = cls(lambda_param=lambda_param)
        return instance.discover(
            trajectory=trajectory,
            point_names=point_names,
            creature_name=creature_name,
            root_name=root_name,
        )


def discover_from_pose_sequence(
    poses: List[PoseFrame],
    creature_name: str = "discovered_creature",
    lambda_param: float = 100.0,
) -> Skeleton:
    """
    Convenience helper function: discovers an articulated skeleton hierarchy from
    an observed sequence of PoseFrames.

    Args:
        poses: Chronological sequence of PoseFrame objects.
        creature_name: Identifier name for the created Skeleton.
        lambda_param: Rigidity variance penalty coefficient.

    Returns:
        Articulated Skeleton instance.
    """
    engine = AutomaticSkeletonDiscovery(lambda_param=lambda_param)
    return engine.discover(poses, creature_name=creature_name)
