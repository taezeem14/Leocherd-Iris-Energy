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


class DiscoveredSkeleton(Skeleton):
    """Articulated skeleton discovered from motion trajectories with symmetry metadata."""

    def __init__(
        self,
        name: str,
        root_name: str,
        symmetry_plane: Optional[Tuple[np.ndarray, float]] = None,
        symmetry_pairs: Optional[Dict[str, str]] = None,
        symmetry_bone_pairs: Optional[Dict[str, str]] = None,
        midline_joints: Optional[List[str]] = None,
        midline_bones: Optional[List[str]] = None,
        morphology_family: str = "discovered",
    ) -> None:
        super().__init__(name=name, root_name=root_name)
        self.symmetry_plane = symmetry_plane  # (normal: np.ndarray, offset: float)
        self.symmetry_pairs: Dict[str, str] = symmetry_pairs or {}
        self.symmetry_bone_pairs: Dict[str, str] = symmetry_bone_pairs or {}
        self.midline_joints: List[str] = midline_joints or []
        self.midline_bones: List[str] = midline_bones or []
        self.morphology_family = morphology_family


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
        self.symmetry_plane: Optional[Tuple[np.ndarray, float]] = None
        self.symmetry_pairs: Dict[str, str] = {}
        self.symmetry_bone_pairs: Dict[str, str] = {}
        self.midline_joints: List[str] = []
        self.midline_bones: List[str] = []

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

    def detect_bilateral_symmetry(
        self,
        trajectory: np.ndarray,
        point_names: List[str],
    ) -> Tuple[Tuple[np.ndarray, float], Dict[str, str], List[str]]:
        """
        Detects the biological reflection symmetry plane (n . x + d = 0) and
        matches bilateral joint pairs and midline joints using PCA, candidate normals,
        gated mutual nearest neighbors, and SVD plane refinement in pure NumPy.

        Args:
            trajectory: 3D point trajectory of shape (T, N, 3).
            point_names: List of joint names of length N.

        Returns:
            Tuple of ((normal, offset), symmetry_pairs, midline_joints).
        """
        import math

        T, N, _ = trajectory.shape
        if N <= 1:
            plane = (np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0)
            return plane, {}, list(point_names)

        # 1. Canonical time-averaged positions and kinematic centroid
        x_bar = np.mean(trajectory, axis=0)  # (N, 3)
        c = np.mean(x_bar, axis=0)  # (3,)
        x_tilde = x_bar - c  # (N, 3)

        diffs = x_bar[:, None, :] - x_bar[None, :, :]
        pair_dists = np.linalg.norm(diffs, axis=-1)
        D_max = float(np.max(pair_dists))
        if D_max < 1e-4:
            D_max = 1.0

        # 2. Candidate normal vectors
        candidates: List[np.ndarray] = []

        # 2a. PCA eigenvectors
        cov = np.dot(x_tilde.T, x_tilde) / float(max(1, N))
        _, eigvecs = np.linalg.eigh(cov)
        for col in range(3):
            candidates.append(eigvecs[:, col])

        # 2b. Standard Cartesian axes
        candidates.extend([
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ])

        # 2c. Pairwise difference vectors between points equidistant from centroid
        rad_dists = np.linalg.norm(x_tilde, axis=-1)
        for i in range(N):
            for j in range(i + 1, N):
                if abs(rad_dists[i] - rad_dists[j]) < 0.08 * D_max:
                    diff_vec = x_bar[i] - x_bar[j]
                    d_norm = float(np.linalg.norm(diff_vec))
                    if d_norm > 0.02 * D_max:
                        candidates.append(diff_vec / d_norm)

        # Deduplicate candidate normals
        unique_candidates: List[np.ndarray] = []
        for cand in candidates:
            c_norm = float(np.linalg.norm(cand))
            if c_norm < 1e-6:
                continue
            cand_unit = (cand / c_norm).astype(np.float32)
            if not any(abs(float(np.dot(cand_unit, prev))) > 0.99 for prev in unique_candidates):
                unique_candidates.append(cand_unit)

        # 3. Score candidate reflection planes
        best_score = -float("inf")
        best_normal = unique_candidates[0]
        best_d = -float(np.dot(best_normal, c))
        best_pairs: List[Tuple[int, int, float]] = []
        best_midline: List[int] = []

        tau_midline = max(0.04 * D_max, 0.02)
        sigma_sym = 0.05 * D_max
        sigma_mid = 0.03 * D_max
        max_cost = 0.25 * D_max

        for cand_n in unique_candidates:
            cand_d = -float(np.dot(cand_n, c))
            s = np.dot(x_bar, cand_n) + cand_d
            h = np.abs(s)

            midline_idx = [i for i in range(N) if h[i] <= tau_midline]
            pos_idx = [i for i in range(N) if s[i] > tau_midline]
            neg_idx = [i for i in range(N) if s[i] < -tau_midline]

            if not pos_idx or not neg_idx:
                score = sum(math.exp(-(float(h[m]) ** 2) / (2.0 * (sigma_mid ** 2))) for m in midline_idx)
                unmatched = N - len(midline_idx)
                score -= 0.5 * unmatched
                if score > best_score:
                    best_score = score
                    best_normal = cand_n
                    best_d = cand_d
                    best_pairs = []
                    best_midline = midline_idx
                continue

            # Compute cost matrix between pos and neg sets
            cost_matrix = np.zeros((len(pos_idx), len(neg_idx)), dtype=np.float32)
            for ip, i in enumerate(pos_idx):
                refl_i = x_bar[i] - 2.0 * (np.dot(cand_n, x_bar[i]) + cand_d) * cand_n
                refl_traj_i = trajectory[:, i, :] - 2.0 * (np.dot(trajectory[:, i, :], cand_n)[:, None] + cand_d) * cand_n
                for ineg, j in enumerate(neg_idx):
                    geom_dist = float(np.linalg.norm(refl_i - x_bar[j]))
                    traj_dist = float(np.mean(np.linalg.norm(refl_traj_i - trajectory[:, j, :], axis=-1)))
                    cost_matrix[ip, ineg] = geom_dist + 0.3 * traj_dist

            # Gated Mutual Nearest Neighbors (MNN)
            best_neg_for_pos = np.argmin(cost_matrix, axis=1)
            best_pos_for_neg = np.argmin(cost_matrix, axis=0)

            pairs: List[Tuple[int, int, float]] = []
            matched_pos = set()
            matched_neg = set()

            for ip, ineg in enumerate(best_neg_for_pos):
                if best_pos_for_neg[ineg] == ip and cost_matrix[ip, ineg] <= max_cost:
                    pairs.append((pos_idx[ip], neg_idx[ineg], float(cost_matrix[ip, ineg])))
                    matched_pos.add(ip)
                    matched_neg.add(ineg)

            # Greedy matching for remaining unmatched
            rem_pos = [ip for ip in range(len(pos_idx)) if ip not in matched_pos]
            rem_neg = [ineg for ineg in range(len(neg_idx)) if ineg not in matched_neg]
            if rem_pos and rem_neg:
                rem_costs = []
                for ip in rem_pos:
                    for ineg in rem_neg:
                        if cost_matrix[ip, ineg] <= max_cost:
                            rem_costs.append((cost_matrix[ip, ineg], ip, ineg))
                rem_costs.sort(key=lambda x: x[0])
                for cost_val, ip, ineg in rem_costs:
                    if ip not in matched_pos and ineg not in matched_neg:
                        pairs.append((pos_idx[ip], neg_idx[ineg], float(cost_val)))
                        matched_pos.add(ip)
                        matched_neg.add(ineg)

            unmatched = N - (2 * len(pairs) + len(midline_idx))
            if not pairs:
                score = sum(math.exp(-(float(h[m]) ** 2) / (2.0 * (sigma_mid ** 2))) for m in midline_idx) - 0.5 * unmatched
            else:
                pair_score = sum(2.0 * math.exp(-(c_val ** 2) / (2.0 * (sigma_sym ** 2))) for _, _, c_val in pairs)
                mid_score = sum(math.exp(-(float(h[m]) ** 2) / (2.0 * (sigma_mid ** 2))) for m in midline_idx)
                bilateral_bonus = 3.0 * len(pairs)
                score = pair_score + mid_score + bilateral_bonus - 0.5 * unmatched

            if score > best_score:
                best_score = score
                best_normal = cand_n
                best_d = cand_d
                best_pairs = pairs
                best_midline = midline_idx

        # 4. SVD Least-Squares Plane Refinement
        coplanar_pts: List[np.ndarray] = []
        for i, j, _ in best_pairs:
            coplanar_pts.append(0.5 * (x_bar[i] + x_bar[j]))
        for m in best_midline:
            coplanar_pts.append(x_bar[m])

        if len(coplanar_pts) >= 3:
            pts_mat = np.array(coplanar_pts, dtype=np.float32)
            c_svd = np.mean(pts_mat, axis=0)
            z_mat = pts_mat - c_svd
            _, _, vh = np.linalg.svd(z_mat, full_matrices=False)
            n_refined = vh[2, :].astype(np.float32)
            n_refined = n_refined / (np.linalg.norm(n_refined) + 1e-8)
            if np.dot(n_refined, best_normal) < 0:
                n_refined = -n_refined
            d_refined = -float(np.dot(n_refined, c_svd))
        else:
            n_refined = best_normal
            d_refined = best_d

        # 5. Canonical plane normal orientation:
        # If bilateral pairs exist, orient normal to point from left to right (x_bar[j] - x_bar[i])
        if best_pairs:
            avg_pair_vec = np.mean([x_bar[j] - x_bar[i] for i, j, _ in best_pairs], axis=0)
            if np.dot(n_refined, avg_pair_vec) < 0:
                n_refined = -n_refined
                d_refined = -d_refined
        else:
            if abs(float(n_refined[0])) > 0.99:
                if n_refined[0] < 0:
                    n_refined = -n_refined
                    d_refined = -d_refined
            elif np.dot(n_refined, np.array([1.0, 0.0, 0.0], dtype=np.float32)) < -1e-6:
                n_refined = -n_refined
                d_refined = -d_refined

        # Re-partition and re-match with refined plane
        s_final = np.dot(x_bar, n_refined) + d_refined
        h_final = np.abs(s_final)
        final_midline = [i for i in range(N) if h_final[i] <= tau_midline]
        final_pos = [i for i in range(N) if s_final[i] > tau_midline]
        final_neg = [i for i in range(N) if s_final[i] < -tau_midline]

        final_pairs: List[Tuple[int, int]] = []
        if final_pos and final_neg:
            cost_mat = np.zeros((len(final_pos), len(final_neg)), dtype=np.float32)
            for ip, i in enumerate(final_pos):
                refl_i = x_bar[i] - 2.0 * (np.dot(n_refined, x_bar[i]) + d_refined) * n_refined
                for ineg, j in enumerate(final_neg):
                    cost_mat[ip, ineg] = float(np.linalg.norm(refl_i - x_bar[j]))
            best_neg = np.argmin(cost_mat, axis=1)
            best_pos = np.argmin(cost_mat, axis=0)
            m_pos, m_neg = set(), set()
            for ip, ineg in enumerate(best_neg):
                if best_pos[ineg] == ip and cost_mat[ip, ineg] <= max_cost:
                    final_pairs.append((final_pos[ip], final_neg[ineg]))
                    m_pos.add(ip)
                    m_neg.add(ineg)
            rem_p = [ip for ip in range(len(final_pos)) if ip not in m_pos]
            rem_n = [ineg for ineg in range(len(final_neg)) if ineg not in m_neg]
            if rem_p and rem_n:
                rem_costs = []
                for ip in rem_p:
                    for ineg in rem_n:
                        if cost_mat[ip, ineg] <= max_cost:
                            rem_costs.append((cost_mat[ip, ineg], ip, ineg))
                rem_costs.sort(key=lambda x: x[0])
                for _, ip, ineg in rem_costs:
                    if ip not in m_pos and ineg not in m_neg:
                        final_pairs.append((final_pos[ip], final_neg[ineg]))
                        m_pos.add(ip)
                        m_neg.add(ineg)
                        m_pos.add(ip)
                        m_neg.add(ineg)

        symmetry_pairs: Dict[str, str] = {}
        for i, j in final_pairs:
            name_i = point_names[i]
            name_j = point_names[j]
            symmetry_pairs[name_i] = name_j
            symmetry_pairs[name_j] = name_i

        midline_joints = [point_names[m] for m in final_midline]

        return (n_refined, d_refined), symmetry_pairs, midline_joints

    def identify_root_joint(
        self,
        trajectory: np.ndarray,
        point_names: List[str],
        root_name: Optional[str] = None,
        midline_indices: Optional[List[int]] = None,
    ) -> Tuple[int, str]:
        """
        Step b: Root Joint Identification.
        Finds the point closest to the kinematic center of mass with minimal variance
        in acceleration relative to global centroid, optionally constrained to midline joints.

        Args:
            trajectory: 3D point trajectory of shape (T, N, 3).
            point_names: List of joint names.
            root_name: Optional explicit root joint name override.
            midline_indices: Optional list of joint indices belonging to the creature's sagittal midline.

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

        # Constrain root joint selection to detected midline joints when available
        if midline_indices:
            best_cand = int(np.argmin(scores[midline_indices]))
            best_idx = midline_indices[best_cand]
        else:
            best_idx = int(np.argmin(scores))

        return best_idx, point_names[best_idx]

    def infer_kinematic_tree(
        self,
        mean_dist: np.ndarray,
        var_dist: np.ndarray,
        lambda_param: Optional[float] = None,
        joint_symmetry_map: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
        """
        Step c: Kinematic Tree Inference via Minimum Spanning Tree (MST).
        Builds undirected weighted graph where edge weight W_ij = mu_ij + lambda * sigma^2_ij
        (penalizing non-rigid pairs) and computes the MST using Kruskal's algorithm.
        Optionally symmetrizes edge weights using bilateral joint pairings.

        Args:
            mean_dist: Mean pairwise distance matrix (N, N).
            var_dist: Variance pairwise distance matrix (N, N).
            lambda_param: Optional override for rigidity penalty.
            joint_symmetry_map: Optional index-to-index symmetric counterpart mapping.

        Returns:
            Tuple of (mst_edges, edge_weight_matrix) where mst_edges is list of (u, v, weight).
        """
        N = mean_dist.shape[0]
        if N <= 1:
            return [], np.zeros((N, N), dtype=np.float32)

        lam = self.lambda_param if lambda_param is None else float(lambda_param)

        # Edge weight W_ij = mu_ij + lambda * sigma^2_ij
        edge_weights = mean_dist + lam * var_dist
        if joint_symmetry_map:
            sym_weights = edge_weights.copy()
            for i in range(N):
                i_sym = joint_symmetry_map.get(i, i)
                for j in range(N):
                    j_sym = joint_symmetry_map.get(j, j)
                    sym_weights[i, j] = 0.5 * (edge_weights[i, j] + edge_weights[i_sym, j_sym])
            edge_weights = sym_weights

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
        symmetry_plane: Optional[Tuple[np.ndarray, float]] = None,
        symmetry_pairs: Optional[Dict[str, str]] = None,
        midline_joints: Optional[List[str]] = None,
    ) -> DiscoveredSkeleton:
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
            symmetry_plane: Optional detected reflection plane normal and offset.
            symmetry_pairs: Optional bilateral joint pair mapping.
            midline_joints: Optional list of joints on sagittal midline.

        Returns:
            Articulated DiscoveredSkeleton instance.
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

        # 4. Construct DiscoveredSkeleton
        skel = DiscoveredSkeleton(
            name=creature_name,
            root_name=root_name,
            symmetry_plane=symmetry_plane,
            symmetry_pairs=symmetry_pairs,
            midline_joints=midline_joints,
        )

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

        # 6b. Symmetric Bone Pairing & Midline Bones
        joint_sym_map = {name: name for name in point_names}
        if symmetry_pairs:
            for u, v in symmetry_pairs.items():
                joint_sym_map[u] = v

        midline_bones: List[str] = []
        symmetry_bone_pairs: Dict[str, str] = {}

        for b_name, bone in skel.bones.items():
            u_prime = joint_sym_map.get(bone.parent_joint, bone.parent_joint)
            v_prime = joint_sym_map.get(bone.child_joint, bone.child_joint)
            if u_prime == bone.parent_joint and v_prime == bone.child_joint:
                midline_bones.append(b_name)
                symmetry_bone_pairs[b_name] = b_name
            else:
                counterpart_name = f"Bone_{u_prime}_{v_prime}"
                if counterpart_name in skel.bones:
                    symmetry_bone_pairs[b_name] = counterpart_name
                    symmetry_bone_pairs[counterpart_name] = b_name

        skel.midline_bones = midline_bones
        skel.symmetry_bone_pairs = symmetry_bone_pairs

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
    ) -> DiscoveredSkeleton:
        """
        Executes end-to-end unsupervised automatic skeleton discovery with bilateral symmetry.

        Args:
            trajectory: 3D point trajectory (array, dict, or PoseFrame list).
            point_names: Optional point/joint names.
            creature_name: Name assigned to the discovered Skeleton.
            root_name: Optional manual root joint name.
            lambda_param: Optional override for rigidity penalty parameter.

        Returns:
            Articulated, validated DiscoveredSkeleton instance.
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

        # Step a.2: Detect bilateral reflection symmetry plane & joint pairings
        sym_plane, sym_pairs, midline_joints = self.detect_bilateral_symmetry(coords, names)
        self.symmetry_plane = sym_plane
        self.symmetry_pairs = sym_pairs
        self.midline_joints = midline_joints

        name_to_idx = {name: i for i, name in enumerate(names)}
        joint_symmetry_map = {i: i for i in range(len(names))}
        for u_name, v_name in sym_pairs.items():
            if u_name in name_to_idx and v_name in name_to_idx:
                joint_symmetry_map[name_to_idx[u_name]] = name_to_idx[v_name]
        midline_indices = [name_to_idx[m] for m in midline_joints if m in name_to_idx]

        # Step b: Root identification constrained to midline
        root_idx, identified_root = self.identify_root_joint(
            coords, names, root_name=root_name, midline_indices=midline_indices
        )
        self.root_index = root_idx
        self.root_name = identified_root

        # Step c: MST kinematic tree inference with symmetrized edge weights
        mst_edges, weights = self.infer_kinematic_tree(
            mean_dist, var_dist, lambda_param=lambda_param, joint_symmetry_map=joint_symmetry_map
        )
        self.mst_edges = mst_edges
        self.edge_weights = weights

        # For unbranched linear chains (serpentine / worms where all node degrees <= 2),
        # ensure root is assigned to an endpoint (leaf, degree 1) to form a sequential chain.
        if root_name is None and len(names) >= 3 and mst_edges:
            node_degrees: Dict[int, int] = {i: 0 for i in range(len(names))}
            for u, v, _ in mst_edges:
                node_degrees[u] += 1
                node_degrees[v] += 1
            if max(node_degrees.values()) <= 2:
                endpoints = [i for i, deg in node_degrees.items() if deg == 1]
                if endpoints and root_idx not in endpoints:
                    root_idx = min(endpoints)
                    identified_root = names[root_idx]
                    self.root_index = root_idx
                    self.root_name = identified_root

        # Step d & e: Directed hierarchy & Skeleton construction
        skeleton = self.build_hierarchy(
            mst_edges=mst_edges,
            root_idx=root_idx,
            point_names=names,
            trajectory=coords,
            mean_dist=mean_dist,
            creature_name=creature_name,
            symmetry_plane=sym_plane,
            symmetry_pairs=sym_pairs,
            midline_joints=midline_joints,
        )
        self.symmetry_bone_pairs = skeleton.symmetry_bone_pairs
        self.midline_bones = skeleton.midline_bones
        self.skeleton = skeleton

        logger.info(
            f"Successfully discovered skeleton '{skeleton.name}': "
            f"Root='{skeleton.root_name}', {len(skeleton.joints)} joints, {len(skeleton.bones)} bones."
        )

        return skeleton

    def discover_from_trajectory(
        self,
        trajectory: Union[np.ndarray, Dict[str, np.ndarray], List[PoseFrame]],
        point_names: Optional[List[str]] = None,
        creature_name: str = "discovered_creature",
        root_name: Optional[str] = None,
        lambda_param: Optional[float] = None,
    ) -> DiscoveredSkeleton:
        """Alias for discover conforming to M2 interface contract."""
        return self.discover(
            trajectory=trajectory,
            point_names=point_names,
            creature_name=creature_name,
            root_name=root_name,
            lambda_param=lambda_param,
        )

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
