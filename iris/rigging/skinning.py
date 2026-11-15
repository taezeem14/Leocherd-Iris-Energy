"""
Skinning engine: Linear Blend Skinning (LBS), Dual Quaternion Skinning (DQS),
and advanced skin weighting (Volumetric, Geodesic, Heat Diffusion).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import csgraph, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from iris.rigging.orientation import DualQuaternion, DualQuaternionMath, QuaternionMath
from iris.rigging.skeleton import Skeleton


class SkinningMethod(str, Enum):
    """Available skinning and skin weight calculation methods."""
    VOLUMETRIC = "volumetric"
    GEODESIC = "geodesic"
    HEAT_DIFFUSION = "heat_diffusion"
    LBS = "lbs"
    DQS = "dqs"


def normalize_skin_weights(
    raw_weights: np.ndarray,
    joint_indices: Optional[np.ndarray] = None,
    max_influences: int = 4,
    index_dtype: np.dtype = np.uint16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sanitizes and normalizes skinning weights to satisfy strict glTF 2.0 partition of unity:
    1. Strict non-negativity: w >= 0.
    2. Top-K descending sort: w_0 >= w_1 >= ... >= w_{K-1}.
    3. Sum to 1.0 normalization: sum_{k=0}^{K-1} w_{ik} = 1.0 +- 1e-4.
       If sum < 1e-6, defaults to w_0 = 1.0, j_0 = 0.
    4. Zero-weight joint index sanitization: if w_{ik} == 0.0, enforces j_{ik} = 0.

    Returns:
        weights: (V, max_influences) float32
        joint_indices: (V, max_influences) of requested index_dtype (default uint16)
    """
    weights = np.asarray(raw_weights, dtype=np.float32)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    V, num_candidates = weights.shape
    K = max_influences

    # Sanitize NaN and Inf, clamp negative values to zero
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.maximum(weights, 0.0)

    if joint_indices is None:
        cand_indices = np.tile(np.arange(num_candidates, dtype=np.int64), (V, 1))
    else:
        cand_indices = np.asarray(joint_indices, dtype=np.int64)
        if cand_indices.ndim == 1:
            cand_indices = cand_indices.reshape(1, -1)

    out_weights = np.zeros((V, K), dtype=np.float32)
    out_indices = np.zeros((V, K), dtype=index_dtype)

    for v in range(V):
        w_row = weights[v]
        j_row = cand_indices[v]

        # Top-K influences sorted descending
        if len(w_row) <= K:
            sort_order = np.argsort(w_row)[::-1]
        else:
            part = np.argpartition(w_row, -K)[-K:]
            sort_order = part[np.argsort(w_row[part])[::-1]]

        top_w = w_row[sort_order[:K]].astype(np.float32)
        top_j = j_row[sort_order[:K]]

        if len(top_w) < K:
            pad_len = K - len(top_w)
            top_w = np.pad(top_w, (0, pad_len), mode="constant", constant_values=0.0)
            top_j = np.pad(top_j, (0, pad_len), mode="constant", constant_values=0)

        total_w = float(np.sum(top_w))
        if total_w < 1e-6:
            top_w = np.zeros(K, dtype=np.float32)
            top_w[0] = 1.0
            top_j = np.zeros(K, dtype=np.int64)
        else:
            top_w = top_w / total_w
            zero_mask = (top_w < 1e-6)
            top_w[zero_mask] = 0.0
            top_j[zero_mask] = 0

            rem_sum = float(np.sum(top_w))
            if rem_sum > 1e-6:
                top_w = top_w / rem_sum
            else:
                top_w[0] = 1.0
                top_j[0] = 0

        final_order = np.argsort(top_w)[::-1]
        top_w = top_w[final_order]
        top_j = top_j[final_order]

        for k in range(K):
            if top_w[k] == 0.0:
                top_j[k] = 0

        out_weights[v] = top_w
        out_indices[v] = top_j.astype(index_dtype)

    return out_weights, out_indices


class VolumetricSkinning:
    """Volumetric capsule skin weighting based on bone segment line distance and radial falloff."""

    @staticmethod
    def calculate_weights(
        vertices: np.ndarray,
        skeleton: Skeleton,
        max_influences: int = 4,
        falloff_power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        V = len(vertices)
        joint_names = list(skeleton.joints.keys())
        J = len(joint_names)
        joint_map = {name: i for i, name in enumerate(joint_names)}

        raw_weights = np.zeros((V, J), dtype=np.float32)

        # 1. Bone segment capsules: line segment (P_parent, P_child)
        for bone in skeleton.bones.values():
            if bone.parent_joint not in joint_map or bone.child_joint not in joint_map:
                continue
            j_p = joint_map[bone.parent_joint]
            j_c = joint_map[bone.child_joint]
            p1 = skeleton.joints[bone.parent_joint].rest_world_pos.astype(np.float32)
            p2 = skeleton.joints[bone.child_joint].rest_world_pos.astype(np.float32)

            bone_vec = p2 - p1
            bone_len_sq = float(np.dot(bone_vec, bone_vec))

            if bone_len_sq < 1e-8:
                dists = np.linalg.norm(vertices - p1, axis=1)
                w = 1.0 / np.power(dists + 0.05, falloff_power)
                raw_weights[:, j_p] += 0.5 * w
                raw_weights[:, j_c] += 0.5 * w
            else:
                v_vec = vertices - p1
                t = np.clip(np.dot(v_vec, bone_vec) / bone_len_sq, 0.0, 1.0)
                p_proj = p1 + t[:, None] * bone_vec
                dists = np.linalg.norm(vertices - p_proj, axis=1)
                w = 1.0 / np.power(dists + 0.02, falloff_power)

                raw_weights[:, j_p] += (1.0 - t) * w
                raw_weights[:, j_c] += t * w

        # 2. Joint center fallback for isolated joints
        for j_name, j_idx in joint_map.items():
            j_pos = skeleton.joints[j_name].rest_world_pos.astype(np.float32)
            dists = np.linalg.norm(vertices - j_pos, axis=1)
            w = 0.5 / np.power(dists + 0.05, falloff_power)
            raw_weights[:, j_idx] += w

        return normalize_skin_weights(raw_weights, max_influences=max_influences)


class GeodesicSkinning:
    """Geodesic distance skin weighting along mesh surface adjacency graph."""

    @staticmethod
    def calculate_weights(
        vertices: np.ndarray,
        faces: Optional[np.ndarray],
        skeleton: Skeleton,
        max_influences: int = 4,
        falloff_power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if faces is None or len(faces) == 0:
            return VolumetricSkinning.calculate_weights(
                vertices, skeleton, max_influences=max_influences, falloff_power=falloff_power
            )

        V = len(vertices)
        joint_names = list(skeleton.joints.keys())
        J = len(joint_names)

        faces_arr = np.asarray(faces, dtype=np.int64)

        e01_u, e01_v = faces_arr[:, 0], faces_arr[:, 1]
        e12_u, e12_v = faces_arr[:, 1], faces_arr[:, 2]
        e20_u, e20_v = faces_arr[:, 2], faces_arr[:, 0]

        all_u = np.concatenate([e01_u, e12_u, e20_u, e01_v, e12_v, e20_v])
        all_v = np.concatenate([e01_v, e12_v, e20_v, e01_u, e12_u, e20_u])

        diff = vertices[all_u] - vertices[all_v]
        dists = np.linalg.norm(diff, axis=1).astype(np.float32)

        adj_matrix = csr_matrix((dists, (all_u, all_v)), shape=(V, V))

        seed_indices = []
        for j_name in joint_names:
            j_pos = skeleton.joints[j_name].rest_world_pos.astype(np.float32)
            dists_to_j = np.linalg.norm(vertices - j_pos, axis=1)
            seed_idx = int(np.argmin(dists_to_j))
            seed_indices.append(seed_idx)

        geo_dists = csgraph.dijkstra(adj_matrix, directed=False, indices=seed_indices)

        raw_weights = np.zeros((V, J), dtype=np.float32)
        for j_idx, j_name in enumerate(joint_names):
            d = geo_dists[j_idx].copy()
            inf_mask = np.isinf(d)
            if np.any(inf_mask):
                j_pos = skeleton.joints[j_name].rest_world_pos.astype(np.float32)
                euc_d = np.linalg.norm(vertices[inf_mask] - j_pos, axis=1)
                d[inf_mask] = euc_d * 5.0

            raw_weights[:, j_idx] = 1.0 / np.power(d + 0.05, falloff_power)

        return normalize_skin_weights(raw_weights, max_influences=max_influences)


class HeatDiffusionSkinning:
    """
    Discrete Heat Diffusion skin weighting based on the cotangent Laplace-Beltrami
    operator and lumped mass matrix with Dirichlet boundary conditions.
    Automatically falls back to GeodesicSkinning if non-manifold, singular, or non-invertible.
    """

    @staticmethod
    def calculate_weights(
        vertices: np.ndarray,
        faces: Optional[np.ndarray],
        skeleton: Skeleton,
        max_influences: int = 4,
        falloff_power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if faces is None or len(faces) == 0:
            return GeodesicSkinning.calculate_weights(
                vertices, faces, skeleton, max_influences=max_influences, falloff_power=falloff_power
            )

        try:
            return HeatDiffusionSkinning._solve_heat_diffusion(
                vertices, faces, skeleton, max_influences=max_influences
            )
        except Exception:
            # Automatic fallback to geodesic distance skinning
            return GeodesicSkinning.calculate_weights(
                vertices, faces, skeleton, max_influences=max_influences, falloff_power=falloff_power
            )

    @staticmethod
    def _solve_heat_diffusion(
        vertices: np.ndarray,
        faces: np.ndarray,
        skeleton: Skeleton,
        max_influences: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        V = len(vertices)
        faces_arr = np.asarray(faces, dtype=np.int64)
        F = len(faces_arr)
        if V < 3 or F < 1:
            raise ValueError("Insufficient vertices or faces for cotangent Laplacian")

        vi = vertices[faces_arr[:, 0]]
        vj = vertices[faces_arr[:, 1]]
        vk = vertices[faces_arr[:, 2]]

        ei = vk - vj
        ej = vi - vk
        ek = vj - vi

        cross_k = np.cross(vj - vi, vk - vi)
        double_areas = np.linalg.norm(cross_k, axis=1)
        valid = double_areas > 1e-12
        if not np.any(valid):
            raise ValueError("All faces are degenerate")

        ei = ei[valid]
        ej = ej[valid]
        ek = ek[valid]
        double_areas = double_areas[valid]
        f_valid = faces_arr[valid]

        cot_i = -np.sum(ej * ek, axis=1) / double_areas
        cot_j = -np.sum(ek * ei, axis=1) / double_areas
        cot_k = -np.sum(ei * ej, axis=1) / double_areas

        w_jk = 0.5 * cot_i
        w_ki = 0.5 * cot_j
        w_ij = 0.5 * cot_k

        row_idx = np.concatenate([
            f_valid[:, 1], f_valid[:, 2],
            f_valid[:, 2], f_valid[:, 0],
            f_valid[:, 0], f_valid[:, 1],
        ])
        col_idx = np.concatenate([
            f_valid[:, 2], f_valid[:, 1],
            f_valid[:, 0], f_valid[:, 2],
            f_valid[:, 1], f_valid[:, 0],
        ])
        data_vals = np.concatenate([w_jk, w_jk, w_ki, w_ki, w_ij, w_ij])

        L_off = csr_matrix((data_vals, (row_idx, col_idx)), shape=(V, V))
        diag_vals = -np.array(L_off.sum(axis=1)).ravel()
        L = L_off + diags(diag_vals, 0, shape=(V, V), format="csr")

        tri_areas = double_areas * 0.5
        m_diag = np.zeros(V, dtype=np.float64)
        for i_v in range(3):
            np.add.at(m_diag, f_valid[:, i_v], tri_areas / 3.0)
        m_diag = np.maximum(m_diag, 1e-8)
        M = diags(m_diag, 0, shape=(V, V), format="csr")

        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        diag_len = float(np.linalg.norm(bbox_max - bbox_min))
        c = 1.0 / max(1e-4, diag_len * diag_len)
        A = -L + c * M

        joint_names = list(skeleton.joints.keys())
        J = len(joint_names)
        if J < 1:
            raise ValueError("Skeleton has no joints")

        seed_vertices = {}
        for j_idx, j_name in enumerate(joint_names):
            j_pos = skeleton.joints[j_name].rest_world_pos.astype(np.float32)
            dists = np.linalg.norm(vertices - j_pos, axis=1)
            best_v = int(np.argmin(dists))
            seed_vertices[j_idx] = best_v

        boundary_v_set = set(seed_vertices.values())
        if len(boundary_v_set) < 2 and J > 1:
            raise ValueError("All joints collapsed to same vertex")

        B = sorted(list(boundary_v_set))
        B_idx_map = {v_id: k for k, v_id in enumerate(B)}
        U = sorted(list(set(range(V)) - boundary_v_set))

        if len(U) == 0:
            raw_weights = np.zeros((V, J), dtype=np.float32)
            for j_idx, s_v in seed_vertices.items():
                raw_weights[s_v, j_idx] = 1.0
            return normalize_skin_weights(raw_weights, max_influences=max_influences)

        A_csr = A.tocsr()
        A_UU = A_csr[U, :][:, U]
        A_UB = A_csr[U, :][:, B]

        raw_weights = np.zeros((V, J), dtype=np.float32)

        for j_idx, s_v in seed_vertices.items():
            w_B = np.zeros(len(B), dtype=np.float64)
            if s_v in B_idx_map:
                w_B[B_idx_map[s_v]] = 1.0

            rhs = -A_UB.dot(w_B)
            w_U = spsolve(A_UU, rhs)

            raw_weights[U, j_idx] = np.maximum(w_U, 0.0)
            raw_weights[s_v, j_idx] = 1.0

        return normalize_skin_weights(raw_weights, max_influences=max_influences)


def calculate_skin_weights(
    mesh: Any,
    skeleton: Skeleton,
    method: Union[SkinningMethod, str] = SkinningMethod.HEAT_DIFFUSION,
    max_influences: int = 4,
    falloff_power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes skin weights and joint indices for a mesh and skeleton.
    Supports SkinningMethod.HEAT_DIFFUSION, GEODESIC, VOLUMETRIC, LBS, DQS.
    Returns:
        weights: (V, max_influences) float32, non-negative, sorted descending, sum to 1.0
        joint_indices: (V, max_influences) uint16, with zero-weight joint indices set to 0.
    """
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        vertices = mesh.vertices
        faces = mesh.faces
    elif isinstance(mesh, tuple) and len(mesh) >= 2:
        vertices = mesh[0]
        faces = mesh[1]
    elif isinstance(mesh, (list, np.ndarray)):
        vertices = np.asarray(mesh, dtype=np.float32)
        faces = None
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    vertices = np.asarray(vertices, dtype=np.float32)
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int64)

    if isinstance(method, str):
        method_str = method.lower()
    elif isinstance(method, SkinningMethod):
        method_str = method.value.lower()
    else:
        method_str = str(method).lower()

    if method_str in ("heat_diffusion", "heat"):
        return HeatDiffusionSkinning.calculate_weights(
            vertices, faces, skeleton, max_influences=max_influences, falloff_power=falloff_power
        )
    elif method_str in ("geodesic", "geo"):
        return GeodesicSkinning.calculate_weights(
            vertices, faces, skeleton, max_influences=max_influences, falloff_power=falloff_power
        )
    elif method_str in ("volumetric", "capsule", "lbs"):
        return VolumetricSkinning.calculate_weights(
            vertices, skeleton, max_influences=max_influences, falloff_power=falloff_power
        )
    elif method_str == "dqs":
        if faces is not None and len(faces) > 0:
            return HeatDiffusionSkinning.calculate_weights(
                vertices, faces, skeleton, max_influences=max_influences, falloff_power=falloff_power
            )
        else:
            return VolumetricSkinning.calculate_weights(
                vertices, skeleton, max_influences=max_influences, falloff_power=falloff_power
            )
    else:
        return VolumetricSkinning.calculate_weights(
            vertices, skeleton, max_influences=max_influences, falloff_power=falloff_power
        )


def deform_lbs(
    vertices: np.ndarray,
    normals: Optional[np.ndarray],
    joint_matrices: Union[List[np.ndarray], np.ndarray],
    weights: np.ndarray,
    joint_indices: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Deforms mesh vertices and normals using Linear Blend Skinning (LBS).
    v' = sum_k w_{vk} * (M_{j_{vk}} * [v, 1])[:3]
    n' = sum_k w_{vk} * (M_{j_{vk}, :3, :3} * n)
    """
    V = len(vertices)
    K = weights.shape[1]
    mats = np.asarray(joint_matrices, dtype=np.float32)

    v_hom = np.hstack([vertices, np.ones((V, 1), dtype=np.float32)])
    deformed_v = np.zeros_like(vertices, dtype=np.float32)
    deformed_n = np.zeros_like(normals, dtype=np.float32) if normals is not None else None

    for k in range(K):
        wk = weights[:, k:k+1]
        jk = joint_indices[:, k]
        M_k = mats[jk]

        v_k = np.einsum("vij,vj->vi", M_k[:, :3, :], v_hom)
        deformed_v += wk * v_k

        if normals is not None and deformed_n is not None:
            R_k = M_k[:, :3, :3]
            n_k = np.einsum("vij,vj->vi", R_k, normals)
            deformed_n += wk * n_k

    if deformed_n is not None:
        n_norms = np.linalg.norm(deformed_n, axis=1, keepdims=True)
        n_norms = np.where(n_norms < 1e-8, 1.0, n_norms)
        deformed_n = (deformed_n / n_norms).astype(np.float32)

    return deformed_v, deformed_n


def deform_dqs(
    vertices: np.ndarray,
    normals: Optional[np.ndarray],
    joint_dqs: Union[List[DualQuaternion], np.ndarray],
    weights: np.ndarray,
    joint_indices: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Deforms mesh vertices and normals using Dual Quaternion Skinning (DQS).
    Applies Dual Quaternion Linear Blending (DLB) with anti-flip handling per vertex.
    Preserves volume under extreme joint twists and eliminates the candy-wrapper artifact.
    """
    V = len(vertices)
    K = weights.shape[1]

    Q0 = np.array([dq.q0 for dq in joint_dqs], dtype=np.float32)
    Q_eps = np.array([dq.q_eps for dq in joint_dqs], dtype=np.float32)

    j0 = joint_indices[:, 0]
    w0 = weights[:, 0:1]
    q0_piv = Q0[j0]

    b0 = w0 * q0_piv
    b_eps = w0 * Q_eps[j0]

    for k in range(1, K):
        jk = joint_indices[:, k]
        wk = weights[:, k:k+1]
        q0_k = Q0[jk]
        q_eps_k = Q_eps[jk]

        dots = np.sum(q0_piv * q0_k, axis=1, keepdims=True)
        signs = np.where(dots < 0.0, -1.0, 1.0)
        eff_w = wk * signs

        b0 += eff_w * q0_k
        b_eps += eff_w * q_eps_k

    norms = np.linalg.norm(b0, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    inv_norms = 1.0 / norms
    r0 = b0 * inv_norms
    r_eps = b_eps * inv_norms

    w0_val = r0[:, 0]
    x0_val = r0[:, 1]
    y0_val = r0[:, 2]
    z0_val = r0[:, 3]

    we_val = r_eps[:, 0]
    xe_val = r_eps[:, 1]
    ye_val = r_eps[:, 2]
    ze_val = r_eps[:, 3]

    tx = 2.0 * (-we_val * x0_val + xe_val * w0_val - ye_val * z0_val + ze_val * y0_val)
    ty = 2.0 * (-we_val * y0_val + xe_val * z0_val + ye_val * w0_val - ze_val * x0_val)
    tz = 2.0 * (-we_val * z0_val - xe_val * y0_val + ye_val * x0_val + ze_val * w0_val)
    t = np.column_stack([tx, ty, tz]).astype(np.float32)

    w_quat = r0[:, 0:1]
    qv = r0[:, 1:4]
    cross1_v = np.cross(qv, vertices)
    v_rot = vertices + 2.0 * (w_quat * cross1_v + np.cross(qv, cross1_v))
    deformed_v = (v_rot + t).astype(np.float32)

    deformed_n = None
    if normals is not None:
        cross1_n = np.cross(qv, normals)
        n_rot = normals + 2.0 * (w_quat * cross1_n + np.cross(qv, cross1_n))
        n_norms = np.linalg.norm(n_rot, axis=1, keepdims=True)
        n_norms = np.where(n_norms < 1e-8, 1.0, n_norms)
        deformed_n = (n_rot / n_norms).astype(np.float32)

    return deformed_v, deformed_n


class DualQuaternionSkinning:
    """Computes vertex bone weights and supports mesh deformation via DQS."""

    @staticmethod
    def calculate_bone_weights(
        vertices: np.ndarray,
        skeleton: Skeleton,
        max_influences: int = 4,
        falloff_power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return calculate_skin_weights(
            vertices, skeleton, method=SkinningMethod.HEAT_DIFFUSION, max_influences=max_influences, falloff_power=falloff_power
        )

    @staticmethod
    def deform(
        vertices: np.ndarray,
        normals: Optional[np.ndarray],
        joint_dqs: Union[List[DualQuaternion], np.ndarray],
        weights: np.ndarray,
        joint_indices: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return deform_dqs(vertices, normals, joint_dqs, weights, joint_indices)


class LinearBlendSkinning:
    """Computes vertex bone weights and supports mesh deformation via LBS."""

    @staticmethod
    def calculate_bone_weights(
        vertices: np.ndarray,
        skeleton: Skeleton,
        max_influences: int = 4,
        falloff_power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates joint indices and normalized weights for each vertex.
        Maintains backwards compatibility for existing tests and pipelines.
        Returns:
            joint_indices: (V, max_influences) int32
            joint_weights: (V, max_influences) float32, normalized to sum to 1.0
        """
        V = len(vertices)
        joint_names = list(skeleton.joints.keys())
        J = len(joint_names)
        dist_matrix = np.full((V, J), 1e6, dtype=np.float32)

        for j_idx, j_name in enumerate(joint_names):
            j_pos = skeleton.joints[j_name].rest_world_pos
            dists = np.linalg.norm(vertices - j_pos, axis=1)
            dist_matrix[:, j_idx] = dists

        raw_weights = 1.0 / np.power(dist_matrix + 0.05, falloff_power)
        weights, indices = normalize_skin_weights(
            raw_weights, max_influences=max_influences, index_dtype=np.int32
        )
        return indices, weights

    @staticmethod
    def deform(
        vertices: np.ndarray,
        normals: Optional[np.ndarray],
        joint_matrices: Union[List[np.ndarray], np.ndarray],
        weights: np.ndarray,
        joint_indices: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return deform_lbs(vertices, normals, joint_matrices, weights, joint_indices)
