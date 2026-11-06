"""
Skinning engine: Linear Blend Skinning (LBS) weight calculation and mesh binding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from iris.rigging.skeleton import Skeleton


class LinearBlendSkinning:
    """Computes vertex bone weights and supports mesh deformation via LBS."""

    @staticmethod
    def calculate_bone_weights(
        vertices: np.ndarray,  # Shape: (V, 3)
        skeleton: Skeleton,
        max_influences: int = 4,
        falloff_power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates joint indices and normalized weights for each vertex.
        Returns:
            joint_indices: (V, max_influences) int32
            joint_weights: (V, max_influences) float32, normalized to sum to 1.0
        """
        V = len(vertices)
        joint_names = list(skeleton.joints.keys())
        J = len(joint_names)
        joint_map = {name: i for i, name in enumerate(joint_names)}

        # Bone line segments: (P1, P2)
        bone_segments = []
        bone_joint_indices = []
        for bone in skeleton.bones.values():
            p1 = skeleton.joints[bone.parent_joint].rest_world_pos
            p2 = skeleton.joints[bone.child_joint].rest_world_pos
            bone_segments.append((p1, p2))
            bone_joint_indices.append((joint_map[bone.parent_joint], joint_map[bone.child_joint]))

        B = len(bone_segments)
        dist_matrix = np.full((V, J), 1e6, dtype=np.float32)

        # Compute point-to-joint distances
        for j_idx, j_name in enumerate(joint_names):
            j_pos = skeleton.joints[j_name].rest_world_pos
            dists = np.linalg.norm(vertices - j_pos, axis=1)
            dist_matrix[:, j_idx] = dists

        # Invert distances for weighting
        raw_weights = 1.0 / np.power(dist_matrix + 0.05, falloff_power)

        # Select top-K influences per vertex
        top_k = min(max_influences, J)
        joint_indices = np.zeros((V, max_influences), dtype=np.int32)
        joint_weights = np.zeros((V, max_influences), dtype=np.float32)

        for v in range(V):
            v_weights = raw_weights[v]
            # Get indices of highest weights
            top_indices = np.argsort(v_weights)[::-1][:top_k]
            top_vals = v_weights[top_indices]
            total_w = np.sum(top_vals)
            if total_w > 1e-6:
                top_vals = top_vals / total_w
            else:
                top_vals = np.zeros(top_k, dtype=np.float32)
                top_vals[0] = 1.0

            joint_indices[v, :top_k] = top_indices
            joint_weights[v, :top_k] = top_vals

        return joint_indices, joint_weights
