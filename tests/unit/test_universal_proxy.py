"""
Unit tests for Universal Procedural Proxy Mesh Generation and multi-morphology export.
"""

import numpy as np
import pytest
from iris.export.gltf import GltfExporter
from iris.morphology.human import HumanMorphology
from iris.morphology.quadruped import QuadrupedMorphology
from iris.morphology.serpentine import SerpentineMorphology
from iris.morphology.tentacular import TentacularMorphology
from iris.retargeting.animation import SkeletalAnimation
from iris.rigging.joints import JointNode
from iris.rigging.proxy_mesh import ProxyMeshGenerator
from iris.rigging.skeleton import Skeleton
from iris.rigging.skeleton_discovery import AutomaticSkeletonDiscovery


def assert_mesh_and_weights_valid(verts, faces, weights, j_indices, num_joints):
    # Shape & types
    assert verts.ndim == 2 and verts.shape[1] == 3 and verts.shape[0] > 0
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.shape[0] > 0
    assert weights.ndim == 2 and weights.shape[1] == 4 and weights.shape[0] == verts.shape[0]
    assert j_indices.ndim == 2 and j_indices.shape[1] == 4 and j_indices.shape[0] == verts.shape[0]

    # Partition of unity: sum of weights = 1.0 +- 1e-4
    w_sum = np.sum(weights, axis=1)
    assert np.allclose(w_sum, 1.0, atol=1e-4)

    # Non-negative weights
    assert np.all(weights >= 0.0)

    # Sorted descending
    for k in range(3):
        assert np.all(weights[:, k] >= weights[:, k + 1] - 1e-6)

    # Zero-weight reset: if w == 0.0, j == 0
    zero_mask = (weights == 0.0)
    assert np.all(j_indices[zero_mask] == 0)

    # In bounds
    assert np.all(j_indices < num_joints)
    assert np.min(faces) >= 0
    assert np.max(faces) < len(verts)

    # No NaNs or Infs
    assert not np.any(np.isnan(verts))
    assert not np.any(np.isnan(weights))


def test_universal_proxy_quadruped():
    skel = Skeleton.from_morphology(QuadrupedMorphology())
    verts, faces, weights, j_indices = ProxyMeshGenerator.create_procedural_mesh(skel)
    assert_mesh_and_weights_valid(verts, faces, weights, j_indices, len(skel.joints))
    assert len(verts) >= len(skel.bones) * 20


def test_universal_proxy_serpentine():
    skel = Skeleton.from_morphology(SerpentineMorphology(segment_count=10, total_length=2.0))
    verts, faces, weights, j_indices = ProxyMeshGenerator.create_procedural_mesh(skel)
    assert_mesh_and_weights_valid(verts, faces, weights, j_indices, len(skel.joints))


def test_universal_proxy_tentacular():
    skel = Skeleton.from_morphology(TentacularMorphology(tentacle_count=6, segments_per_tentacle=4))
    verts, faces, weights, j_indices = ProxyMeshGenerator.create_procedural_mesh(skel)
    assert_mesh_and_weights_valid(verts, faces, weights, j_indices, len(skel.joints))


def test_universal_proxy_discovered_skeleton():
    T, N = 30, 8
    np.random.seed(42)
    # Generate rigid moving point cloud
    base_pos = np.random.uniform(-1, 1, (N, 3)).astype(np.float32)
    traj = np.zeros((T, N, 3), dtype=np.float32)
    for t in range(T):
        traj[t] = base_pos + np.array([0.1 * np.sin(t * 0.2), 0.0, 0.0], dtype=np.float32)

    point_names = [f"node_{i}" for i in range(N)]
    discovery = AutomaticSkeletonDiscovery()
    skel = discovery.discover(traj, point_names, creature_name="alien_discovered")

    verts, faces, weights, j_indices = ProxyMeshGenerator.create_procedural_mesh(skel)
    assert_mesh_and_weights_valid(verts, faces, weights, j_indices, len(skel.joints))


def test_create_skinned_mannequin_automatic_fallback():
    # Humanoid still uses stylized mannequin
    skel_human = Skeleton.from_morphology(HumanMorphology())
    v_h, n_h, f_h, ji_h, jw_h = ProxyMeshGenerator.create_skinned_mannequin(skel_human)
    assert len(v_h) > 0

    # Quadruped falls back to procedural proxy without KeyError: 'Chest'
    skel_quad = Skeleton.from_morphology(QuadrupedMorphology())
    v_q, n_q, f_q, ji_q, jw_q = ProxyMeshGenerator.create_skinned_mannequin(skel_quad)
    assert len(v_q) > 0 and len(f_q) > 0
    assert_mesh_and_weights_valid(v_q, f_q, jw_q, ji_q, len(skel_quad.joints))

    # Serpentine falls back to procedural proxy without returning 0 vertices
    skel_serp = Skeleton.from_morphology(SerpentineMorphology())
    v_s, n_s, f_s, ji_s, jw_s = ProxyMeshGenerator.create_skinned_mannequin(skel_serp)
    assert len(v_s) > 0 and len(f_s) > 0
    assert_mesh_and_weights_valid(v_s, f_s, jw_s, ji_s, len(skel_serp.joints))


def test_gltf_export_universal_morphologies(tmp_path):
    exporter = GltfExporter()
    morphologies = [
        QuadrupedMorphology(),
        SerpentineMorphology(),
        TentacularMorphology(),
    ]
    for morph in morphologies:
        skel = Skeleton.from_morphology(morph)
        T = 5
        times = np.linspace(0, 1.0, T, dtype=np.float32)
        root_trans = np.zeros((T, 3), dtype=np.float32)
        rotations = {
            j_name: np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
            for j_name in skel.joints.keys()
        }
        anim = SkeletalAnimation(
            name=f"Anim_{morph.name}",
            fps=10.0,
            timestamps=times,
            root_joint_name=skel.root_name,
            root_translations=root_trans,
            rotations=rotations,
        )
        out_file = tmp_path / f"{morph.name}.gltf"
        res = exporter.export(skel, anim, out_file)
        assert res.exists()
        assert res.stat().st_size > 1000  # Non-trivial glTF file generated


def test_universal_proxy_watertight_manifold():
    p_start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p_end = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v, n, f, ji, jw = ProxyMeshGenerator.generate_capsule_segment(
        p_start, p_end, 0.05, 0.04, 0, 1, slices=12, rings=6, cap_rings=3
    )
    # Check Euler characteristic for topological sphere S^2: V - E + F = 2
    # In triangular mesh: 3 * F = 2 * E -> E = 3 * F / 2
    F = len(f)
    V = len(v)
    E = (3 * F) // 2
    chi = V - E + F
    assert chi == 2, f"Euler characteristic must be 2 for a closed manifold sphere, got {chi}"

    # Normals unit length
    norm_lens = np.linalg.norm(n, axis=1)
    assert np.allclose(norm_lens, 1.0, atol=1e-3)


def test_single_joint_skeleton_edge_case():
    skel = Skeleton("single_joint", "Root")
    skel.joints["Root"] = JointNode("Root")
    verts, faces, weights, j_indices = ProxyMeshGenerator.create_procedural_mesh(skel)
    assert len(verts) > 0
    assert len(faces) > 0
    assert weights[0, 0] == 1.0
    assert j_indices[0, 0] == 0
