"""
Unit tests for AI monocular depth and 3D surface mesh reconstruction.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from iris.mesh.surface_reconstruction import (
    MonocularDepthReconstructor,
    Point3DCloud,
    SurfaceMesh,
)


def test_point3d_cloud_downsampling_and_outliers():
    # Synthetic cluster of points + noise
    np.random.seed(42)
    cluster = np.random.randn(200, 3).astype(np.float32) * 0.1
    outliers = np.random.uniform(5.0, 10.0, size=(10, 3)).astype(np.float32)
    all_pts = np.vstack([cluster, outliers])
    colors = np.random.rand(len(all_pts), 3).astype(np.float32)

    pcd = Point3DCloud(points=all_pts, colors=colors)
    assert pcd.count == 210

    # Test outlier removal
    cleaned = pcd.remove_statistical_outliers(nb_neighbors=10, std_ratio=1.5)
    assert cleaned.count < pcd.count
    assert cleaned.count >= 180

    # Test voxel downsampling
    ds = cleaned.downsample(voxel_size=0.05)
    assert ds.count <= cleaned.count

    # Test normal estimation
    ds.estimate_normals(k=8)
    assert ds.normals is not None
    assert ds.normals.shape == ds.points.shape


def test_surface_mesh_obj_and_ply_export():
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    mesh = SurfaceMesh(vertices=verts, faces=faces, vertex_colors=colors)
    assert mesh.vertex_count == 3
    assert mesh.face_count == 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        obj_path = tmp_path / "test.obj"
        ply_path = tmp_path / "test.ply"

        mesh.save_obj(obj_path)
        mesh.save_ply(ply_path)

        assert obj_path.exists()
        assert obj_path.stat().st_size > 50

        assert ply_path.exists()
        assert ply_path.stat().st_size > 50


def test_monocular_depth_estimator_projection():
    recon = MonocularDepthReconstructor()

    # Synthetic RGB frame (64x64)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[16:48, 16:48] = [200, 150, 100]

    depth_map = recon.estimate_depth_map(rgb)
    assert depth_map.shape == (64, 64)
    assert depth_map.min() >= 0.0
    assert depth_map.max() > 0.0

    pcd = recon.project_to_point_cloud(rgb, depth_map, stride=4)
    assert pcd.count > 0
    assert pcd.points.shape[1] == 3

    mesh = recon.reconstruct_mesh_from_point_cloud(pcd)
    assert mesh.vertex_count > 0
    assert mesh.face_count > 0
