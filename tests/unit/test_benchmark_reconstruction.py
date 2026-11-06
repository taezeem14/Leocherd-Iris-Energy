"""
Unit tests for the 3D Rigging & Asset Reconstruction Benchmarking Tool.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from iris.export.bvh import BVHExporter
from iris.export.gltf import GltfExporter
from iris.morphology.human import HumanMorphology
from iris.retargeting.animation import SkeletalAnimation
from iris.rigging.skeleton import Skeleton
from scripts.benchmark_reconstruction import (
    benchmark_reconstruction,
    compute_quaternion_metrics,
    compute_root_motion_metrics,
)


@pytest.fixture
def sample_rig_and_motion():
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)

    T = 15
    times = np.linspace(0, 1.0, T, dtype=np.float32)
    root_trans = np.zeros((T, 3), dtype=np.float32)
    # Give a small realistic step in Y and X
    root_trans[:, 0] = np.sin(times * np.pi) * 0.1
    root_trans[:, 1] = np.cos(times * np.pi) * 0.05

    rotations = {
        j_name: np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
        for j_name in skel.joints.keys()
    }

    anim = SkeletalAnimation(
        name="BenchmarkTestMotion",
        fps=15.0,
        timestamps=times,
        root_joint_name="Hips",
        root_translations=root_trans,
        rotations=rotations,
    )
    return skel, anim


def test_benchmark_reconstruction_valid(tmp_path, sample_rig_and_motion):
    skel, anim = sample_rig_and_motion
    gltf_p = tmp_path / "valid.gltf"
    bvh_p = tmp_path / "valid.bvh"

    GltfExporter().export(skel, anim, gltf_p)
    BVHExporter().export(skel, anim, bvh_p)

    report = benchmark_reconstruction(gltf_p, bvh_p)

    assert report.overall_status == "PASS"
    assert report.gltf_spec.valid is True
    assert report.gltf_spec.asset_version == "2.0"
    assert report.bvh_spec.valid is True
    assert report.bvh_spec.frame_count == 15
    assert report.quaternion_stats.is_valid is True
    assert abs(report.quaternion_stats.mean_norm - 1.0) < 1e-4
    assert report.gltf_root_motion is not None
    assert report.gltf_root_motion.is_physically_plausible is True
    assert report.bvh_root_motion is not None
    assert report.bvh_root_motion.is_physically_plausible is True


def test_quaternion_metrics_calculation():
    # Test normalized quaternions
    q_norm = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.70710678, 0.0, 0.70710678, 0.0],
    ], dtype=np.float32)
    stats = compute_quaternion_metrics({"Root": q_norm})
    assert stats.is_valid is True
    assert abs(stats.mean_norm - 1.0) < 1e-4
    assert stats.total_samples == 2

    # Test unnormalized quaternions
    q_unnorm = np.array([
        [1.5, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2],
    ], dtype=np.float32)
    stats_bad = compute_quaternion_metrics({"Root": q_unnorm})
    assert stats_bad.is_valid is False
    assert stats_bad.max_deviation > 0.4


def test_root_motion_metrics_calculation():
    translations = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ], dtype=np.float32)

    stats = compute_root_motion_metrics(translations, fps=30.0, source_name="Test")
    assert stats.frame_count == 3
    assert stats.is_physically_plausible is True
    assert pytest.approx(stats.total_trajectory_distance, rel=1e-3) == 0.2
    assert pytest.approx(stats.max_frame_displacement, rel=1e-3) == 0.1


def test_benchmark_detects_corrupted_gltf(tmp_path, sample_rig_and_motion):
    skel, anim = sample_rig_and_motion
    gltf_p = tmp_path / "corrupt.gltf"
    bvh_p = tmp_path / "valid.bvh"

    GltfExporter().export(skel, anim, gltf_p)
    BVHExporter().export(skel, anim, bvh_p)

    # Corrupt glTF version
    with open(gltf_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["asset"]["version"] = "1.0"
    with open(gltf_p, "w", encoding="utf-8") as f:
        json.dump(data, f)

    report = benchmark_reconstruction(gltf_p, bvh_p)
    assert report.overall_status == "FAIL"
    assert report.gltf_spec.valid is False
    assert any("Expected asset.version '2.0'" in iss.message for iss in report.issues)


def test_benchmark_detects_corrupted_bvh(tmp_path, sample_rig_and_motion):
    skel, anim = sample_rig_and_motion
    gltf_p = tmp_path / "valid.gltf"
    bvh_p = tmp_path / "corrupt.bvh"

    GltfExporter().export(skel, anim, gltf_p)
    BVHExporter().export(skel, anim, bvh_p)

    # Corrupt BVH by removing MOTION
    content = bvh_p.read_text(encoding="utf-8")
    corrupt_content = content.replace("MOTION", "BAD_SECTION")
    bvh_p.write_text(corrupt_content, encoding="utf-8")

    report = benchmark_reconstruction(gltf_p, bvh_p)
    assert report.overall_status == "FAIL"
    assert report.bvh_spec.valid is False
    assert any("Missing 'MOTION' block" in iss.message for iss in report.issues)
