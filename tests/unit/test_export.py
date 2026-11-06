"""
Unit tests for glTF 2.0 and BVH asset exporters.
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


@pytest.fixture
def dummy_rig_and_anim():
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)

    T = 10
    times = np.linspace(0, 1.0, T, dtype=np.float32)
    root_trans = np.zeros((T, 3), dtype=np.float32)
    rotations = {
        j_name: np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
        for j_name in skel.joints.keys()
    }

    anim = SkeletalAnimation(
        name="TestAnimation",
        fps=10.0,
        timestamps=times,
        root_joint_name="Hips",
        root_translations=root_trans,
        rotations=rotations,
    )
    return skel, anim


def test_gltf_export(tmp_path, dummy_rig_and_anim):
    skel, anim = dummy_rig_and_anim
    exporter = GltfExporter()
    out_file = tmp_path / "test.gltf"
    res = exporter.export(skel, anim, out_file)

    assert res.exists()
    with open(res, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["asset"]["version"] == "2.0"
    assert len(doc["nodes"]) == len(skel.joints)
    assert len(doc["animations"]) == 1


def test_bvh_export(tmp_path, dummy_rig_and_anim):
    skel, anim = dummy_rig_and_anim
    exporter = BVHExporter()
    out_file = tmp_path / "test.bvh"
    res = exporter.export(skel, anim, out_file)

    assert res.exists()
    content = res.read_text(encoding="utf-8")
    assert "HIERARCHY" in content
    assert "ROOT Hips" in content
    assert "MOTION" in content
    assert "Frames: 10" in content
