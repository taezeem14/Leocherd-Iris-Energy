"""
Unit tests for 2D Creature Image-to-3D Skinned Rig and Multi-Engine Exports.
"""

from pathlib import Path
import numpy as np
import pytest

from iris.mesh.creature_rig import (
    CreatureImageAnalyzer,
    CreatureMorphologyProfile,
    CreatureReconstructionPipeline,
    CreatureRigBuilder,
    CreatureSkinnedMeshGenerator,
)


def test_creature_analysis_and_rig_generation(tmp_path: Path):
    creature_img = Path("data/samples/creature_reference.jpg")
    assert creature_img.exists(), "Creature reference sample must exist"

    # 1. Silhouette & Morphology Analysis
    profile, texture = CreatureImageAnalyzer.analyze_image(creature_img)
    assert profile.height_m > 1.0
    assert profile.is_stocky is True
    assert "Hips" in profile.anchors
    assert "LeftArm" in profile.anchors
    assert texture.shape[0] > 100

    # 2. Skeleton Construction (19 joints including LeftShoulder and RightShoulder)
    skeleton = CreatureRigBuilder.build_skeleton(profile)
    assert len(skeleton.joints) == 19
    assert len(skeleton.bones) == 18
    assert "LeftShoulder" in skeleton.joints
    assert "RightShoulder" in skeleton.joints
    assert skeleton.root_name == "Hips"

    # 3. Skinned Mesh Generation
    v, n, uvs, f, ji, jw = CreatureSkinnedMeshGenerator.generate_mesh(skeleton, profile, texture)
    assert len(v) > 500
    assert len(f) > 500
    assert uvs.shape == (len(v), 2)
    assert ji.shape == (len(v), 4)
    assert jw.shape == (len(v), 4)

    # Weights must sum to 1.0
    weight_sums = np.sum(jw, axis=1)
    np.testing.assert_allclose(weight_sums, 1.0, atol=1e-3)

    # 4. End-to-End Pipeline Export
    pipeline = CreatureReconstructionPipeline()
    out_dir = tmp_path / "creature_test_export"
    res = pipeline.process_creature(
        image_path=creature_img,
        output_dir=out_dir,
        export_formats=["gltf", "bvh"],
    )

    assert "gltf" in res
    assert Path(res["gltf"]).exists()
    assert "bvh" in res
    assert Path(res["bvh"]).exists()
    assert "texture" in res
    assert Path(res["texture"]).exists()
