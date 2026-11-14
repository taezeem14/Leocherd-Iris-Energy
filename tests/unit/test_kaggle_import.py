"""Unit test for Kaggle GPU pose data importer."""
import json
import pytest
from pathlib import Path
from iris.models.adapters.kaggle_import import KagglePoseImportAdapter
from iris.core.pipeline import ExecutionContext, Pipeline
from iris.motion.reconstruction import MotionProcessingStage
from iris.rigging.topology import SkeletonReconstructionStage
from iris.retargeting.animation import AnimationGenerationStage
from iris.export import AssetExportStage

def test_kaggle_import_adapter(tmp_path: Path):
    dummy_data = {
        "generator": "KaggleWorker",
        "morphology": "human",
        "fps": 15.0,
        "total_frames": 2,
        "frames": [
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "joints": {
                    "Hips": {"x": 0.0, "y": 1.0, "z": 0.0, "confidence": 1.0},
                    "Chest": {"x": 0.0, "y": 1.45, "z": 0.0, "confidence": 1.0},
                    "LeftArm": {"x": -0.20, "y": 1.45, "z": 0.0, "confidence": 1.0},
                    "RightArm": {"x": 0.20, "y": 1.45, "z": 0.0, "confidence": 1.0},
                }
            },
            {
                "frame_index": 1,
                "timestamp_sec": 0.066,
                "joints": {
                    "Hips": {"x": 0.0, "y": 1.02, "z": 0.05, "confidence": 1.0},
                    "Chest": {"x": 0.0, "y": 1.47, "z": 0.05, "confidence": 1.0},
                    "LeftArm": {"x": -0.22, "y": 1.40, "z": 0.05, "confidence": 1.0},
                    "RightArm": {"x": 0.22, "y": 1.40, "z": 0.05, "confidence": 1.0},
                }
            }
        ]
    }
    json_file = tmp_path / "test_kaggle_poses.json"
    with open(json_file, "w") as f:
        json.dump(dummy_data, f)

    adapter = KagglePoseImportAdapter(json_file)
    poses = adapter.estimate_sequence([])
    assert len(poses) == 2
    assert poses[0].get_joint("Hips").y == 1.0
    assert poses[1].get_joint("Hips").y == 1.02

    context = ExecutionContext(
        input_path=str(json_file),
        output_dir=str(tmp_path / "out"),
        morphology_name="human",
        target_fps=15,
    )
    context.pose_sequence = poses

    pipeline = Pipeline("TestKaggleImport")
    pipeline.add_stage(MotionProcessingStage())
    pipeline.add_stage(SkeletonReconstructionStage())
    pipeline.add_stage(AnimationGenerationStage())
    pipeline.add_stage(AssetExportStage(formats=["gltf", "bvh"]))

    res = pipeline.execute(context)
    assert "gltf" in res.exported_assets
    assert "bvh" in res.exported_assets
    assert Path(res.exported_assets["gltf"]).exists()
