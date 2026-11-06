"""
End-to-end integration test executing the complete video-to-3D rigging and motion reconstruction pipeline.
"""

from pathlib import Path
import pytest
from iris.core.pipeline import ExecutionContext, Pipeline
from iris.export import AssetExportStage
from iris.motion.reconstruction import MotionProcessingStage
from iris.pose.estimator import PoseEstimationStage
from iris.retargeting.animation import AnimationGenerationStage
from iris.rigging.topology import SkeletonReconstructionStage
from iris.video.ingestion import VideoIngestionStage


def test_end_to_end_pipeline(tmp_path):
    sample_video = Path("data/samples/sample_walk.mp4")
    assert sample_video.exists(), "Sample video fixture missing."

    out_dir = tmp_path / "integration_out"
    context = ExecutionContext(
        job_id="integ_01",
        input_path=str(sample_video),
        output_dir=str(out_dir),
        morphology_name="human",
        target_fps=15,
    )

    pipeline = Pipeline("IntegrationTestPipeline")
    pipeline.add_stage(VideoIngestionStage())
    pipeline.add_stage(PoseEstimationStage())
    pipeline.add_stage(MotionProcessingStage())
    pipeline.add_stage(SkeletonReconstructionStage())
    pipeline.add_stage(AnimationGenerationStage())
    pipeline.add_stage(AssetExportStage(formats=["gltf", "bvh", "blender"]))

    result = pipeline.execute(context)

    # Assert all intermediate stages populated
    assert result.video_metadata is not None
    assert len(result.frame_sequence) == 30
    assert len(result.pose_sequence) == 30
    assert result.motion_sequence is not None
    assert result.skeleton is not None
    assert len(result.skeleton.joints) == 19
    assert result.animation is not None
    assert result.animation.frame_count == 30

    # Assert exported files exist
    assert "gltf" in result.exported_assets
    assert Path(result.exported_assets["gltf"]).exists()
    assert "bvh" in result.exported_assets
    assert Path(result.exported_assets["bvh"]).exists()
    assert "blender" in result.exported_assets
    assert Path(result.exported_assets["blender"]).exists()
