"""
Unit tests for video ingestion, validation, and frame extraction.
"""

from pathlib import Path
import pytest
from iris.video.metadata import VideoMetadata
from iris.video.sampling import FrameSampler
from iris.video.validation import VideoValidator
from iris.video.extraction import FrameExtractor


def test_sampling_schedule():
    # 60 frames @ 30 FPS downsampled to 15 FPS -> should yield 30 frames
    schedule = FrameSampler.calculate_sampling_indices(total_frames=60, source_fps=30.0, target_fps=15.0)
    assert len(schedule) == 30
    assert schedule[0] == (0, 0.0)
    assert schedule[1][0] == 2  # Step of 2.0


def test_sampling_source_lower_than_target():
    # 30 frames @ 10 FPS with target 15 FPS -> should retain all 30 frames
    schedule = FrameSampler.calculate_sampling_indices(total_frames=30, source_fps=10.0, target_fps=15.0)
    assert len(schedule) == 30


def test_video_validator_nonexistent():
    from iris.core.exceptions import VideoProcessingError
    with pytest.raises(VideoProcessingError):
        VideoValidator.validate_file("nonexistent_video.mp4")


def test_video_validator_real_sample():
    sample_path = Path("data/samples/sample_walk.mp4")
    if sample_path.exists():
        valid, issues = VideoValidator.validate_file(sample_path)
        assert valid is True
        assert len(issues) == 0


def test_frame_extractor_dimensions():
    extractor = FrameExtractor(max_dimension=640)
    w, h = extractor._calculate_scaled_dims(1920, 1080)
    assert max(w, h) == 640
    assert round(w / h, 2) == round(1920 / 1080, 2)
