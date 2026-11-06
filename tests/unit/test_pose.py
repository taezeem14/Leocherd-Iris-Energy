"""
Unit tests for 3D pose estimation, confidence filtering, and anatomical synthesis.
"""

import numpy as np
import pytest
from iris.pose.base import Joint3D, PoseFrame
from iris.pose.confidence import ConfidenceFilter
from iris.pose.human import HumanPoseSynthesizer
from iris.models.adapters.kinematic_tracker import KinematicPoseEstimator


def test_joint_bounds_clamping():
    # Value over 1.0 should be clamped to 1.0
    j = Joint3D(name="test", x=0.0, y=1.0, z=0.0, confidence=1.5, visibility=2.5)
    assert j.confidence <= 1.0
    assert j.visibility <= 1.0


def test_human_synthesizer_virtual_joints():
    raw_joints = {
        "left_hip": Joint3D(name="left_hip", x=-0.15, y=0.9, z=0.0, confidence=0.9),
        "right_hip": Joint3D(name="right_hip", x=0.15, y=0.9, z=0.0, confidence=0.9),
        "left_shoulder": Joint3D(name="left_shoulder", x=-0.20, y=1.4, z=0.0, confidence=0.9),
        "right_shoulder": Joint3D(name="right_shoulder", x=0.20, y=1.4, z=0.0, confidence=0.9),
        "nose": Joint3D(name="nose", x=0.0, y=1.7, z=0.05, confidence=0.9),
    }

    rig = HumanPoseSynthesizer.synthesize_rig_joints(raw_joints)
    assert "Hips" in rig
    assert "Chest" in rig
    assert "Spine" in rig
    assert "Neck" in rig

    # Hips should be at midpoint of hips
    assert abs(rig["Hips"].x) < 1e-4
    assert abs(rig["Hips"].y - 0.9) < 1e-4


def test_confidence_filter():
    filt = ConfidenceFilter(min_confidence=0.35)
    frame = PoseFrame(
        frame_index=0,
        timestamp_sec=0.0,
        joints={
            f"j_{i}": Joint3D(name=f"j_{i}", x=0, y=0, z=0, confidence=0.8, visibility=0.8)
            for i in range(10)
        },
    )
    eval_res = filt.evaluate_frame(frame)
    assert eval_res["is_usable"] is True
    assert eval_res["high_confidence_count"] == 10
