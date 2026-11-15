"""
Unit tests for Universal Morphology Understanding & Classification Module.
Tests synthetic human, quadruped, and serpentine trajectories,
as well as static landmark classification and morphology instantiation.
"""

import math
from typing import Dict, List
import numpy as np
import pytest

from iris.morphology import (
    BaseMorphology,
    BipedMorphology,
    CustomMorphology,
    HumanMorphology,
    MorphologyClassificationStage,
    MorphologyClassifier,
    MorphologyPrediction,
    MorphologyRegistry,
    MorphologyType,
    QuadrupedMorphology,
    SerpentineMorphology,
    TentacularMorphology,
)
from iris.pose.base import Joint3D, PoseFrame


def _create_synthetic_human_poses(num_frames: int = 10) -> List[PoseFrame]:
    """Generates synthetic human pose trajectory with upright vertical spine and 2 feet."""
    poses: List[PoseFrame] = []
    for f in range(num_frames):
        t = f * 0.05
        sway = 0.02 * math.sin(2.0 * math.pi * t)
        joints = {
            "Hips": Joint3D(name="Hips", x=0.0 + sway, y=1.00, z=0.0),
            "Spine": Joint3D(name="Spine", x=0.0 + sway, y=1.20, z=0.0),
            "Chest": Joint3D(name="Chest", x=0.0 + sway, y=1.45, z=0.0),
            "Neck": Joint3D(name="Neck", x=0.0 + sway, y=1.58, z=0.0),
            "Head": Joint3D(name="Head", x=0.0 + sway, y=1.72, z=0.0),
            # Arms
            "LeftShoulder": Joint3D(name="LeftShoulder", x=-0.20 + sway, y=1.45, z=0.0),
            "LeftArm": Joint3D(name="LeftArm", x=-0.35 + sway, y=1.20, z=0.0),
            "LeftHand": Joint3D(name="LeftHand", x=-0.35 + sway, y=0.90, z=0.0),
            "RightShoulder": Joint3D(name="RightShoulder", x=0.20 + sway, y=1.45, z=0.0),
            "RightArm": Joint3D(name="RightArm", x=0.35 + sway, y=1.20, z=0.0),
            "RightHand": Joint3D(name="RightHand", x=0.35 + sway, y=0.90, z=0.0),
            # Legs & Feet (2 persistent ground contacts at Y ≈ 0.08)
            "LeftUpLeg": Joint3D(name="LeftUpLeg", x=-0.12, y=0.90, z=0.0),
            "LeftLeg": Joint3D(name="LeftLeg", x=-0.12, y=0.48, z=0.0),
            "LeftFoot": Joint3D(name="LeftFoot", x=-0.12, y=0.08, z=0.05),
            "RightUpLeg": Joint3D(name="RightUpLeg", x=0.12, y=0.90, z=0.0),
            "RightLeg": Joint3D(name="RightLeg", x=0.12, y=0.48, z=0.0),
            "RightFoot": Joint3D(name="RightFoot", x=0.12, y=0.08, z=0.05),
        }
        poses.append(
            PoseFrame(
                frame_index=f,
                timestamp_sec=t,
                subject_id=1,
                morphology="human",
                joints=joints,
            )
        )
    return poses


def _create_synthetic_quadruped_poses(num_frames: int = 10) -> List[PoseFrame]:
    """Generates synthetic quadruped pose trajectory with horizontal spine and 4 paws."""
    poses: List[PoseFrame] = []
    for f in range(num_frames):
        t = f * 0.05
        bob = 0.01 * math.sin(2.0 * math.pi * t)
        joints = {
            # Horizontal Spine Chain
            "Pelvis": Joint3D(name="Pelvis", x=0.0, y=0.70 + bob, z=-0.40),
            "Spine_Lumbar": Joint3D(name="Spine_Lumbar", x=0.0, y=0.72 + bob, z=-0.20),
            "Spine_Thoracic": Joint3D(name="Spine_Thoracic", x=0.0, y=0.75 + bob, z=0.05),
            "Neck": Joint3D(name="Neck", x=0.0, y=0.85 + bob, z=0.35),
            "Head": Joint3D(name="Head", x=0.0, y=0.95 + bob, z=0.55),
            "Tail_Tip": Joint3D(name="Tail_Tip", x=0.0, y=0.50 + bob, z=-0.80),
            # Forelimbs (Front Paws at Y ≈ 0.05)
            "FrontLeft_Scapula": Joint3D(name="FrontLeft_Scapula", x=-0.15, y=0.72, z=0.10),
            "FrontLeft_Knee": Joint3D(name="FrontLeft_Knee", x=-0.15, y=0.40, z=0.10),
            "FrontLeft_Paw": Joint3D(name="FrontLeft_Paw", x=-0.15, y=0.05, z=0.15),
            "FrontRight_Scapula": Joint3D(name="FrontRight_Scapula", x=0.15, y=0.72, z=0.10),
            "FrontRight_Knee": Joint3D(name="FrontRight_Knee", x=0.15, y=0.40, z=0.10),
            "FrontRight_Paw": Joint3D(name="FrontRight_Paw", x=0.15, y=0.05, z=0.15),
            # Hindlimbs (Back Paws at Y ≈ 0.05)
            "BackLeft_Hip": Joint3D(name="BackLeft_Hip", x=-0.15, y=0.68, z=-0.38),
            "BackLeft_Knee": Joint3D(name="BackLeft_Knee", x=-0.15, y=0.38, z=-0.42),
            "BackLeft_Paw": Joint3D(name="BackLeft_Paw", x=-0.15, y=0.05, z=-0.35),
            "BackRight_Hip": Joint3D(name="BackRight_Hip", x=0.15, y=0.68, z=-0.38),
            "BackRight_Knee": Joint3D(name="BackRight_Knee", x=0.15, y=0.38, z=-0.42),
            "BackRight_Paw": Joint3D(name="BackRight_Paw", x=0.15, y=0.05, z=-0.35),
        }
        poses.append(
            PoseFrame(
                frame_index=f,
                timestamp_sec=t,
                subject_id=1,
                morphology="quadruped",
                joints=joints,
            )
        )
    return poses


def _create_synthetic_serpentine_poses(num_frames: int = 15, num_verts: int = 12) -> List[PoseFrame]:
    """Generates synthetic serpentine pose trajectory with undulating traveling wave and no limbs."""
    poses: List[PoseFrame] = []
    wavelength = 1.2
    speed = 2.0
    amplitude = 0.20

    for f in range(num_frames):
        t = f * 0.05
        joints: Dict[str, Joint3D] = {}
        for i in range(num_verts):
            z = i * 0.15
            # Traveling wave: x = A * sin(k*z - omega*t)
            x = amplitude * math.sin((2.0 * math.pi / wavelength) * z - speed * t)
            y = 0.05  # Resting directly on ground plane
            name = f"Vertebra_{i}"
            joints[name] = Joint3D(name=name, x=x, y=y, z=z)

        poses.append(
            PoseFrame(
                frame_index=f,
                timestamp_sec=t,
                subject_id=1,
                morphology="serpentine",
                joints=joints,
            )
        )
    return poses


def test_human_trajectory_classification():
    """Verify that upright vertical spine is correctly identified as MorphologyType.HUMAN with high confidence."""
    classifier = MorphologyClassifier()
    poses = _create_synthetic_human_poses(num_frames=10)

    prediction = classifier.classify_from_poses(poses)

    assert prediction.predicted_type == MorphologyType.HUMAN
    assert prediction.morphology_name == "human"
    assert prediction.confidence >= 0.85

    # Check key metrics
    assert prediction.metrics["spine_tilt_degrees"] <= 25.0  # Upright ~0°-25°
    assert prediction.metrics["ground_contact_count"] == 2.0  # 2 distinct feet
    assert prediction.metrics["aspect_ratio"] >= 1.0  # Tall upright proportions
    assert prediction.metrics["limb_chain_count"] == 4.0  # 4 limbs (2 arms, 2 legs)


def test_quadruped_trajectory_classification():
    """Verify that horizontal 4-limb motion is correctly classified as MorphologyType.QUADRUPED with high confidence."""
    classifier = MorphologyClassifier()
    poses = _create_synthetic_quadruped_poses(num_frames=10)

    prediction = classifier.classify_from_poses(poses)

    assert prediction.predicted_type == MorphologyType.QUADRUPED
    assert prediction.morphology_name == "quadruped"
    assert prediction.confidence >= 0.85

    # Check key metrics
    assert prediction.metrics["spine_tilt_degrees"] >= 65.0  # Horizontal ~65°-90°
    assert prediction.metrics["ground_contact_count"] == 4.0  # 4 persistent paws
    assert prediction.metrics["aspect_ratio"] < 1.0  # Length > Height
    assert prediction.metrics["limb_chain_count"] == 4.0  # 4 limbs


def test_serpentine_trajectory_classification():
    """Verify that linear undulating chain with no limbs is classified as MorphologyType.SERPENTINE."""
    classifier = MorphologyClassifier()
    poses = _create_synthetic_serpentine_poses(num_frames=15, num_verts=12)

    prediction = classifier.classify_from_poses(poses)

    assert prediction.predicted_type == MorphologyType.SERPENTINE
    assert prediction.morphology_name == "serpentine"
    assert prediction.confidence >= 0.85

    # Check key metrics
    assert prediction.metrics["ground_contact_count"] == 0.0  # No distinct limb contacts
    assert prediction.metrics["limb_chain_count"] == 0.0  # No branched limbs
    assert prediction.metrics["aspect_ratio"] < 0.35  # Very flat / elongated
    assert prediction.metrics["wave_propagation"] > 0.40  # Undulating traveling wave
    assert prediction.metrics["curvature_variance"] > 0.005  # S-curvature variation


def test_get_morphology_instance():
    """Verify that get_morphology_instance returns the proper BaseMorphology subclass."""
    classifier = MorphologyClassifier()

    # Human
    pred_human = MorphologyPrediction(
        predicted_type=MorphologyType.HUMAN,
        confidence=0.95,
        metrics={},
        morphology_name="human",
    )
    inst_human = classifier.get_morphology_instance(pred_human)
    assert isinstance(inst_human, HumanMorphology)
    assert isinstance(inst_human, BaseMorphology)
    assert inst_human.root_joint == "Hips"

    # Quadruped
    pred_quad = MorphologyPrediction(
        predicted_type=MorphologyType.QUADRUPED,
        confidence=0.95,
        metrics={},
        morphology_name="quadruped",
    )
    inst_quad = classifier.get_morphology_instance(pred_quad)
    assert isinstance(inst_quad, QuadrupedMorphology)
    assert isinstance(inst_quad, BaseMorphology)
    assert inst_quad.root_joint == "Pelvis"

    # Serpentine
    pred_serp = MorphologyPrediction(
        predicted_type=MorphologyType.SERPENTINE,
        confidence=0.95,
        metrics={},
        morphology_name="serpentine",
    )
    inst_serp = classifier.get_morphology_instance(pred_serp)
    assert isinstance(inst_serp, SerpentineMorphology)
    assert isinstance(inst_serp, BaseMorphology)
    assert inst_serp.root_joint == "Vertebra_0"

    # Tentacular
    pred_tent = MorphologyPrediction(
        predicted_type=MorphologyType.TENTACULAR,
        confidence=0.90,
        metrics={},
        morphology_name="tentacular",
    )
    inst_tent = classifier.get_morphology_instance(pred_tent)
    assert isinstance(inst_tent, TentacularMorphology)
    assert isinstance(inst_tent, BaseMorphology)
    assert inst_tent.root_joint == "CentralBody"

    # Avian / Biped
    pred_avian = MorphologyPrediction(
        predicted_type=MorphologyType.AVIAN,
        confidence=0.90,
        metrics={},
        morphology_name="avian",
    )
    inst_avian = classifier.get_morphology_instance(pred_avian)
    assert isinstance(inst_avian, BipedMorphology)
    assert isinstance(inst_avian, BaseMorphology)


def test_classify_from_landmarks_static_fallback():
    """Verify single-frame static posture fallback classifier."""
    classifier = MorphologyClassifier()

    # Human static posture
    human_pose = _create_synthetic_human_poses(num_frames=1)[0]
    pred_human = classifier.classify_from_landmarks(human_pose.joints)
    assert pred_human.predicted_type == MorphologyType.HUMAN
    assert pred_human.confidence >= 0.80

    # Quadruped static posture
    quad_pose = _create_synthetic_quadruped_poses(num_frames=1)[0]
    pred_quad = classifier.classify_from_landmarks(quad_pose.joints)
    assert pred_quad.predicted_type == MorphologyType.QUADRUPED
    assert pred_quad.confidence >= 0.80


def test_tentacular_radial_classification():
    """Verify classification of radial tentacular morphology."""
    classifier = MorphologyClassifier()
    # 6 radial tentacles radiating in 360°
    joints: Dict[str, Joint3D] = {
        "CentralBody": Joint3D(name="CentralBody", x=0.0, y=1.0, z=0.0)
    }
    for t_idx in range(6):
        angle = t_idx * (2.0 * math.pi / 6.0)
        dir_x = math.cos(angle)
        dir_z = math.sin(angle)
        for s_idx in range(3):
            dist = (s_idx + 1) * 0.3
            name = f"Tentacle_{t_idx}_Seg_{s_idx}"
            joints[name] = Joint3D(
                name=name,
                x=dir_x * dist,
                y=1.0 - s_idx * 0.1,
                z=dir_z * dist,
            )

    pose = PoseFrame(frame_index=0, timestamp_sec=0.0, joints=joints)
    pred = classifier.classify_from_poses([pose])

    assert pred.predicted_type == MorphologyType.TENTACULAR
    assert pred.metrics["radial_symmetry"] >= 0.50
    assert pred.metrics["limb_chain_count"] >= 5.0


def test_empty_poses_fallback():
    """Verify that an empty pose sequence returns CUSTOM with 0 confidence."""
    classifier = MorphologyClassifier()
    pred = classifier.classify_from_poses([])
    assert pred.predicted_type == MorphologyType.CUSTOM
    assert pred.confidence == 0.0


def test_morphology_classification_stage():
    """Verify execution of MorphologyClassificationStage within a pipeline execution context."""
    from iris.core.pipeline import ExecutionContext

    stage = MorphologyClassificationStage()
    ctx = ExecutionContext()
    ctx.pose_sequence = _create_synthetic_human_poses(num_frames=5)

    ctx = stage.run(ctx)
    assert ctx.morphology_prediction is not None
    assert ctx.morphology_prediction.predicted_type == MorphologyType.HUMAN
    assert ctx.morphology_name == "human"
    assert isinstance(ctx.morphology, HumanMorphology)
