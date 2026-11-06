"""
Unit tests for 3D quaternion mathematics and skeleton kinematics.
"""

import math
import numpy as np
import pytest
from iris.morphology.human import HumanMorphology
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton


def test_quaternion_identity():
    q_id = QuaternionMath.identity()
    assert np.allclose(q_id, [1.0, 0.0, 0.0, 0.0])


def test_quaternion_rotation_between_vectors():
    # 90 degrees around Z axis: vector (1, 0, 0) -> (0, 1, 0)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    q = QuaternionMath.rotation_between_vectors(v1, v2)

    # Rotating v1 by q should yield v2
    v1_rot = QuaternionMath.rotate_vector(q, v1)
    assert np.allclose(v1_rot, v2, atol=1e-5)


def test_skeleton_from_morphology():
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)
    assert skel.root_name == "Hips"
    assert len(skel.joints) == 19
    assert len(skel.bones) == 18


def test_skeleton_bone_length_enforcement():
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)

    # Intentionally stretch children positions
    for node in skel.joints.values():
        node.world_pos = node.world_pos * 1.5

    # Enforce lengths
    skel._enforce_lengths_recursive(skel.root_name)

    # Distance between Hips and Spine must match rest length exactly
    hips_pos = skel.joints["Hips"].world_pos
    spine_pos = skel.joints["Spine"].world_pos
    bone_len = float(np.linalg.norm(spine_pos - hips_pos))
    rest_len = skel.bones["Bone_Hips_Spine"].rest_length

    assert math.isclose(bone_len, rest_len, rel_tol=1e-4)
