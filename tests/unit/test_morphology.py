"""
Unit tests for morphology plugins and skeletal graph validation.
"""

import pytest
from iris.core.exceptions import MorphologyError
from iris.morphology import (
    HumanMorphology,
    MorphologyRegistry,
    QuadrupedMorphology,
    SerpentineMorphology,
    TentacularMorphology,
)


def test_human_morphology_validation():
    human = HumanMorphology()
    human.validate()
    assert human.root_joint == "Hips"
    assert len(human.joints) == 19
    assert len(human.bones) == 18


def test_quadruped_morphology_validation():
    quad = QuadrupedMorphology()
    quad.validate()
    assert quad.root_joint == "Pelvis"
    assert "Head" in quad.joints
    assert "Tail_Tip" in quad.joints


def test_serpentine_morphology():
    snake = SerpentineMorphology(segment_count=8, total_length=1.5)
    snake.validate()
    assert len(snake.joints) == 8
    assert len(snake.bones) == 7


def test_tentacular_morphology():
    tent = TentacularMorphology(tentacle_count=4, segments_per_tentacle=3)
    tent.validate()
    assert tent.root_joint == "CentralBody"
    assert len(tent.joints) == 1 + (4 * 3)  # 13 joints


def test_morphology_registry():
    assert "human" in MorphologyRegistry.list_available()
    assert "quadruped" in MorphologyRegistry.list_available()
    h = MorphologyRegistry.get("human")
    assert isinstance(h, HumanMorphology)
