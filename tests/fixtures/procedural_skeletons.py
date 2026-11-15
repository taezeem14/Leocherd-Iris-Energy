"""
Procedural Skeleton Fixtures.

Instantiates fully populated, kinematically validated Skeleton objects
across all five core morphologies:
1. Human / Biped (19 joints, 18 bones)
2. Quadruped (24 joints, 23 bones)
3. Serpentine (N vertebrae linear chain, default 12)
4. Avian (21-22 joints with wings and digitigrade limbs)
5. Tentacular (CentralBody hub with M radial arms and K segments each)

All skeletons satisfy skeleton.validate_hierarchy() (single root, acyclic, non-zero bone lengths),
and include bilateral symmetry mappings and axial joint annotations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from iris.morphology.base import BaseMorphology
from iris.morphology.biped import BipedMorphology
from iris.morphology.human import HumanMorphology
from iris.morphology.quadruped import QuadrupedMorphology
from iris.morphology.serpentine import SerpentineMorphology
from iris.morphology.tentacular import TentacularMorphology
from iris.rigging.joints import BoneEdge, JointNode
from iris.rigging.skeleton import Skeleton


def _annotate_skeleton(
    skel: Skeleton,
    morphology_name: str,
    symmetry_pairs: Dict[str, str],
    axial_joints: Set[str],
) -> Skeleton:
    """Attaches metadata attributes to a validated skeleton."""
    skel.validate_hierarchy()
    setattr(skel, "morphology_name", morphology_name)
    setattr(skel, "symmetry_pairs", dict(symmetry_pairs))
    setattr(skel, "axial_joints", set(axial_joints))
    return skel


# =============================================================================
# 1. Human Skeleton
# =============================================================================

def get_human_symmetry_pairs() -> Dict[str, str]:
    """Returns bilateral symmetry pairs for Human morphology."""
    pairs = {
        "LeftShoulder": "RightShoulder",
        "LeftArm": "RightArm",
        "LeftForeArm": "RightForeArm",
        "LeftHand": "RightHand",
        "LeftUpLeg": "RightUpLeg",
        "LeftLeg": "RightLeg",
        "LeftFoot": "RightFoot",
    }
    # Add reflexive pairs
    full_pairs = dict(pairs)
    for l, r in pairs.items():
        full_pairs[r] = l
    return full_pairs


def get_human_axial_joints() -> Set[str]:
    """Returns sagittal midline (axial) joints for Human morphology."""
    return {"Hips", "Spine", "Chest", "Neck", "Head"}


def create_human_skeleton() -> Skeleton:
    """Creates a validated 19-joint Human skeleton with standard T-Pose."""
    morph = HumanMorphology()
    skel = Skeleton.from_morphology(morph)
    return _annotate_skeleton(skel, "human", get_human_symmetry_pairs(), get_human_axial_joints())


# =============================================================================
# 2. Quadruped Skeleton
# =============================================================================

def get_quadruped_symmetry_pairs() -> Dict[str, str]:
    """Returns bilateral symmetry pairs for Quadruped morphology."""
    pairs = {
        "FrontLeft_Scapula": "FrontRight_Scapula",
        "FrontLeft_Knee": "FrontRight_Knee",
        "FrontLeft_Ankle": "FrontRight_Ankle",
        "FrontLeft_Paw": "FrontRight_Paw",
        "BackLeft_Hip": "BackRight_Hip",
        "BackLeft_Knee": "BackRight_Knee",
        "BackLeft_Hock": "BackRight_Hock",
        "BackLeft_Paw": "BackRight_Paw",
    }
    full_pairs = dict(pairs)
    for l, r in pairs.items():
        full_pairs[r] = l
    return full_pairs


def get_quadruped_axial_joints() -> Set[str]:
    """Returns sagittal midline (axial) joints for Quadruped morphology."""
    return {
        "Pelvis", "Spine_Lumbar", "Spine_Thoracic", "Neck", "Head",
        "Tail_Base", "Tail_Mid", "Tail_Tip",
    }


def create_quadruped_skeleton() -> Skeleton:
    """Creates a validated 24-joint Quadruped skeleton with horizontal spine and 4 limbs."""
    morph = QuadrupedMorphology()
    skel = Skeleton.from_morphology(morph)
    return _annotate_skeleton(skel, "quadruped", get_quadruped_symmetry_pairs(), get_quadruped_axial_joints())


# =============================================================================
# 3. Serpentine Skeleton
# =============================================================================

def get_serpentine_symmetry_pairs(num_vertebrae: int = 12) -> Dict[str, str]:
    """Returns bilateral symmetry pairs for Serpentine morphology (empty; all axial)."""
    return {}


def get_serpentine_axial_joints(num_vertebrae: int = 12) -> Set[str]:
    """Returns axial joints for Serpentine morphology (all vertebrae)."""
    return {f"Vertebra_{i}" for i in range(num_vertebrae)}


def create_serpentine_skeleton(num_vertebrae: int = 12, total_length: float = 2.0) -> Skeleton:
    """Creates a validated N-joint linear Serpentine skeleton along Z."""
    morph = SerpentineMorphology(segment_count=num_vertebrae, total_length=total_length)
    skel = Skeleton.from_morphology(morph)
    return _annotate_skeleton(
        skel,
        "serpentine",
        get_serpentine_symmetry_pairs(num_vertebrae),
        get_serpentine_axial_joints(num_vertebrae),
    )


# =============================================================================
# 4. Avian Skeleton
# =============================================================================

def get_avian_symmetry_pairs() -> Dict[str, str]:
    """Returns bilateral symmetry pairs for Avian morphology."""
    pairs = {
        "LeftWing_Base": "RightWing_Base",
        "LeftWing_Mid": "RightWing_Mid",
        "LeftWing_Tip": "RightWing_Tip",
        "LeftThigh": "RightThigh",
        "LeftKnee": "RightKnee",
        "LeftAnkle": "RightAnkle",
        "LeftFoot": "RightFoot",
    }
    full_pairs = dict(pairs)
    for l, r in pairs.items():
        full_pairs[r] = l
    return full_pairs


def get_avian_axial_joints(with_tail: bool = True) -> Set[str]:
    """Returns sagittal midline (axial) joints for Avian morphology."""
    axial = {"Hips", "Spine", "Chest", "Neck", "Head"}
    if with_tail:
        axial.update({"Tail_1", "Tail_2", "Tail_3"})
    return axial


def create_avian_skeleton(with_tail: bool = True) -> Skeleton:
    """Creates a validated Avian skeleton with wings, digitigrade legs, and tail."""
    morph = BipedMorphology(with_tail=with_tail)
    skel = Skeleton.from_morphology(morph)
    return _annotate_skeleton(skel, "avian", get_avian_symmetry_pairs(), get_avian_axial_joints(with_tail))


# =============================================================================
# 5. Tentacular Skeleton
# =============================================================================

def get_tentacular_symmetry_pairs(num_arms: int = 6, segs_per_arm: int = 4) -> Dict[str, str]:
    """
    Returns bilateral symmetry pairs for radial Tentacular morphology reflected across X=0.
    For arm a with azimuth angle theta, pairs with arm having azimuth pi - theta (or -theta).
    """
    pairs: Dict[str, str] = {}
    angle_step = (2.0 * math.pi) / num_arms
    for a in range(num_arms):
        angle_a = a * angle_step
        # Target reflection across X=0: x' = -x, z' = z -> angle' = pi - angle
        ref_angle = (math.pi - angle_a) % (2.0 * math.pi)
        best_partner = a
        min_diff = float("inf")
        for b in range(num_arms):
            if b == a:
                continue
            angle_b = (b * angle_step) % (2.0 * math.pi)
            diff = abs(angle_b - ref_angle)
            diff = min(diff, 2.0 * math.pi - diff)
            if diff < min_diff:
                min_diff = diff
                best_partner = b

        if min_diff < 0.05 and best_partner != a:
            for s in range(segs_per_arm):
                pairs[f"Tentacle_{a}_Seg_{s}"] = f"Tentacle_{best_partner}_Seg_{s}"
    return pairs


def get_tentacular_axial_joints() -> Set[str]:
    """Returns axial joints for Tentacular morphology."""
    return {"CentralBody"}


def create_tentacular_skeleton(num_arms: int = 6, segs_per_arm: int = 4) -> Skeleton:
    """Creates a validated radial Tentacular monster skeleton."""
    morph = TentacularMorphology(tentacle_count=num_arms, segments_per_tentacle=segs_per_arm)
    skel = Skeleton.from_morphology(morph)
    return _annotate_skeleton(
        skel,
        "tentacular",
        get_tentacular_symmetry_pairs(num_arms, segs_per_arm),
        get_tentacular_axial_joints(),
    )


# =============================================================================
# 6. Branching Tree & Pendulum Skeletons (Unit & Topology Verification)
# =============================================================================

def create_branching_tree_skeleton(levels: int = 3, branch_factor: int = 2) -> Skeleton:
    """
    Creates a procedural symmetric branching tree skeleton for graph topology verification.
    Root at [0, 0, 0], branching outward along +Y and ±X.
    """
    skel = Skeleton(name="BranchingTree", root_name="Root")
    root_node = JointNode(
        name="Root",
        parent_name=None,
        rest_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        local_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    skel.joints["Root"] = root_node

    current_level_nodes = [("Root", np.array([0.0, 0.0, 0.0], dtype=np.float32))]
    symmetry_pairs: Dict[str, str] = {}

    for lvl in range(1, levels):
        next_level_nodes = []
        seg_length = 0.5 / lvl
        spread = 0.3 / lvl

        for parent_name, parent_pos in current_level_nodes:
            for b in range(branch_factor):
                sign = -1.0 if b == 0 else 1.0
                node_name = f"Node_L{lvl}_{parent_name}_B{b}"
                pos = parent_pos + np.array([sign * spread, seg_length, 0.0], dtype=np.float32)
                node = JointNode(
                    name=node_name,
                    parent_name=parent_name,
                    rest_world_pos=pos.copy(),
                    world_pos=pos.copy(),
                    local_pos=pos - parent_pos,
                )
                skel.joints[node_name] = node
                skel.joints[parent_name].children_names.append(node_name)

                bone_name = f"Bone_{parent_name}_{node_name}"
                skel.bones[bone_name] = BoneEdge(
                    name=bone_name,
                    parent_joint=parent_name,
                    child_joint=node_name,
                    rest_length=float(np.linalg.norm(pos - parent_pos)),
                )
                next_level_nodes.append((node_name, pos))

            # Register symmetry if branch_factor == 2
            if branch_factor == 2:
                left_name = f"Node_L{lvl}_{parent_name}_B0"
                right_name = f"Node_L{lvl}_{parent_name}_B1"
                symmetry_pairs[left_name] = right_name
                symmetry_pairs[right_name] = left_name

        current_level_nodes = next_level_nodes

    return _annotate_skeleton(skel, "branching_tree", symmetry_pairs, {"Root"})


def create_pendulum_skeleton(num_links: int = 3, link_length: float = 0.5) -> Skeleton:
    """Creates a simple N-link linear chain skeleton along -Y."""
    skel = Skeleton(name="Pendulum", root_name="Anchor")
    anchor = JointNode(
        name="Anchor",
        parent_name=None,
        rest_world_pos=np.array([0.0, float(num_links * link_length), 0.0], dtype=np.float32),
    )
    skel.joints["Anchor"] = anchor

    parent = "Anchor"
    parent_pos = anchor.rest_world_pos
    for i in range(num_links):
        child_name = f"Link_{i}"
        pos = parent_pos - np.array([0.0, link_length, 0.0], dtype=np.float32)
        node = JointNode(
            name=child_name,
            parent_name=parent,
            rest_world_pos=pos.copy(),
            world_pos=pos.copy(),
            local_pos=np.array([0.0, -link_length, 0.0], dtype=np.float32),
        )
        skel.joints[child_name] = node
        skel.joints[parent].children_names.append(child_name)

        b_name = f"Bone_{parent}_{child_name}"
        skel.bones[b_name] = BoneEdge(
            name=b_name,
            parent_joint=parent,
            child_joint=child_name,
            rest_length=link_length,
        )
        parent = child_name
        parent_pos = pos

    return _annotate_skeleton(skel, "pendulum", {}, {"Anchor"})


# =============================================================================
# Dispatcher
# =============================================================================

def get_procedural_skeleton(morphology_name: str, **kwargs: Any) -> Skeleton:
    """Factory creating a validated procedural skeleton for any morphology name."""
    name = morphology_name.lower()
    if name in ("human", "biped"):
        return create_human_skeleton()
    elif name == "quadruped":
        return create_quadruped_skeleton()
    elif name == "serpentine":
        return create_serpentine_skeleton(
            num_vertebrae=kwargs.get("num_vertebrae", 12),
            total_length=kwargs.get("total_length", 2.0),
        )
    elif name == "avian":
        return create_avian_skeleton(with_tail=kwargs.get("with_tail", True))
    elif name == "tentacular":
        return create_tentacular_skeleton(
            num_arms=kwargs.get("num_arms", 6),
            segs_per_arm=kwargs.get("segs_per_arm", 4),
        )
    elif name == "branching_tree":
        return create_branching_tree_skeleton(
            levels=kwargs.get("levels", 3),
            branch_factor=kwargs.get("branch_factor", 2),
        )
    elif name == "pendulum":
        return create_pendulum_skeleton(
            num_links=kwargs.get("num_links", 3),
            link_length=kwargs.get("link_length", 0.5),
        )
    else:
        raise ValueError(f"Unknown morphology skeleton requested: '{morphology_name}'")
