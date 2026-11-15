"""
Rigging, skeleton graph, and deformation subsystem.
"""

from iris.rigging.ik import (
    FABRIKResult,
    FABRIKSolver,
    IKChain,
    TwoBoneIKResult,
    TwoBoneIKSolver,
)
from iris.rigging.joints import Bone, BoneEdge, JointNode
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton
from iris.rigging.skeleton_discovery import (
    AutomaticSkeletonDiscovery,
    discover_from_pose_sequence,
)
from iris.rigging.skinning import LinearBlendSkinning
from iris.rigging.topology import SkeletonReconstructionStage

__all__ = [
    "QuaternionMath",
    "JointNode",
    "BoneEdge",
    "Bone",
    "Skeleton",
    "AutomaticSkeletonDiscovery",
    "discover_from_pose_sequence",
    "LinearBlendSkinning",
    "SkeletonReconstructionStage",
    "TwoBoneIKSolver",
    "TwoBoneIKResult",
    "FABRIKSolver",
    "FABRIKResult",
    "IKChain",
]
