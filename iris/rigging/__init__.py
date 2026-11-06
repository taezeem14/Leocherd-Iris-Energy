"""
Rigging, skeleton graph, and deformation subsystem.
"""

from iris.rigging.joints import BoneEdge, JointNode
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton
from iris.rigging.skinning import LinearBlendSkinning
from iris.rigging.topology import SkeletonReconstructionStage

__all__ = [
    "QuaternionMath",
    "JointNode",
    "BoneEdge",
    "Skeleton",
    "LinearBlendSkinning",
    "SkeletonReconstructionStage",
]
