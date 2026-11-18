"""
3D Surface Reconstruction Subsystem for Leocherd Iris Energy.
Integrates AI Monocular Depth Estimation (Depth Anything V2 / DPT),
Open3D Point Cloud Processing, and Structure-from-Motion (SfM).
"""

from iris.mesh.creature_rig import (
    CreatureImageAnalyzer,
    CreatureMorphologyProfile,
    CreatureReconstructionPipeline,
    CreatureRigBuilder,
    CreatureSkinnedMeshGenerator,
)
from iris.mesh.surface_reconstruction import (
    DepthAnythingMeshReconstructor,
    MonocularDepthReconstructor,
    Point3DCloud,
    SurfaceMesh,
)

__all__ = [
    "MonocularDepthReconstructor",
    "DepthAnythingMeshReconstructor",
    "Point3DCloud",
    "SurfaceMesh",
    "CreatureImageAnalyzer",
    "CreatureMorphologyProfile",
    "CreatureRigBuilder",
    "CreatureSkinnedMeshGenerator",
    "CreatureReconstructionPipeline",
]
