"""
Pytest Shared Configuration and Fixtures for E2E Tests.

Exposes reusable fixtures for:
1. Temporary output directories with automatic cleanup
2. Procedural trajectories across all 5 creature morphologies
3. Validated Skeleton hierarchies across all 5 morphologies
4. Watertight synthetic meshes (twist cylinders, capsules) and volume tools
5. Opaque-box CLI and Benchmark Reconstruction runners
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest

from scripts.benchmark_reconstruction import (
    BenchmarkReport,
    benchmark_reconstruction,
)
from tests.fixtures import (
    CapsuleMesh,
    CylinderMesh,
    TrajectoryResult,
    compute_mesh_boundary,
    compute_mesh_volume,
    compute_rigidity_invariants,
    create_avian_skeleton,
    create_capsule_mesh,
    create_cylinder_twist_mesh,
    create_human_skeleton,
    create_quadruped_skeleton,
    create_serpentine_skeleton,
    create_tentacular_skeleton,
    generate_avian_trajectory,
    generate_human_trajectory,
    generate_quadruped_trajectory,
    generate_serpentine_trajectory,
    generate_tentacular_trajectory,
    get_avian_symmetry_pairs,
    get_human_symmetry_pairs,
    get_procedural_skeleton,
    get_quadruped_symmetry_pairs,
    get_serpentine_symmetry_pairs,
    get_tentacular_symmetry_pairs,
    is_mesh_watertight,
    verify_bone_rigidity,
)


# =============================================================================
# 1. Output & Directory Fixtures
# =============================================================================

@pytest.fixture
def e2e_output_dir(tmp_path: Path) -> Path:
    """Provides a fresh, isolated temporary output directory for E2E test runs."""
    out_dir = tmp_path / "e2e_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =============================================================================
# 2. Trajectory Fixtures
# =============================================================================

@pytest.fixture
def human_trajectory() -> TrajectoryResult:
    """Procedural human walking trajectory (30 frames @ 15 FPS)."""
    return generate_human_trajectory(frames=30, fps=15.0, noise_std=0.0)


@pytest.fixture
def quadruped_trajectory() -> TrajectoryResult:
    """Procedural quadruped trotting trajectory (30 frames @ 15 FPS)."""
    return generate_quadruped_trajectory(frames=30, fps=15.0, noise_std=0.0)


@pytest.fixture
def serpentine_trajectory() -> TrajectoryResult:
    """Procedural serpentine undulating trajectory (30 frames @ 15 FPS, 12 vertebrae)."""
    return generate_serpentine_trajectory(frames=30, fps=15.0, num_vertebrae=12, noise_std=0.0)


@pytest.fixture
def avian_trajectory() -> TrajectoryResult:
    """Procedural avian flight & flap trajectory (30 frames @ 15 FPS)."""
    return generate_avian_trajectory(frames=30, fps=15.0, noise_std=0.0)


@pytest.fixture
def tentacular_trajectory() -> TrajectoryResult:
    """Procedural tentacular monster trajectory (30 frames @ 15 FPS, 6 arms x 4 segs)."""
    return generate_tentacular_trajectory(frames=30, fps=15.0, num_arms=6, segs_per_arm=4, noise_std=0.0)


@pytest.fixture
def all_creature_trajectories(
    human_trajectory: TrajectoryResult,
    quadruped_trajectory: TrajectoryResult,
    serpentine_trajectory: TrajectoryResult,
    avian_trajectory: TrajectoryResult,
    tentacular_trajectory: TrajectoryResult,
) -> Dict[str, TrajectoryResult]:
    """Dictionary mapping morphology names to procedural trajectory arrays."""
    return {
        "human": human_trajectory,
        "quadruped": quadruped_trajectory,
        "serpentine": serpentine_trajectory,
        "avian": avian_trajectory,
        "tentacular": tentacular_trajectory,
    }


@pytest.fixture
def noisy_creature_trajectories() -> Dict[str, TrajectoryResult]:
    """Dictionary of creature trajectories with tracking noise (noise_std=0.01)."""
    return {
        "human": generate_human_trajectory(frames=30, fps=15.0, noise_std=0.01),
        "quadruped": generate_quadruped_trajectory(frames=30, fps=15.0, noise_std=0.01),
        "serpentine": generate_serpentine_trajectory(frames=30, fps=15.0, num_vertebrae=12, noise_std=0.01),
        "avian": generate_avian_trajectory(frames=30, fps=15.0, noise_std=0.01),
        "tentacular": generate_tentacular_trajectory(frames=30, fps=15.0, num_arms=6, segs_per_arm=4, noise_std=0.01),
    }


# =============================================================================
# 3. Skeleton Fixtures
# =============================================================================

@pytest.fixture
def human_skeleton():
    """Validated 19-joint Human skeleton."""
    return create_human_skeleton()


@pytest.fixture
def quadruped_skeleton():
    """Validated 24-joint Quadruped skeleton."""
    return create_quadruped_skeleton()


@pytest.fixture
def serpentine_skeleton():
    """Validated 12-vertebra Serpentine skeleton."""
    return create_serpentine_skeleton(num_vertebrae=12)


@pytest.fixture
def avian_skeleton():
    """Validated Avian skeleton with wings and tail."""
    return create_avian_skeleton(with_tail=True)


@pytest.fixture
def tentacular_skeleton():
    """Validated radial Tentacular monster skeleton."""
    return create_tentacular_skeleton(num_arms=6, segs_per_arm=4)


@pytest.fixture
def all_creature_skeletons(
    human_skeleton,
    quadruped_skeleton,
    serpentine_skeleton,
    avian_skeleton,
    tentacular_skeleton,
) -> Dict[str, Any]:
    """Dictionary mapping morphology names to validated Skeleton objects."""
    return {
        "human": human_skeleton,
        "quadruped": quadruped_skeleton,
        "serpentine": serpentine_skeleton,
        "avian": avian_skeleton,
        "tentacular": tentacular_skeleton,
    }


# =============================================================================
# 4. Synthetic Mesh Fixtures
# =============================================================================

@pytest.fixture
def cylinder_twist_mesh() -> CylinderMesh:
    """Watertight articulated twist cylinder mesh (radius=0.2, height=1.0, 20 rings)."""
    return create_cylinder_twist_mesh(radius=0.2, height=1.0, rings=20, radial_segments=16)


@pytest.fixture
def capsule_mesh() -> CapsuleMesh:
    """Watertight articulated capsule mesh (radius=0.15, length=0.6)."""
    return create_capsule_mesh(radius=0.15, length=0.6, rings=8, radial_segments=12)


@pytest.fixture
def mesh_volume_fn() -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns the divergence theorem volume calculation function."""
    return compute_mesh_volume


@pytest.fixture
def mesh_boundary_fn() -> Callable[[np.ndarray], np.ndarray]:
    """Returns the mesh boundary edge detector function."""
    return compute_mesh_boundary


# =============================================================================
# 5. CLI & Benchmark Runner Helpers
# =============================================================================

class CliResult:
    """Encapsulates subprocess CLI execution results."""

    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    @property
    def success(self) -> bool:
        return self.exit_code == 0


@pytest.fixture
def cli_runner() -> Callable[..., CliResult]:
    """
    Fixture providing an opaque-box CLI invocation runner for `main.py`.
    Executes in a separate process to faithfully evaluate CLI argument parsing,
    exit codes, and stdout/stderr emission without interfering with pytest.
    """
    def _run(*args: str, timeout: float = 60.0) -> CliResult:
        cmd = [sys.executable, "main.py"] + list(args)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CliResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    return _run


@pytest.fixture
def benchmark_runner() -> Callable[..., BenchmarkReport]:
    """
    Fixture providing direct programmatic execution of `scripts/benchmark_reconstruction.py`.
    Evaluates glTF 2.0 and BVH motion capture assets against strict specification rules.
    """
    def _run(
        gltf_path: Union[str, Path],
        bvh_path: Union[str, Path],
        tolerance: float = 1e-3,
        max_jump: float = 5.0,
    ) -> BenchmarkReport:
        return benchmark_reconstruction(
            Path(gltf_path),
            Path(bvh_path),
            tolerance=tolerance,
            max_jump_threshold=max_jump,
        )

    return _run
