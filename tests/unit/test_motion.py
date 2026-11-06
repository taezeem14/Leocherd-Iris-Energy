"""
Unit tests for temporal motion trajectories, derivatives, and One Euro filtering.
"""

import math
import numpy as np
import pytest
from iris.motion.interpolation import TrajectoryInterpolator
from iris.motion.representation import JointTrajectory
from iris.motion.smoothing import OneEuroFilter, TemporalSmoother


def test_trajectory_derivatives():
    # Linear movement along X: x(t) = 2.0 * t -> v = 2.0, a = 0.0
    times = np.linspace(0, 1.0, 11)
    pos = np.zeros((11, 3), dtype=np.float32)
    pos[:, 0] = 2.0 * times
    confs = np.ones(11, dtype=np.float32)

    traj = JointTrajectory(name="hand", timestamps=times, positions=pos, confidences=confs)
    traj.compute_derivatives()

    assert traj.velocities is not None
    # Check velocity along X is approximately 2.0
    assert np.allclose(traj.velocities[1:-1, 0], 2.0, atol=0.05)


def test_one_euro_filter_noise_reduction():
    # Stationary signal corrupted with Gaussian noise
    np.random.seed(42)
    times = np.linspace(0, 2.0, 60)
    true_x = np.full_like(times, 2.5)
    noise = np.random.normal(0, 0.2, size=len(times))
    noisy = true_x + noise

    filt = OneEuroFilter(freq=30.0, min_cutoff=0.5, beta=0.001)
    filtered = np.array([filt.filter(np.array([val]), t)[0] for val, t in zip(noisy, times)])

    # Discard warmup frames (first 5)
    var_noisy = np.var(noisy[5:] - true_x[5:])
    var_filtered = np.var(filtered[5:] - true_x[5:])
    assert var_filtered < var_noisy


def test_trajectory_interpolator():
    times = np.linspace(0, 1.0, 5)
    pos = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [3, 3, 3], [4, 4, 4]], dtype=np.float32)
    confs = np.array([1.0, 1.0, 0.1, 1.0, 1.0], dtype=np.float32)  # Index 2 is occluded/invalid

    imputed = TrajectoryInterpolator.impute_missing(pos, confs, times, min_conf=0.35)
    # Index 2 should be linearly interpolated between (1,1,1) and (3,3,3) -> (2,2,2)
    assert np.allclose(imputed[2], [2.0, 2.0, 2.0], atol=1e-3)
