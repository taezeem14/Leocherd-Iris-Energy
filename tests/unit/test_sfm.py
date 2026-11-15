"""
Unit tests for Structure-from-Motion (SfM) feature matching and triangulation.
"""

import numpy as np
import pytest
from pathlib import Path

from iris.mesh.sfm_reconstruction import VideoStructureFromMotion


def test_sfm_feature_extractor():
    sfm = VideoStructureFromMotion(feature_detector="orb", max_features=100)

    # Synthetic image with high-contrast checkerboard
    img = np.zeros((120, 120), dtype=np.uint8)
    img[::20, :] = 255
    img[:, ::20] = 255

    kps, descs = sfm.extract_features(img)
    assert len(kps) > 0
    assert len(descs) == len(kps)

    # Match identical images
    matches = sfm.match_features(descs, descs)
    assert len(matches) > 0
