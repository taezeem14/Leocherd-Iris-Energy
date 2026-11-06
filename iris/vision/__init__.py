"""
Computer vision module for detection, tracking, and visualization.
"""

from iris.vision.detection import BoundingBox, SubjectDetector
from iris.vision.tracking import IoUTracker, TrackedSubject
from iris.vision.visualization import Visualizer

__all__ = ["BoundingBox", "SubjectDetector", "IoUTracker", "TrackedSubject", "Visualizer"]
