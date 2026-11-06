"""
Temporal bounding box tracking using IoU and smooth motion persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from iris.vision.detection import BoundingBox


@dataclass
class TrackedSubject:
    subject_id: int
    bbox: BoundingBox
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    history: List[BoundingBox] = field(default_factory=list)


class IoUTracker:
    """Lightweight, reliable multi-subject tracker for video sequences."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.tracks: Dict[int, TrackedSubject] = {}
        self._next_id: int = 1

    def update(self, detections: List[BoundingBox]) -> List[Tuple[int, BoundingBox]]:
        """
        Updates internal tracks with new frame detections.
        Returns active matched (subject_id, bbox) pairs.
        """
        # Increment age and time since update for existing tracks
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1

        unmatched_dets = set(range(len(detections)))
        matched_pairs: List[Tuple[int, int]] = []

        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            # Compute IoU matrix
            for t_idx, t_id in enumerate(track_ids):
                best_iou = 0.0
                best_d_idx = -1
                for d_idx in unmatched_dets:
                    iou = self.tracks[t_id].bbox.iou(detections[d_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_d_idx = d_idx

                if best_iou >= self.iou_threshold and best_d_idx != -1:
                    matched_pairs.append((t_id, best_d_idx))
                    unmatched_dets.discard(best_d_idx)

        # Update matched tracks
        for t_id, d_idx in matched_pairs:
            track = self.tracks[t_id]
            track.bbox = detections[d_idx]
            track.history.append(detections[d_idx])
            track.hits += 1
            track.time_since_update = 0

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            new_id = self._next_id
            self._next_id += 1
            self.tracks[new_id] = TrackedSubject(
                subject_id=new_id,
                bbox=detections[d_idx],
                history=[detections[d_idx]],
            )

        # Remove dead tracks
        dead_ids = [
            t_id for t_id, t in self.tracks.items() if t.time_since_update > self.max_lost_frames
        ]
        for d_id in dead_ids:
            del self.tracks[d_id]

        # Return active tracks
        results = [(t.subject_id, t.bbox) for t in self.tracks.values() if t.time_since_update == 0]
        # Sort so primary track (highest hits) is first
        results.sort(key=lambda item: self.tracks[item[0]].hits, reverse=True)
        return results
