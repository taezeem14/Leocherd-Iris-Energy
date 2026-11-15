"""
Biovision Hierarchy (.bvh) skeletal animation exporter.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from iris.core.logging import get_logger
from iris.export.base import BaseAssetExporter
from iris.retargeting.animation import SkeletalAnimation
from iris.rigging.skeleton import Skeleton

logger = get_logger("iris.export.bvh")


def quat_to_euler_zxy_deg(q: np.ndarray) -> tuple[float, float, float]:
    """
    Converts unit quaternion [w, x, y, z] to ZXY Euler angles in degrees.
    Handles gimbal lock singularities at pitch ≈ ±90° robustly without losing roll angle.
    """
    w, x, y, z = q
    # Normalize
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 1e-6:
        w, x, y, z = w / n, x / n, y / n, z / n

    # Matrix elements
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    r20 = 2.0 * (x * z - w * y)
    r01 = 2.0 * (x * y - w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)

    # Singularity check (gimbal lock pitch ≈ ±90°)
    if abs(r21) > 0.99999:
        rot_x = math.copysign(90.0, r21)
        rot_z = math.degrees(math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z)))
        rot_y = 0.0
    else:
        rot_x = math.degrees(math.asin(max(-1.0, min(1.0, r21))))
        rot_y = math.degrees(math.atan2(-r20, r22))
        rot_z = math.degrees(math.atan2(-r01, r11))

    return round(rot_z, 3), round(rot_x, 3), round(rot_y, 3)


def quat_sequence_to_euler_zxy_deg(quats: np.ndarray) -> np.ndarray:
    """
    Converts a temporal sequence of quaternions (T, 4) in [w, x, y, z] order to continuous
    ZXY Euler angles (T, 3) in degrees, applying:
    1. Antipodal quaternion flip neutralization (q_t . q_{t-1} < 0 => -q_t).
    2. Robust gimbal lock singularity recovery when pitch ≈ ±90°.
    3. Continuous temporal phase unwrapping ((curr - prev + 180) % 360 - 180) to eliminate 360° discontinuities.
    """
    quats = np.asarray(quats, dtype=np.float32)
    if quats.ndim == 1:
        quats = quats[np.newaxis, :]
    T = len(quats)
    if T == 0:
        return np.zeros((0, 3), dtype=np.float32)

    eulers = np.zeros((T, 3), dtype=np.float32)

    q_curr = quats[0].copy()
    z0, x0, y0 = quat_to_euler_zxy_deg(q_curr)
    eulers[0] = [z0, x0, y0]
    prev_q = q_curr

    for t in range(1, T):
        q_t = quats[t].copy()
        # Antipodal check: ensure smooth path on S^3
        if np.dot(q_t, prev_q) < 0.0:
            q_t = -q_t
        prev_q = q_t

        raw_z, raw_x, raw_y = quat_to_euler_zxy_deg(q_t)
        prev_z, prev_x, prev_y = eulers[t - 1]

        # Phase unwrap each angle relative to preceding frame
        diff_z = (raw_z - prev_z + 180.0) % 360.0 - 180.0
        diff_x = (raw_x - prev_x + 180.0) % 360.0 - 180.0
        diff_y = (raw_y - prev_y + 180.0) % 360.0 - 180.0

        eulers[t, 0] = prev_z + diff_z
        eulers[t, 1] = prev_x + diff_x
        eulers[t, 2] = prev_y + diff_y

    return eulers


class BVHExporter(BaseAssetExporter):
    """Exports skeleton hierarchy and motion curves into standard .bvh format."""

    def __init__(self) -> None:
        super().__init__(name="BVHExporter", extension=".bvh")

    def export(
        self,
        skeleton: Skeleton,
        animation: SkeletalAnimation,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        out_file = Path(output_path).with_suffix(self.extension)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting BVH motion capture asset to {out_file}...")

        lines: List[str] = ["HIERARCHY"]
        channel_order: List[str] = []

        # 1. Recursive hierarchy formatting
        def format_joint(name: str, indent_level: int) -> None:
            indent = "  " * indent_level
            node = skeleton.joints[name]
            channel_order.append(name)

            if node.parent_name is None:
                lines.append(f"{indent}ROOT {name}")
                lines.append(f"{indent}{{")
                lines.append(f"{indent}  OFFSET {node.rest_world_pos[0]:.4f} {node.rest_world_pos[1]:.4f} {node.rest_world_pos[2]:.4f}")
                lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
            else:
                parent_node = skeleton.joints[node.parent_name]
                offset = node.rest_world_pos - parent_node.rest_world_pos
                lines.append(f"{indent}JOINT {name}")
                lines.append(f"{indent}{{")
                lines.append(
                    f"{indent}  OFFSET {offset[0]:.4f} {offset[1]:.4f} {offset[2]:.4f}"
                )
                lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")

            if node.children_names:
                for child in node.children_names:
                    format_joint(child, indent_level + 1)
            else:
                # End site
                lines.append(f"{indent}  End Site")
                lines.append(f"{indent}  {{")
                lines.append(f"{indent}    OFFSET 0.0 0.1 0.0")
                lines.append(f"{indent}  }}")

            lines.append(f"{indent}}}")

        format_joint(skeleton.root_name, 0)

        # 2. Motion data section
        T = animation.frame_count
        frame_time = 1.0 / max(1.0, animation.fps)

        lines.append("MOTION")
        lines.append(f"Frames: {T}")
        lines.append(f"Frame Time: {frame_time:.6f}")

        # Precompute continuous Euler trajectories for all joints
        joint_eulers: Dict[str, np.ndarray] = {}
        for j_name in channel_order:
            if j_name in animation.rotations:
                q_seq = animation.rotations[j_name]
                if len(q_seq) > 0 and q_seq.ndim == 1:
                    q_seq = np.tile(q_seq, (T, 1))
            else:
                q_seq = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
            joint_eulers[j_name] = quat_sequence_to_euler_zxy_deg(q_seq)

        for t in range(T):
            frame_vals: List[str] = []
            for j_name in channel_order:
                node = skeleton.joints[j_name]
                if node.parent_name is None:
                    # Root has 3 positions + 3 rotations
                    tx, ty, tz = animation.root_translations[t]
                    frame_vals.extend([f"{tx:.4f}", f"{ty:.4f}", f"{tz:.4f}"])

                rz, rx, ry = joint_eulers[j_name][t]
                frame_vals.extend([f"{rz:.2f}", f"{rx:.2f}", f"{ry:.2f}"])

            lines.append(" ".join(frame_vals))

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Successfully generated BVH asset ({T} frames, {len(channel_order)} articulated channels)")
        return out_file
