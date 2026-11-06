#!/usr/bin/env python3
"""
Generates a real, valid MP4 video containing a moving human figure for end-to-end integration testing.
"""

import math
from pathlib import Path
import cv2
import numpy as np

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_walk.mp4"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def generate_video(
    output_path=OUT_PATH,
    width=640,
    height=480,
    fps=15,
    duration_sec=2.0,
):
    total_frames = int(round(fps * duration_sec))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path: {output_path}")

    print(f"[*] Generating {total_frames} frames ({width}x{height} @ {fps} FPS) -> {output_path}...")

    cx = width // 2
    ground_y = int(height * 0.85)

    for i in range(total_frames):
        # Create gradient studio background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Ground plane
        cv2.rectangle(frame, (0, ground_y), (width, height), (30, 30, 30), -1)

        t = (i / total_frames) * 2.0 * math.pi
        walk_offset_x = int(math.sin(t) * 40)
        bob_y = int(abs(math.sin(t * 2)) * 8)

        # Center of human
        hx = cx + walk_offset_x
        hips_y = ground_y - 120 - bob_y
        chest_y = hips_y - 60
        head_y = chest_y - 45

        # Draw Head
        cv2.circle(frame, (hx, head_y), 18, (220, 200, 180), -1)

        # Draw Torso
        cv2.line(frame, (hx, head_y + 18), (hx, hips_y), (180, 140, 100), 12)

        # Draw Arms
        arm_swing = math.sin(t) * 35
        # Left Arm
        cv2.line(frame, (hx - 15, chest_y), (int(hx - 35 - arm_swing), chest_y + 40), (160, 120, 80), 8)
        cv2.line(frame, (int(hx - 35 - arm_swing), chest_y + 40), (int(hx - 45 - arm_swing * 1.3), chest_y + 80), (220, 200, 180), 6)
        # Right Arm
        cv2.line(frame, (hx + 15, chest_y), (int(hx + 35 + arm_swing), chest_y + 40), (160, 120, 80), 8)
        cv2.line(frame, (int(hx + 35 + arm_swing), chest_y + 40), (int(hx + 45 + arm_swing * 1.3), chest_y + 80), (220, 200, 180), 6)

        # Draw Legs
        leg_stride = math.sin(t) * 40
        # Left Leg
        cv2.line(frame, (hx - 10, hips_y), (int(hx - 15 - leg_stride), hips_y + 60), (80, 80, 150), 9)
        cv2.line(frame, (int(hx - 15 - leg_stride), hips_y + 60), (int(hx - 20 - leg_stride * 1.2), ground_y), (50, 50, 50), 7)
        # Right Leg
        cv2.line(frame, (hx + 10, hips_y), (int(hx + 15 + leg_stride), hips_y + 60), (80, 80, 150), 9)
        cv2.line(frame, (int(hx + 15 + leg_stride), hips_y + 60), (int(hx + 20 + leg_stride * 1.2), ground_y), (50, 50, 50), 7)

        out.write(frame)

    out.release()
    print(f"[+] Sample video successfully created: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_video()
