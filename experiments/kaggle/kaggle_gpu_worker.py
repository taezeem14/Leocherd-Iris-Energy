#!/usr/bin/env python3
"""
Leocherd Iris Energy — Standalone GPU Pose Extraction Worker.
Designed to run on Kaggle GPU / Google Colab / remote Linux servers with CUDA.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import savgol_filter

try:
    import mediapipe as mp
except ImportError:
    print("Error: mediapipe not installed. Run: pip install mediapipe opencv-python-headless scipy")
    sys.exit(1)

LANDMARK_MAP = {
    0: "Nose",
    7: "LeftEar",
    8: "RightEar",
    11: "LeftShoulder",
    12: "RightShoulder",
    13: "LeftArm",
    14: "RightArm",
    15: "LeftForeArm",
    16: "RightForeArm",
    17: "LeftHand",
    18: "RightHand",
    23: "LeftUpLeg",
    24: "RightUpLeg",
    25: "LeftLeg",
    26: "RightLeg",
    27: "LeftFoot",
    28: "RightFoot",
}


def process_video(video_path: str, max_frames: int | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Processing {video_path}: {total_frames} frames @ {fps:.2f} FPS")

    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    raw_frames = []
    frame_idx = 0

    while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)

        frame_joints = {}
        if results.pose_world_landmarks:
            wl = results.pose_world_landmarks.landmark
            l_hip, r_hip = wl[23], wl[24]
            l_sh, r_sh = wl[11], wl[12]

            hips_x = (l_hip.x + r_hip.x) * 0.5
            hips_y = -(l_hip.y + r_hip.y) * 0.5
            hips_z = (l_hip.z + r_hip.z) * 0.5
            hips_x = (l_hip.x + r_hip.x) * 0.5
            hips_y = -(l_hip.y + r_hip.y) * 0.5 + 1.0
            hips_z = -(l_hip.z + r_hip.z) * 0.5
            frame_joints["Hips"] = {"x": float(hips_x), "y": float(hips_y), "z": float(hips_z), "confidence": 1.0}

            chest_x = (l_sh.x + r_sh.x) * 0.5
            chest_y = -(l_sh.y + r_sh.y) * 0.5 + 1.0
            chest_z = -(l_sh.z + r_sh.z) * 0.5
            frame_joints["Chest"] = {"x": float(chest_x), "y": float(chest_y), "z": float(chest_z), "confidence": 1.0}

            frame_joints["Spine"] = {
                "x": float((hips_x + chest_x) * 0.5),
                "y": float((hips_y + chest_y) * 0.5),
                "z": float((hips_z + chest_z) * 0.5),
                "confidence": 1.0,
            }

            nose = wl[0]
            l_ear, r_ear = wl[7], wl[8]

            # Head Center (Ear midpoint or Nose offset)
            if (l_ear.visibility or 0) > 0.4 and (r_ear.visibility or 0) > 0.4:
                head_x = (l_ear.x + r_ear.x) * 0.5
                head_y = -(l_ear.y + r_ear.y) * 0.5 + 1.0
                head_z = -(l_ear.z + r_ear.z) * 0.5
            else:
                head_x = nose.x
                head_y = -nose.y + 1.0
                head_z = -nose.z

            frame_joints["Head"] = {
                "x": float(head_x),
                "y": float(head_y),
                "z": float(head_z),
                "confidence": float(nose.visibility or 1.0),
            }

            # Neck (Dynamic anatomical cervical bridge smoothly following Head and Spine)
            frame_joints["Neck"] = {
                "x": float(chest_x * 0.50 + head_x * 0.50),
                "y": float(chest_y * 0.50 + head_y * 0.50),
                "z": float(chest_z * 0.50 + head_z * 0.50),
                "confidence": 1.0,
            }

            # Nose (Facial gaze landmark)
            frame_joints["Nose"] = {
                "x": float(nose.x),
                "y": float(-nose.y + 1.0),
                "z": float(-nose.z),
                "confidence": float(nose.visibility or 1.0),
            }

            for lm_idx, j_name in LANDMARK_MAP.items():
                lm = wl[lm_idx]
                frame_joints[j_name] = {
                    "x": float(lm.x),
                    "y": float(-lm.y + 1.0),
                    "z": float(-lm.z),
                    "confidence": float(lm.visibility),
                    "visibility": float(lm.visibility),
                }

        raw_frames.append({
            "frame_index": frame_idx,
            "timestamp_sec": float(frame_idx) / float(fps),
            "joints": frame_joints,
        })
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"   [+] Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    pose_model.close()
    return raw_frames, fps


def smooth_trajectories(frames, window_len=7, polyorder=2):
    if len(frames) < window_len:
        return frames

    all_joint_names = set()
    for f in frames:
        all_joint_names.update(f["joints"].keys())

    T = len(frames)
    for j_name in all_joint_names:
        traj = np.zeros((T, 3), dtype=np.float32)
        for t, f in enumerate(frames):
            j = f["joints"].get(j_name, {"x": 0.0, "y": 0.0, "z": 0.0})
            traj[t, 0] = j["x"]
            traj[t, 1] = j["y"]
            traj[t, 2] = j["z"]

        for d in range(3):
            traj[:, d] = savgol_filter(traj[:, d], window_length=window_len, polyorder=polyorder)

        for t, f in enumerate(frames):
            if j_name in f["joints"]:
                f["joints"][j_name]["x"] = float(traj[t, 0])
                f["joints"][j_name]["y"] = float(traj[t, 1])
                f["joints"][j_name]["z"] = float(traj[t, 2])

    return frames


def main():
    parser = argparse.ArgumentParser(description="Leocherd Iris Energy GPU Pose Worker")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", default="iris_kaggle_pose_data.json", help="Output JSON path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--window", type=int, default=7, help="Smoothing window length")
    args = parser.parse_args()

    raw, fps = process_video(args.video, max_frames=args.max_frames)
    smoothed = smooth_trajectories(raw, window_len=args.window)

    out_data = {
        "generator": "Leocherd Iris Energy Kaggle GPU Worker 0.2.0",
        "morphology": "human",
        "fps": float(fps),
        "total_frames": len(smoothed),
        "frames": smoothed,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"[OK] Successfully exported GPU pose data to: {args.output}")


if __name__ == "__main__":
    main()
