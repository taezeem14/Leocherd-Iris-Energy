#!/usr/bin/env python3
"""
Generates the 8 standard Kaggle experimentation notebooks for Leocherd Iris Energy.
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "kaggle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


# Notebook 0: Environment Diagnostics
nb0_cells = [
    md_cell("# Leocherd Iris Energy - Kaggle Experiment 00: Environment Diagnostics\nEvaluates GPU, CPU, RAM, disk space, and package availability in the Kaggle session."),
    code_cell("""import os
import sys
import platform
import shutil

print(f"Python Version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"CPU Cores: {os.cpu_count()}")

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
except ImportError:
    print("PyTorch not installed.")

usage = shutil.disk_usage('/')
print(f"Disk Free: {usage.free / (1024**3):.2f} GB / {usage.total / (1024**3):.2f} GB")
"""),
]

# Notebook 1: Video Preprocessing
nb1_cells = [
    md_cell("# Leocherd Iris Energy - Kaggle Experiment 01: Video Preprocessing\nMemory-efficient chunked frame extraction, aspect preservation, and downsampling."),
    code_cell("""!pip install -q opencv-python-headless numpy pillow"""),
    code_cell("""import cv2
import numpy as np

def inspect_and_extract_sample(video_path, max_frames=60, target_fps=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
    
    step = max(1.0, fps / target_fps)
    frames = []
    idx = 0.0
    while len(frames) < max_frames and int(idx) < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        idx += step
    cap.release()
    print(f"Extracted {len(frames)} sample frames at {target_fps} FPS.")
    return frames
"""),
]

# Notebook 2: Human Pose Estimation
nb2_cells = [
    md_cell("# Leocherd Iris Energy - Kaggle Experiment 02: Human Pose Estimation\nEvaluates 2D landmark inference with MediaPipe / ONNX on GPU vs CPU."),
    code_cell("""!pip install -q onnxruntime-gpu opencv-python-headless requests"""),
    code_cell("""import onnxruntime as ort
print("ONNX Runtime Available Providers:", ort.get_available_providers())
"""),
    code_cell("""# Download MediaPipe 3D ONNX model checkpoint if not present
import urllib.request
url = 'https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx'
model_file = 'pose_estimation_mediapipe_2023mar.onnx'
urllib.request.urlretrieve(url, model_file)
print(f"Model downloaded to {model_file}")

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
sess = ort.InferenceSession(model_file, providers=providers)
print(f"InferenceSession initialized with: {sess.get_providers()}")
"""),
]

# Notebook 3: 3D Pose Experiments
nb3_cells = [
    md_cell("# Leocherd Iris Energy - Kaggle Experiment 03: Monocular 3D Pose Lifting\nEvaluates 3D metric coordinate reconstruction and kinematic regularizers."),
    code_cell("""import numpy as np

def regularize_bone_lengths(parent_pos, child_pos, rest_length):
    diff = child_pos - parent_pos
    norm = np.linalg.norm(diff)
    if norm < 1e-6:
        return parent_pos + np.array([0, -rest_length, 0], dtype=np.float32)
    direction = diff / norm
    return parent_pos + direction * rest_length

# Demonstration of bone regularization
p_hip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
p_knee_noisy = np.array([0.0, 0.45, 0.1], dtype=np.float32)
rest_femur = 0.45

p_knee_clean = regularize_bone_lengths(p_hip, p_knee_noisy, rest_femur)
print(f"Original length: {np.linalg.norm(p_knee_noisy - p_hip):.4f}")
print(f"Stabilized length: {np.linalg.norm(p_knee_clean - p_hip):.4f}")
"""),
]

# Notebook 4: Motion Reconstruction
nb4_cells = [
    md_cell("# Leocherd Iris Energy - Kaggle Experiment 04: Temporal Motion Smoothing\nEvaluates One Euro adaptive low-pass filter vs Savitzky-Golay on noisy trajectories."),
    code_cell("""import math
import numpy as np
import matplotlib.pyplot as plt

class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007):
        self.freq, self.min_cutoff, self.beta = freq, min_cutoff, beta
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        cutoff = self.min_cutoff + self.beta * abs(dx)
        alpha = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        hat_x = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = hat_x
        return hat_x

# Benchmark on noisy sine wave
times = np.linspace(0, 2.0, 60)
true_signal = np.sin(2 * np.pi * 1.5 * times)
noisy_signal = true_signal + np.random.normal(0, 0.15, size=len(times))

f = OneEuroFilter()
smoothed = [f.filter(val, t) for val, t in zip(noisy_signal, times)]
print(f"MSE Noisy: {np.mean((noisy_signal - true_signal)**2):.4f}")
print(f"MSE Smoothed: {np.mean((np.array(smoothed) - true_signal)**2):.4f}")
"""),
]

# Notebook 5: Skeleton Generation
nb5_cells = [
    md_cell("# Leocherd Iris Energy - Kaggle Experiment 05: Skeleton Hierarchy & Quaternions\nComputes bone orientation unit quaternions and local hierarchical transforms."),
    code_cell("""import numpy as np

def quat_from_vectors(v_from, v_to):
    u1 = v_from / np.linalg.norm(v_from)
    u2 = v_to / np.linalg.norm(v_to)
    dot = np.dot(u1, u2)
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(u1, u2)
    w = math.sqrt((1.0 + dot) * 2.0) * 0.5
    inv_w4 = 1.0 / (4.0 * w)
    return np.array([w, axis[0]*inv_w4, axis[1]*inv_w4, axis[2]*inv_w4], dtype=np.float32)

import math
q = quat_from_vectors(np.array([0, 1, 0]), np.array([1, 0, 0]))
print("Computed Rotation Quaternion [w, x, y, z]:", q)
"""),
]

# Notebook 6: Model Benchmarking
nb6_cells = [
    md_cell("# Leocherd Iris Energy - Kaggle Experiment 06: Model Benchmarking Suite\nMeasures FPS, memory footprint, and latency across batch sizes and resolutions."),
    code_cell("""import time
import numpy as np

def benchmark_inference(session, input_name, shape=(1, 256, 256, 3), iterations=100):
    dummy = np.random.rand(*shape).astype(np.float32)
    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy})
    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start
    fps = iterations / elapsed
    ms = (elapsed / iterations) * 1000.0
    print(f"Latency: {ms:.2f} ms | Throughput: {fps:.1f} FPS")
    return {"latency_ms": ms, "fps": fps}
"""),
]

# Notebook 7: Artifact Export
nb7_cells = [
    md_cell("# Leocherd Iris Energy - Kaggle Experiment 07: Artifact Export & Bundling\nPackages reconstructed animations into glTF 2.0 and BVH files for local download."),
    code_cell("""import json
import base64
print("Ready to export experiment results to /kaggle/working/outputs.")
"""),
]

notebooks = {
    "00_environment_diagnostics.ipynb": nb0_cells,
    "01_video_preprocessing.ipynb": nb1_cells,
    "02_human_pose_estimation.ipynb": nb2_cells,
    "03_3d_pose_experiments.ipynb": nb3_cells,
    "04_motion_reconstruction.ipynb": nb4_cells,
    "05_skeleton_generation.ipynb": nb5_cells,
    "06_model_benchmarking.ipynb": nb6_cells,
    "07_artifact_export.ipynb": nb7_cells,
}

for name, cells in notebooks.items():
    nb_path = OUT_DIR / name
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"[+] Created notebook: {nb_path.name}")
