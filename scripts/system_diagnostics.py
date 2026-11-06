#!/usr/bin/env python3
"""
Leocherd Iris Energy - System Diagnostics Script
Evaluates host environment, CPU/GPU, Python runtime, storage, and external tools
to determine the optimal execution profile.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StorageInfo:
    drive: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float


@dataclass
class HardwareProfile:
    os_name: str
    os_release: str
    os_arch: str
    python_version: str
    python_executable: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    has_avx: Optional[bool] = None
    gpu_devices: List[str] = field(default_factory=list)
    has_cuda: bool = False
    cuda_version: Optional[str] = None
    storage: List[StorageInfo] = field(default_factory=list)
    has_pytorch: bool = False
    pytorch_version: Optional[str] = None
    has_opencv: bool = False
    opencv_version: Optional[str] = None
    has_onnxruntime: bool = False
    onnxruntime_version: Optional[str] = None
    has_blender: bool = False
    blender_path: Optional[str] = None
    blender_version: Optional[str] = None
    has_ffmpeg: bool = False
    ffmpeg_path: Optional[str] = None
    has_git: bool = False
    has_node: bool = False
    recommended_profile: str = "CPU_LIGHTWEIGHT"
    warnings: List[str] = field(default_factory=list)


def check_storage() -> List[StorageInfo]:
    storage_list = []
    if platform.system() == "Windows":
        import ctypes

        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if bitmask & 1:
                drive_path = f"{letter}:\\"
                try:
                    usage = shutil.disk_usage(drive_path)
                    storage_list.append(
                        StorageInfo(
                            drive=f"{letter}:",
                            total_gb=round(usage.total / (1024**3), 2),
                            used_gb=round(usage.used / (1024**3), 2),
                            free_gb=round(usage.free / (1024**3), 2),
                            percent_used=round((usage.used / usage.total) * 100, 1),
                        )
                    )
                except Exception:
                    pass
            bitmask >>= 1
    else:
        usage = shutil.disk_usage("/")
        storage_list.append(
            StorageInfo(
                drive="/",
                total_gb=round(usage.total / (1024**3), 2),
                used_gb=round(usage.used / (1024**3), 2),
                free_gb=round(usage.free / (1024**3), 2),
                percent_used=round((usage.used / usage.total) * 100, 1),
            )
        )
    return storage_list


def get_cpu_info() -> tuple[str, int, int]:
    cpu_name = platform.processor() or "Unknown CPU"
    cores_logical = os.cpu_count() or 1
    cores_physical = cores_logical
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "cpu", "get", "name,NumberOfCores,NumberOfLogicalProcessors"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
            if len(lines) >= 2:
                parts = lines[1].split()
                # wmic output layout varies, fallback to powershell if needed
                cpu_name = " ".join(parts[:-2]) if len(parts) >= 3 else cpu_name
        except Exception:
            pass
    return cpu_name, cores_physical, cores_logical


def get_ram_info() -> tuple[float, float]:
    total_gb = 0.0
    avail_gb = 0.0
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_gb = round(mem.total / (1024**3), 2)
        avail_gb = round(mem.available / (1024**3), 2)
        return total_gb, avail_gb
    except ImportError:
        pass

    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = round(stat.ullTotalPhys / (1024**3), 2)
            avail_gb = round(stat.ullAvailPhys / (1024**3), 2)
        except Exception:
            pass
    return total_gb, avail_gb


def find_blender() -> tuple[bool, Optional[str], Optional[str]]:
    blender_bin = shutil.which("blender")
    common_windows_paths = [
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        r"D:\Program Files\Blender Foundation\Blender\blender.exe",
        r"D:\Blender\blender.exe",
    ]
    if not blender_bin and platform.system() == "Windows":
        for p in common_windows_paths:
            if os.path.exists(p):
                blender_bin = p
                break

    if blender_bin:
        try:
            out = subprocess.check_output([blender_bin, "--version"], text=True, stderr=subprocess.DEVNULL)
            first_line = out.strip().splitlines()[0] if out else "Blender (version unknown)"
            return True, blender_bin, first_line
        except Exception:
            return True, blender_bin, "Blender (installed)"
    return False, None, None


def find_ffmpeg() -> tuple[bool, Optional[str]]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin and platform.system() == "Windows":
        known_paths = [
            r"C:\Program Files\Softdeluxe\Free Download Manager\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"D:\ffmpeg\bin\ffmpeg.exe",
        ]
        for kp in known_paths:
            if os.path.exists(kp):
                ffmpeg_bin = kp
                break
    return (True, ffmpeg_bin) if ffmpeg_bin else (False, None)


def inspect_system() -> HardwareProfile:
    warnings = []
    os_name = platform.system()
    os_release = platform.release()
    os_arch = platform.machine()
    py_ver = platform.python_version()
    py_exec = sys.executable

    cpu_name, cpu_physical, cpu_logical = get_cpu_info()
    ram_total, ram_avail = get_ram_info()
    storage = check_storage()

    # GPU & CUDA checks
    has_cuda = False
    cuda_ver = None
    gpu_devices = []

    # Check PyTorch
    has_pytorch = False
    pytorch_version = None
    try:
        import torch

        has_pytorch = True
        pytorch_version = torch.__version__
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            cuda_ver = torch.version.cuda
            gpu_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass

    # Check OpenCV
    has_opencv = False
    opencv_version = None
    try:
        import cv2

        has_opencv = True
        opencv_version = cv2.__version__
    except ImportError:
        pass

    # Check ONNX Runtime
    has_onnxruntime = False
    onnxruntime_version = None
    try:
        import onnxruntime

        has_onnxruntime = True
        onnxruntime_version = onnxruntime.__version__
    except ImportError:
        pass

    # Tools
    has_blender, blender_path, blender_version = find_blender()
    has_ffmpeg, ffmpeg_path = find_ffmpeg()
    has_git = shutil.which("git") is not None
    has_node = shutil.which("node") is not None

    # Windows Video Controller inspection if torch doesn't find CUDA
    if not gpu_devices and platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in output.strip().splitlines():
                if line.strip():
                    gpu_devices.append(line.strip())
        except Exception:
            pass

    # Check disk warnings
    for st in storage:
        if st.drive.upper().startswith("C") and st.free_gb < 5.0:
            warnings.append(
                f"Drive C: has low disk space ({st.free_gb:.1f} GB free). "
                "Ensure models, package caches, and output assets are stored on Drive D:."
            )

    # Determine recommended execution profile
    if has_cuda and gpu_devices:
        recommended_profile = "CUDA_GPU_HIGH_PRECISION"
    elif ram_avail >= 6.0 and cpu_logical >= 4:
        recommended_profile = "CPU_STANDARD_PRECISION"
    else:
        recommended_profile = "CPU_LIGHTWEIGHT"
        warnings.append(
            f"Host is operating in {recommended_profile} mode: "
            f"2-thread/limited RAM ({ram_total:.1f} GB total, {ram_avail:.1f} GB avail). "
            "Using ONNX Runtime CPU inference, downsampled frame rates, and chunked processing."
        )

    return HardwareProfile(
        os_name=os_name,
        os_release=os_release,
        os_arch=os_arch,
        python_version=py_ver,
        python_executable=py_exec,
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_physical,
        cpu_cores_logical=cpu_logical,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        gpu_devices=gpu_devices,
        has_cuda=has_cuda,
        cuda_version=cuda_ver,
        storage=storage,
        has_pytorch=has_pytorch,
        pytorch_version=pytorch_version,
        has_opencv=has_opencv,
        opencv_version=opencv_version,
        has_onnxruntime=has_onnxruntime,
        onnxruntime_version=onnxruntime_version,
        has_blender=has_blender,
        blender_path=blender_path,
        blender_version=blender_version,
        has_ffmpeg=has_ffmpeg,
        ffmpeg_path=ffmpeg_path,
        has_git=has_git,
        has_node=has_node,
        recommended_profile=recommended_profile,
        warnings=warnings,
    )


def print_report(profile: HardwareProfile) -> None:
    sep = "=" * 65
    print(sep)
    print(" LEOCHERD IRIS ENERGY - HARDWARE & ENVIRONMENT DIAGNOSTICS")
    print(sep)
    print(f" Operating System       : {profile.os_name} {profile.os_release} ({profile.os_arch})")
    print(f" Python Version         : {profile.python_version} ({profile.python_executable})")
    print(f" CPU Model              : {profile.cpu_name}")
    print(f" CPU Cores / Threads    : {profile.cpu_cores_physical} cores / {profile.cpu_cores_logical} threads")
    print(f" System RAM             : {profile.ram_total_gb:.1f} GB total ({profile.ram_available_gb:.1f} GB available)")
    print(f" GPU Devices            : {', '.join(profile.gpu_devices) if profile.gpu_devices else 'None detected'}")
    print(f" CUDA Acceleration      : {'AVAILABLE (' + str(profile.cuda_version) + ')' if profile.has_cuda else 'NOT AVAILABLE (CPU Mode Enforced)'}")
    print("-" * 65)
    print(" STORAGE STATUS:")
    for st in profile.storage:
        print(f"   Drive {st.drive:<5} : {st.free_gb:>6.1f} GB free / {st.total_gb:>6.1f} GB total ({st.percent_used:>5.1f}% used)")
    print("-" * 65)
    print(" RUNTIME DEPENDENCIES:")
    print(f"   PyTorch              : {profile.pytorch_version if profile.has_pytorch else 'Not Installed'}")
    print(f"   ONNX Runtime         : {profile.onnxruntime_version if profile.has_onnxruntime else 'Not Installed'}")
    print(f"   OpenCV               : {profile.opencv_version if profile.has_opencv else 'Not Installed'}")
    print(f"   Blender 3D           : {profile.blender_version or 'Not Detected (Standalone glTF/BVH output active)'}")
    if profile.blender_path:
        print(f"                          Path: {profile.blender_path}")
    print(f"   FFmpeg               : {'Available at ' + profile.ffmpeg_path if profile.has_ffmpeg else 'Not Detected'}")
    print(f"   Node.js              : {'Available' if profile.has_node else 'Not Installed'}")
    print(f"   Git                  : {'Available' if profile.has_git else 'Not Installed'}")
    print("-" * 65)
    print(f" RECOMMENDED PROFILE    : >>> {profile.recommended_profile} <<<")
    if profile.warnings:
        print(" ADVISORIES & WARNINGS:")
        for w in profile.warnings:
            print(f"   [!] {w}")
    print(sep)


def main() -> None:
    profile = inspect_system()
    if "--json" in sys.argv:
        print(json.dumps(asdict(profile), indent=2))
    else:
        print_report(profile)


if __name__ == "__main__":
    main()
