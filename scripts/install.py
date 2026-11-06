#!/usr/bin/env python3
"""
Leocherd Iris Energy - Environment Setup and Dependency Installer
Installs required packages safely with local cache management on Drive D.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "pip"
VENV_DIR = PROJECT_ROOT / ".venv"


def get_venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_venv_pip() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def create_venv() -> None:
    if not VENV_DIR.exists():
        print(f"[*] Creating Python virtual environment in {VENV_DIR}...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("[+] Virtual environment created successfully.")
    else:
        print(f"[i] Existing virtual environment found at {VENV_DIR}.")


def install_packages() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pip_bin = str(get_venv_pip())
    py_bin = str(get_venv_python())

    print(f"[*] Upgrading pip and setuptools (Cache: {CACHE_DIR})...")
    subprocess.check_call(
        [py_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--cache-dir", str(CACHE_DIR)]
    )

    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        print(f"[*] Installing dependencies from {req_file}...")
        subprocess.check_call(
            [py_bin, "-m", "pip", "install", "-r", str(req_file), "--cache-dir", str(CACHE_DIR)]
        )
    else:
        print("[!] requirements.txt not found, skipping batch install.")


def verify_installation() -> None:
    py_bin = str(get_venv_python())
    check_script = """
import sys
modules = ['numpy', 'cv2', 'scipy', 'onnxruntime', 'fastapi', 'pydantic', 'pytest', 'httpx']
missing = []
for m in modules:
    try:
        __import__(m)
        print(f"  [OK] {m}")
    except ImportError as e:
        print(f"  [MISSING] {m}: {e}")
        missing.append(m)
if missing:
    sys.exit(1)
print("All core dependencies verified successfully.")
"""
    print("[*] Verifying critical modules in virtual environment...")
    result = subprocess.run([py_bin, "-c", check_script])
    if result.returncode == 0:
        print("[+] Environment installation verified!")
    else:
        print("[!] Some dependencies could not be verified.")


def main() -> None:
    print("=" * 60)
    print("  Leocherd Iris Energy - Dependency Setup (Drive D Safe)")
    print("=" * 60)
    create_venv()
    install_packages()
    verify_installation()


if __name__ == "__main__":
    main()
