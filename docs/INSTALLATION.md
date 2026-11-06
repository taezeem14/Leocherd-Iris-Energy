# Installation & Quickstart Guide

## Prerequisites
- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS (12+)
- **Python:** Python 3.10 to 3.13 (64-bit)
- **Disk Space:** ~500 MB for core dependencies (Drive D: recommended on partitioned Windows drives)

## Quick Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd D:\download-chrome\Leocherd-Project
   ```

2. **Run System Diagnostics:**
   ```bash
   python scripts/system_diagnostics.py
   ```
   This will inspect your CPU, RAM, GPU, storage, and tell you which profile will be used (`CPU_LIGHTWEIGHT` vs `CUDA_GPU_HIGH_PRECISION`).

3. **Install Dependencies (Safely configured for Drive D:):**
   ```bash
   python scripts/install.py
   ```

4. **Activate the Virtual Environment:**
   - Windows PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows Command Prompt:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```

5. **Run the Test Suite:**
   ```bash
   pytest tests/
   ```
