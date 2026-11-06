"""
Blender integration coordinator and background automation exporter.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict
import numpy as np

from iris.core.config import get_settings
from iris.core.logging import get_logger
from iris.export.base import BaseAssetExporter
from iris.retargeting.animation import SkeletalAnimation
from iris.rigging.skeleton import Skeleton

logger = get_logger("iris.export.blender")


class BlenderExporter(BaseAssetExporter):
    """Exports rig and animation data for Blender, and runs headless compilation if Blender is installed."""

    def __init__(self) -> None:
        super().__init__(name="BlenderExporter", extension=".blend")

    def export(
        self,
        skeleton: Skeleton,
        animation: SkeletalAnimation,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(output_path).stem

        rig_json_path = out_dir / f"{base_name}_rig_data.json"
        anim_json_path = out_dir / f"{base_name}_anim_data.json"
        target_blend = out_dir / f"{base_name}.blend"

        logger.info(f"Serializing Blender JSON definitions to {out_dir}...")

        # 1. Save rig JSON
        with open(rig_json_path, "w", encoding="utf-8") as f:
            json.dump(skeleton.to_dict(), f, indent=2)

        # 2. Save animation JSON
        anim_dict = {
            "name": str(animation.name),
            "fps": float(animation.fps),
            "timestamps": [float(x) for x in animation.timestamps],
            "root_joint_name": str(animation.root_joint_name),
            "root_translations": [[float(v) for v in row] for row in animation.root_translations],
            "rotations": {k: [[float(v) for v in row] for row in arr] for k, arr in animation.rotations.items()},
        }
        with open(anim_json_path, "w", encoding="utf-8") as f:
            json.dump(anim_dict, f, indent=2)

        settings = get_settings()
        blender_bin = settings.blender_executable

        if blender_bin and Path(blender_bin).exists():
            logger.info(f"Executing headless Blender build using: {blender_bin}")
            script_create = settings.project_root / "blender" / "create_armature.py"
            script_anim = settings.project_root / "blender" / "apply_animation.py"

            try:
                # Build armature
                subprocess.check_call(
                    [
                        str(blender_bin),
                        "--background",
                        "--python",
                        str(script_create),
                        "--",
                        "--input",
                        str(rig_json_path),
                        "--output",
                        str(target_blend),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Apply animation
                subprocess.check_call(
                    [
                        str(blender_bin),
                        "--background",
                        str(target_blend),
                        "--python",
                        str(script_anim),
                        "--",
                        "--animation",
                        str(anim_json_path),
                        "--output",
                        str(target_blend),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"Successfully compiled native Blender scene: {target_blend}")
                return target_blend
            except Exception as e:
                logger.warning(f"Headless Blender compilation failed ({e}). Package remains available via JSON.")
        else:
            logger.info("Blender binary not detected on host. Exported standalone JSON rig package for external import.")

        return rig_json_path
