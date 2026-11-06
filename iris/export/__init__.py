"""
Multi-format 3D and skeletal asset export subsystem.
"""

from pathlib import Path
from typing import Dict, List, Optional

from iris.core.config import get_settings
from iris.core.events import EventBus, EventType, PipelineEvent
from iris.core.exceptions import ExportError
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, PipelineStage
from iris.export.base import BaseAssetExporter
from iris.export.blender import BlenderExporter
from iris.export.bvh import BVHExporter
from iris.export.gltf import GltfExporter

logger = get_logger("iris.export")


class AssetExportStage(PipelineStage):
    """Pipeline stage exporting reconstructed skeletons and animation curves to 3D asset formats."""

    def __init__(
        self,
        name: str = "AssetExportStage",
        formats: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name)
        self.formats = [f.lower() for f in (formats or ["gltf", "bvh", "blender"])]
        self.exporters: Dict[str, BaseAssetExporter] = {
            "gltf": GltfExporter(),
            "bvh": BVHExporter(),
            "blender": BlenderExporter(),
        }

    def run(self, context: ExecutionContext, event_bus: Optional[EventBus] = None) -> ExecutionContext:
        if not context.skeleton:
            raise ExportError("No skeleton available in ExecutionContext.")
        if not context.animation:
            raise ExportError("No animation available in ExecutionContext.")

        settings = get_settings()
        out_dir = Path(context.output_dir or (settings.outputs_dir / context.job_id))
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"reconstruction_{context.job_id}"

        logger.info(f"Exporting assets ({', '.join(self.formats)}) to {out_dir}...")

        exported: Dict[str, str] = {}

        for fmt in self.formats:
            exporter = self.exporters.get(fmt)
            if not exporter:
                logger.warning(f"Unrecognized export format '{fmt}', skipping.")
                continue

            target_path = out_dir / f"{base_name}{exporter.extension}"
            try:
                res_path = exporter.export(
                    skeleton=context.skeleton,
                    animation=context.animation,
                    output_path=target_path,
                )
                exported[fmt] = str(res_path)
                logger.info(f"Exported {fmt.upper()} to: {res_path}")

                if event_bus:
                    event_bus.emit(
                        PipelineEvent(
                            event_type=EventType.ASSET_EXPORTED,
                            stage_name=self.name,
                            message=f"Exported {fmt.upper()} asset.",
                            payload={"format": fmt, "path": str(res_path)},
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to export format '{fmt}': {e}")
                raise ExportError(f"Export of {fmt} failed: {e}") from e

        context.exported_assets = exported
        return context


__all__ = [
    "BaseAssetExporter",
    "GltfExporter",
    "BVHExporter",
    "BlenderExporter",
    "AssetExportStage",
]
