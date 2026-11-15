#!/usr/bin/env python3
"""
Leocherd Iris Energy - Central Command-Line Interface (CLI).
Universal Python-First AI Video-to-3D Rigging Engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iris.core.config import get_settings
from iris.core.logging import get_logger
from iris.core.pipeline import ExecutionContext, Pipeline
from iris.export import AssetExportStage
from iris.motion.reconstruction import MotionProcessingStage
from iris.pose.estimator import PoseEstimationStage
from iris.retargeting.animation import AnimationGenerationStage
from iris.rigging.topology import SkeletonReconstructionStage
from iris.video.ingestion import VideoIngestionService, VideoIngestionStage

logger = get_logger("iris.cli")


def cmd_inspect_video(args: argparse.Namespace) -> None:
    video_path = args.video_path
    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - VIDEO CONTAINER INSPECTION")
    print("=" * 65)
    try:
        service = VideoIngestionService()
        meta = service.probe(video_path)
        print(f" File Path              : {meta.path}")
        print(f" File Size              : {meta.file_size_bytes / (1024*1024):.2f} MB")
        print(f" Resolution             : {meta.width} x {meta.height} (Aspect Ratio: {meta.aspect_ratio:.2f})")
        print(f" Frame Rate             : {meta.fps:.2f} FPS")
        print(f" Total Frames           : {meta.frame_count}")
        print(f" Duration               : {meta.duration_sec:.2f} seconds")
        print(f" Codec / FourCC         : {meta.codec}")
        print("-" * 65)
        if meta.warnings:
            print(" Advisories:")
            for w in meta.warnings:
                print(f"   [!] {w}")
        else:
            print(" [OK] Video container is structurally valid and ready for 3D reconstruction.")
        print("=" * 65)
    except Exception as e:
        logger.error(f"Video inspection failed: {e}")
        sys.exit(1)


def cmd_reconstruct(args: argparse.Namespace) -> None:
    video_path = args.video_path
    morphology = args.morphology or "human"
    target_fps = args.fps or 15
    out_dir = args.output_dir or "outputs"
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    max_frames = args.max_frames

    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - 3D RIGGING & RECONSTRUCTION PIPELINE")
    print("=" * 65)
    print(f" Input Video            : {video_path}")
    print(f" Creature Morphology    : {morphology}")
    print(f" Processing Rate        : {target_fps} FPS")
    print(f" Max Frames             : {max_frames if max_frames else 'All'}")
    print(f" Target Exports         : {', '.join(formats).upper()}")
    print(f" Destination Directory  : {out_dir}")
    print("-" * 65)

    overrides = {}
    if max_frames:
        overrides["max_frames"] = max_frames

    context = ExecutionContext(
        input_path=video_path,
        output_dir=out_dir,
        morphology_name=morphology,
        target_fps=target_fps,
        config_overrides=overrides,
    )

    pipeline = Pipeline("IrisEndToEndReconstruction")
    pipeline.add_stage(VideoIngestionStage())
    pipeline.add_stage(PoseEstimationStage())
    pipeline.add_stage(MotionProcessingStage())
    pipeline.add_stage(SkeletonReconstructionStage())
    pipeline.add_stage(AnimationGenerationStage())
    pipeline.add_stage(AssetExportStage(formats=formats))

    try:
        result = pipeline.execute(context)
        print("-" * 65)
        print(" RECONSTRUCTION SUCCESSFUL!")
        print(f" Duration               : {result.end_time - result.start_time:.2f} seconds")
        print(" Exported 3D Assets:")
        for fmt, path in result.exported_assets.items():
            print(f"   [{fmt.upper():<7}] -> {path}")
        print("=" * 65)
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        sys.exit(1)


def cmd_import_kaggle(args: argparse.Namespace) -> None:
    json_path = args.json_path
    morphology = args.morphology or "human"
    out_dir = args.output_dir or "outputs"
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - IMPORT KAGGLE GPU 3D MOTION")
    print("=" * 65)
    print(f" Kaggle JSON File       : {json_path}")
    print(f" Creature Morphology    : {morphology}")
    print(f" Target Exports         : {', '.join(formats).upper()}")
    print(f" Destination Directory  : {out_dir}")
    print("-" * 65)

    from iris.models.adapters.kaggle_import import KagglePoseImportAdapter

    adapter = KagglePoseImportAdapter(json_path)
    poses = adapter.estimate_sequence([])
    fps = adapter.data.get("fps", 15)

    context = ExecutionContext(
        input_path=json_path,
        output_dir=out_dir,
        morphology_name=morphology,
        target_fps=int(fps),
    )
    context.pose_sequence = poses

    pipeline = Pipeline("IrisKaggleReconstruction")
    pipeline.add_stage(MotionProcessingStage())
    pipeline.add_stage(SkeletonReconstructionStage())
    pipeline.add_stage(AnimationGenerationStage())
    pipeline.add_stage(AssetExportStage(formats=formats))

    try:
        result = pipeline.execute(context)
        print("-" * 65)
        print(" RECONSTRUCTION FROM KAGGLE GPU DATA SUCCESSFUL!")
        print(f" Duration               : {result.end_time - result.start_time:.2f} seconds")
        print(" Exported 3D Assets:")
        for fmt, path in result.exported_assets.items():
            print(f"   [{fmt.upper():<7}] -> {path}")
        print("=" * 65)
    except Exception as e:
        logger.error(f"Kaggle import failed: {e}")
        sys.exit(1)


def cmd_run_server(args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    print(f"[*] Starting Leocherd Iris Energy REST Service on http://{host}:{port} ...")
    uvicorn.run("iris.api.main:app", host=host, port=port, reload=args.reload)


def cmd_diagnostics(args: argparse.Namespace) -> None:
    from scripts.system_diagnostics import inspect_system, print_report

    profile = inspect_system()
    print_report(profile)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="iris",
        description="Leocherd Iris Energy - Python-First Universal AI Video-to-3D Rigging Engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect-video
    p_inspect = subparsers.add_parser("inspect-video", help="Inspect video metadata and compatibility")
    p_inspect.add_argument("video_path", help="Path to input video file (.mp4, .mov, etc.)")
    p_inspect.set_defaults(func=cmd_inspect_video)

    # reconstruct
    p_recon = subparsers.add_parser("reconstruct", help="Run end-to-end 3D rigging and reconstruction")
    p_recon.add_argument("video_path", help="Path to input video file")
    p_recon.add_argument("--morphology", "-m", default="human", help="Creature morphology (human, quadruped, etc.)")
    p_recon.add_argument("--fps", type=int, default=15, help="Sampling frame rate (default: 15)")
    p_recon.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to extract and process")
    p_recon.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    p_recon.add_argument("--formats", default="gltf,bvh,blender", help="Comma-separated export formats")
    p_recon.set_defaults(func=cmd_reconstruct)

    # import-kaggle
    p_kaggle = subparsers.add_parser("import-kaggle", help="Generate 3D rigs and animation from Kaggle GPU exported JSON")
    p_kaggle.add_argument("json_path", help="Path to Kaggle exported iris_kaggle_pose_data.json")
    p_kaggle.add_argument("--morphology", "-m", default="human", help="Creature morphology (human, quadruped, etc.)")
    p_kaggle.add_argument("--output-dir", "-o", default="outputs/kaggle_export", help="Output directory")
    p_kaggle.add_argument("--formats", default="gltf,bvh,blender", help="Comma-separated export formats")
    p_kaggle.set_defaults(func=cmd_import_kaggle)

    # run-server
    p_server = subparsers.add_parser("run-server", help="Launch FastAPI REST service")
    p_server.add_argument("--host", default="127.0.0.1", help="API Host")
    p_server.add_argument("--port", type=int, default=8000, help="API Port")
    p_server.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_server.set_defaults(func=cmd_run_server)

    # diagnostics
    p_diag = subparsers.add_parser("diagnostics", help="Run system hardware and environment diagnostics")
    p_diag.set_defaults(func=cmd_diagnostics)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


cli = main


if __name__ == "__main__":
    main()
