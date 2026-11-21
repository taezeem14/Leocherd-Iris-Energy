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
from iris.retargeting.retargeter import RetargetingStage
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
    skinning_method = getattr(args, "skinning_method", "heat_diffusion")
    retarget_to = getattr(args, "retarget_to", None)

    max_frames = args.max_frames

    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - 3D RIGGING & RECONSTRUCTION PIPELINE")
    print("=" * 65)
    print(f" Input Video            : {video_path}")
    print(f" Creature Morphology    : {morphology}")
    if retarget_to:
        print(f" Retargeting Target     : {retarget_to}")
    print(f" Skinning Method        : {skinning_method}")
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
        skinning_method=skinning_method,
        retarget_to=retarget_to,
        config_overrides=overrides,
    )

    pipeline = Pipeline("IrisEndToEndReconstruction")
    pipeline.add_stage(VideoIngestionStage())
    pipeline.add_stage(PoseEstimationStage())
    pipeline.add_stage(MotionProcessingStage())
    pipeline.add_stage(SkeletonReconstructionStage())
    pipeline.add_stage(AnimationGenerationStage())
    pipeline.add_stage(RetargetingStage())
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


def cmd_mesh_reconstruct(args: argparse.Namespace) -> None:
    video_path = args.video_path
    method = args.method or "depth_anything"
    out_dir = args.output_dir or "outputs/surface_mesh"
    step = args.frame_interval or 15
    max_f = args.max_frames or 10

    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - 3D SURFACE & MESH RECONSTRUCTION")
    print("=" * 65)
    print(f" Input Video            : {video_path}")
    print(f" Reconstruction Method  : {method.upper()}")
    print(f" Frame Step Interval    : {step}")
    print(f" Max Evaluated Frames   : {max_f}")
    print(f" Destination Directory  : {out_dir}")
    print("-" * 65)

    try:
        if method == "sfm":
            from iris.mesh.sfm_reconstruction import VideoStructureFromMotion
            sfm = VideoStructureFromMotion()
            res = sfm.process_video(video_path, out_dir, frame_step=step, max_pairs=max_f)
        else:
            from iris.mesh.surface_reconstruction import MonocularDepthReconstructor
            recon = MonocularDepthReconstructor()
            res = recon.process_video(video_path, out_dir, frame_interval=step, max_frames=max_f)

        print("-" * 65)
        print(" 3D SURFACE RECONSTRUCTION SUCCESSFUL!")
        for k, v in res.items():
            print(f"   [{k.upper():<18}] -> {v}")
        print("=" * 65)
    except Exception as e:
        logger.error(f"3D mesh reconstruction failed: {e}")
        sys.exit(1)


def cmd_creature_reconstruct(args: argparse.Namespace) -> None:
    from iris.mesh.creature_rig import CreatureReconstructionPipeline

    image_path = args.image_path
    anim_source = getattr(args, "animation", None)
    out_dir = args.output_dir or "outputs/creature_rig"
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]

    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - 2D CREATURE IMAGE TO 3D SKINNED RIG")
    print("=" * 65)
    print(f" Creature Image        : {image_path}")
    print(f" Motion Source         : {anim_source or 'Default Creature Pose / Sway'}")
    print(f" Target Engine Formats : {', '.join(formats).upper()}")
    print(f" Destination Directory : {out_dir}")
    print("-" * 65)

    pipeline = CreatureReconstructionPipeline()
    res = pipeline.process_creature(
        image_path=image_path,
        output_dir=out_dir,
        animation_source_json=anim_source,
        export_formats=formats,
    )

    print("-" * 65)
    print(" 🎉 CREATURE 3D RIGGING & SKINNED MESH RECONSTRUCTION SUCCESSFUL!")
    print(" Exported Multi-Engine Production Assets:")
    for k, v in res.items():
        print(f"   [{k.upper():<14}] -> {v}")
    print("=" * 65)
    print("👉 View instantly in browser: python main.py view")
    print("👉 Or drag creature_rigged.gltf into http://localhost:8000/viewer")
    print("=" * 65)


def cmd_view(args: argparse.Namespace) -> None:
    import webbrowser
    import uvicorn
    from iris.core.config import get_settings

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    url = f"http://{host}:{port}/viewer"

    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - WEB 3D SKELETAL & MOCAP VIEWER")
    print("=" * 65)
    print(f" Serving Viewer at     : {url}")
    print(f" Drop Zone Ready For   : .gltf, .glb, .obj")
    print("=" * 65)

    webbrowser.open(url)
    uvicorn.run("iris.api.main:app", host=host, port=port, reload=False)


def cmd_diagnostics(args: argparse.Namespace) -> None:
    from scripts.system_diagnostics import inspect_system, print_report

    profile = inspect_system()
    print_report(profile)


def cmd_verify_exports(args: argparse.Namespace) -> None:
    """Runs the full automated QA suite across exported glTF, .blend, .fbx, OBJ/MTL, and BVH assets."""
    from scripts.verify_production_bundle import verify_bundle

    report = verify_bundle(Path(args.directory))
    print("=" * 70)
    print(f" LEOCHERD IRIS ENERGY - PRODUCTION EXPORT VERIFICATION: {report['status']}")
    print("=" * 70)
    k = report["checks"].get("khronos_gltf_validator", {})
    if k:
        print(
            f" Khronos glTF-Validator ({k.get('validator_version')}): "
            f"{k.get('errors')} errors, {k.get('warnings')} warnings, {k.get('infos')} infos, {k.get('hints')} hints"
        )
    bl = report["checks"].get("blender_blend", {})
    if bl:
        print(
            f" Blender .blend : frames {bl['frame_start']}..{bl['frame_end']} @ {bl['fps']} FPS | "
            f"action {bl['action_frame_range']} | bones={bl['bone_count']} | cubes={bl['cube_count']}"
        )
    fbx = report["checks"].get("blender_fbx", {})
    if fbx:
        print(
            f" Unreal/Unity .fbx: raw_meshes={len(fbx['raw_mesh_models'])} | raw_cubes={fbx['raw_cube_count']} | "
            f"clean_import_cubes={fbx['clean_import_cube_count']} | bones={fbx['bone_count']}"
        )
    if report["failures"]:
        for f in report["failures"]:
            print(f" [FAIL] {f}")
        sys.exit(1)
    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
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
    p_recon.add_argument(
        "--morphology",
        "-m",
        default="human",
        help="Creature morphology: human, quadruped, avian, serpentine, tentacular, auto, discover (default: human)",
    )
    p_recon.add_argument("--fps", type=int, default=15, help="Sampling frame rate (default: 15)")
    p_recon.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to extract and process")
    p_recon.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    p_recon.add_argument("--formats", default="gltf,bvh,blender", help="Comma-separated export formats")
    p_recon.add_argument(
        "--skinning-method",
        choices=["heat_diffusion", "volumetric", "geodesic", "lbs"],
        default="heat_diffusion",
        help="Skin weighting method: heat_diffusion, volumetric, geodesic, lbs (default: heat_diffusion)",
    )
    p_recon.add_argument(
        "--retarget-to",
        default=None,
        help="Cross-morphology retargeting target: human, quadruped, avian, serpentine, tentacular (default: None)",
    )
    p_recon.set_defaults(func=cmd_reconstruct)

    # import-kaggle
    p_kaggle = subparsers.add_parser("import-kaggle", help="Generate 3D rigs and animation from Kaggle GPU exported JSON")
    p_kaggle.add_argument("json_path", help="Path to Kaggle exported iris_kaggle_pose_data.json")
    p_kaggle.add_argument(
        "--morphology",
        "-m",
        default="human",
        help="Creature morphology: human, quadruped, avian, serpentine, tentacular, auto, discover (default: human)",
    )
    p_kaggle.add_argument("--output-dir", "-o", default="outputs/kaggle_export", help="Output directory")
    p_kaggle.add_argument("--formats", default="gltf,bvh,blender", help="Comma-separated export formats")
    p_kaggle.set_defaults(func=cmd_import_kaggle)

    # creature-reconstruct
    p_creature = subparsers.add_parser("creature-reconstruct", help="Reconstruct 3D skinned mesh and rig from 2D creature image")
    p_creature.add_argument("image_path", help="Path to creature image / concept art (.jpg, .png, etc.)")
    p_creature.add_argument("--animation", "-a", default=None, help="Optional animation JSON or mocap to retarget onto creature")
    p_creature.add_argument("--output-dir", "-o", default="outputs/creature_rig", help="Output directory")
    p_creature.add_argument("--formats", default="gltf,blend,fbx,bvh", help="Comma-separated export formats (gltf,blend,fbx,bvh)")
    p_creature.set_defaults(func=cmd_creature_reconstruct)

    # view
    p_view = subparsers.add_parser("view", help="Launch interactive 3D Web Viewer in browser")
    p_view.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    p_view.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_view.set_defaults(func=cmd_view)

    # run-server
    p_server = subparsers.add_parser("run-server", help="Launch FastAPI REST service")
    p_server.add_argument("--host", default="127.0.0.1", help="API Host")
    p_server.add_argument("--port", type=int, default=8000, help="API Port")
    p_server.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_server.set_defaults(func=cmd_run_server)

    # mesh-reconstruct
    p_mesh = subparsers.add_parser("mesh-reconstruct", help="Extract 3D point cloud and surface mesh from video (Depth Anything / SfM)")
    p_mesh.add_argument("video_path", help="Path to input video file")
    p_mesh.add_argument(
        "--method",
        choices=["depth_anything", "sfm"],
        default="depth_anything",
        help="Reconstruction method: depth_anything (AI Depth) or sfm (Structure-from-Motion)",
    )
    p_mesh.add_argument("--frame-interval", type=int, default=15, help="Sampling frame interval (default: 15)")
    p_mesh.add_argument("--max-frames", type=int, default=10, help="Maximum keyframes to extract for 3D meshing")
    p_mesh.add_argument("--output-dir", "-o", default="outputs/surface_mesh", help="Output directory")
    p_mesh.set_defaults(func=cmd_mesh_reconstruct)

    # diagnostics
    p_diag = subparsers.add_parser("diagnostics", help="Run system hardware and environment diagnostics")
    p_diag.set_defaults(func=cmd_diagnostics)

    # verify-exports
    p_verify = subparsers.add_parser(
        "verify-exports",
        help="Run automated QA verification (Khronos glTF-Validator, Blender .blend, FBX, OBJ/MTL, BVH)",
    )
    p_verify.add_argument("directory", nargs="?", default="outputs/creature_dwarf", help="Export directory to audit")
    p_verify.set_defaults(func=cmd_verify_exports)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


cli = main


if __name__ == "__main__":
    main()
