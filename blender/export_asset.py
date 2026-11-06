"""
Blender Standalone Python Automation: Scene Exporter
Exports animated Blender armatures to .glb or .fbx formats.

Usage:
    blender --background scene.blend --python blender/export_asset.py -- --output export.glb
"""

import argparse
import sys

try:
    import bpy
except ImportError:
    print("[ERROR] This script must be run inside Blender's Python runtime.")
    sys.exit(1)


def parse_args():
    argv = sys.argv
    args_list = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Export Blender scene to glTF/GLB")
    parser.add_argument("--output", required=True, help="Path to output export file (.glb, .gltf, .fbx)")
    return parser.parse_args(args_list)


def main():
    args = parse_args()
    out_path = args.output
    print(f"[*] Exporting scene to: {out_path}")

    if out_path.lower().endswith(".glb") or out_path.lower().endswith(".gltf"):
        bpy.ops.export_scene.gltf(
            filepath=out_path,
            export_format="GLB" if out_path.lower().endswith(".glb") else "GLTF_EMBEDDED",
            export_animations=True,
            export_skins=True,
            export_def_bones=True,
        )
    elif out_path.lower().endswith(".fbx"):
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            bake_anim=True,
            add_leaf_bones=False,
        )
    else:
        print(f"[!] Unsupported format extension: {out_path}")
        sys.exit(1)

    print(f"[+] Successfully exported asset to {out_path}")


if __name__ == "__main__":
    main()
