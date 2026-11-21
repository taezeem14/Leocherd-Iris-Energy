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
        for obj in list(bpy.data.objects):
            if obj.type not in {"ARMATURE", "MESH"} or obj.name.lower().startswith("cube") or "icosphere" in obj.name.lower():
                bpy.data.objects.remove(obj, do_unlink=True)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.type in {"ARMATURE", "MESH"}:
                obj.select_set(True)
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=True,
            bake_anim=True,
            bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=False,
            add_leaf_bones=False,
            path_mode="COPY",
            embed_textures=True,
            object_types={"ARMATURE", "MESH"},
        )
    else:
        print(f"[!] Unsupported format extension: {out_path}")
        sys.exit(1)

    print(f"[+] Successfully exported asset to {out_path}")


if __name__ == "__main__":
    main()
