"""
Clean FBX Importer for Blender (4.2+ / 4.5+).

Clears Blender's default startup scene (removing the default 2-meter Cube, Camera, and Light),
sets 30 FPS, imports the target FBX with anim_offset=0.0, and automatically synchronizes
the Blender scene timeline (frame_start..frame_end) to the full imported animation action.

Usage (GUI or headless):
    blender --python blender/import_fbx_clean.py -- outputs/creature_dwarf/creature_rigged.fbx
    blender -b --python blender/import_fbx_clean.py -- outputs/creature_dwarf/creature_rigged.fbx
"""

import sys
from pathlib import Path
import bpy


def main():
    argv = sys.argv
    args = argv[argv.index("--") + 1 :] if "--" in argv else []
    if not args:
        print("Usage: blender --python blender/import_fbx_clean.py -- <path/to/model.fbx>")
        sys.exit(1)

    fbx_path = Path(args[0]).resolve()
    if not fbx_path.exists():
        print(f"[ERROR] FBX file not found: {fbx_path}")
        sys.exit(1)

    # Clear Blender's default startup Cube/Camera/Light
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0

    # Import FBX with exact 1-based keyframe alignment
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), anim_offset=0.0)

    # Sync scene playback range to full imported action range
    f_start, f_end = 1, 1
    for act in bpy.data.actions:
        a0, a1 = int(round(act.frame_range[0])), int(round(act.frame_range[1]))
        if a1 > f_end:
            f_start, f_end = max(1, a0), a1

    bpy.context.scene.frame_start = f_start
    bpy.context.scene.frame_end = f_end
    bpy.context.scene.frame_set(f_start)

    print(f"[OK] Imported '{fbx_path.name}' cleanly into empty scene:")
    print(f"     Objects    : {[o.name for o in bpy.data.objects]}")
    print(f"     Timeline   : {f_start}..{f_end} @ {bpy.context.scene.render.fps} FPS ({(f_end - f_start)/30.0:.2f}s)")


if __name__ == "__main__":
    main()
