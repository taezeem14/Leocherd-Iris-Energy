"""
Blender Standalone Python Automation: Animation Keyframer
Applies Iris Energy animation keyframes (root translation and bone rotations) to a Blender armature.

Usage:
    blender --background scene.blend --python blender/apply_animation.py -- --animation anim_data.json --output animated_scene.blend
"""

import argparse
import json
import sys

try:
    import bpy
    import mathutils
except ImportError:
    print("[ERROR] This script must be run inside Blender's Python runtime.")
    sys.exit(1)


def parse_args():
    argv = sys.argv
    args_list = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Apply animation to Blender Armature")
    parser.add_argument("--animation", required=True, help="Path to anim_data.json")
    parser.add_argument("--output", default="animated_scene.blend", help="Path to output .blend file")
    return parser.parse_args(args_list)


def apply_animation(armature_obj, anim_data):
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="POSE")

    pose = armature_obj.pose
    fps = anim_data.get("fps", 30.0)
    bpy.context.scene.render.fps = int(round(fps))

    timestamps = anim_data.get("timestamps", [])
    root_translations = anim_data.get("root_translations", [])
    rotations = anim_data.get("rotations", {})

    T = len(timestamps)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = T

    # Set rotation mode to QUATERNION on all pose bones
    for pbone in pose.bones:
        pbone.rotation_mode = "QUATERNION"

    root_bone_name = None
    for pb in pose.bones:
        if pb.parent is None:
            root_bone_name = pb.name
            break

    print(f"[*] Keyframing {T} frames on armature '{armature_obj.name}'...")

    for frame_idx in range(T):
        current_frame = frame_idx + 1

        # Keyframe root translation
        if root_bone_name and frame_idx < len(root_translations):
            rx, ry, rz = root_translations[frame_idx]
            # Convert Iris coordinates to Blender
            pbone = pose.bones[root_bone_name]
            pbone.location = mathutils.Vector((rx, rz, ry))
            pbone.keyframe_insert(data_path="location", frame=current_frame)

        # Keyframe bone rotations
        for b_name, rot_array in rotations.items():
            if b_name in pose.bones and frame_idx < len(rot_array):
                qw, qx, qy, qz = rot_array[frame_idx]
                # Convert Iris quaternion to Blender: (w, x, z, y)
                pbone = pose.bones[b_name]
                pbone.rotation_quaternion = mathutils.Quaternion((qw, qx, qz, qy))
                pbone.keyframe_insert(data_path="rotation_quaternion", frame=current_frame)

    bpy.ops.object.mode_set(mode="OBJECT")


def main():
    args = parse_args()
    with open(args.animation, "r", encoding="utf-8") as f:
        anim_data = json.load(f)

    armature_obj = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armature_obj = obj
            break

    if not armature_obj:
        print("[ERROR] No Armature object found in current Blender scene.")
        sys.exit(1)

    apply_animation(armature_obj, anim_data)
    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print(f"[+] Animation keyframed successfully. Saved to {args.output}")


if __name__ == "__main__":
    main()
