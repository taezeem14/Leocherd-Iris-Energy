"""
Blender Standalone Python Automation: Armature Generator
Executes headlessly inside Blender to build native Armature rigs from Iris Energy JSON definitions.

Usage:
    blender --background --python blender/create_armature.py -- --input rig_data.json --output scene.blend
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
    # Extract arguments after "--"
    argv = sys.argv
    if "--" in argv:
        args_list = argv[argv.index("--") + 1 :]
    else:
        args_list = []

    parser = argparse.ArgumentParser(description="Create Blender Armature from JSON")
    parser.add_argument("--input", required=True, help="Path to input rig_data.json")
    parser.add_argument("--output", default="output_scene.blend", help="Path to output .blend file")
    return parser.parse_args(args_list)


def clean_scene():
    """Removes default cube, camera, and lamp."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_armature(rig_data):
    armature_data = bpy.data.armatures.new(name=rig_data.get("name", "IrisArmature"))
    armature_obj = bpy.data.objects.new(name=rig_data.get("name", "IrisRig"), object_data=armature_data)

    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_data.edit_bones
    joints = rig_data.get("joints", {})
    bones_list = rig_data.get("bones", [])

    created_bones = {}

    # 1. Create edit bones for each defined bone connection
    for bone_info in bones_list:
        b_name = bone_info["name"]
        p_name = bone_info["parent"]
        c_name = bone_info["child"]

        p_pos = joints[p_name]["rest_world_pos"]
        c_pos = joints[c_name]["rest_world_pos"]

        eb = edit_bones.new(b_name)
        # Coordinate conversion: Iris (+X right, +Y up, +Z fwd) -> Blender (+X right, +Y fwd, +Z up)
        eb.head = mathutils.Vector((p_pos[0], p_pos[2], p_pos[1]))
        eb.tail = mathutils.Vector((c_pos[0], c_pos[2], c_pos[1]))

        created_bones[b_name] = eb

    # 2. Establish parenting among edit bones
    for bone_info in bones_list:
        b_name = bone_info["name"]
        p_name = bone_info["parent"]
        # Find if parent joint is child of an upstream bone
        for other_bone in bones_list:
            if other_bone["child"] == p_name:
                parent_bone_name = other_bone["name"]
                if parent_bone_name in created_bones:
                    created_bones[b_name].parent = created_bones[parent_bone_name]
                    created_bones[b_name].use_connect = True
                break

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature_obj


def main():
    args = parse_args()
    print(f"[*] Reading rig data from: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        rig_data = json.load(f)

    clean_scene()
    arm_obj = build_armature(rig_data)
    print(f"[+] Successfully constructed Armature '{arm_obj.name}' with {len(arm_obj.data.bones)} bones.")

    bpy.ops.wm.save_as_mainfile(filepath=args.output)
    print(f"[+] Saved Blender scene to: {args.output}")


if __name__ == "__main__":
    main()
