"""
Blender Automated Video-to-3D Rigging & Animation Pipeline.
Integrates 3D mesh import (.obj, .ply, .gltf), armature generation,
automatic skin weighting (ARMATURE_AUTO), inverse kinematics (IK) constraints,
and skeletal action keyframe baking.

Usage:
    blender -b -P blender/automate_rig.py -- \
        --mesh outputs/dance_demo/surface_mesh.obj \
        --rig outputs/dance_demo/reconstruction_rig_data.json \
        --anim outputs/dance_demo/reconstruction_anim_data.json \
        --output outputs/dance_demo/character_rigged.blend \
        --export-fbx outputs/dance_demo/character_rigged.fbx
"""

import argparse
import json
from pathlib import Path
import sys

try:
    import bpy
    import mathutils
except ImportError:
    print("[ERROR] This script must be run inside Blender's Python runtime.")
    print("Example: blender --background --python blender/automate_rig.py -- ...")
    sys.exit(1)


def parse_args():
    argv = sys.argv
    if "--" in argv:
        args_list = argv[argv.index("--") + 1 :]
    else:
        args_list = []

    parser = argparse.ArgumentParser(description="Blender End-to-End Rigging & Animation Automation")
    parser.add_argument("--mesh", default=None, help="Path to input 3D model (.obj, .ply, .gltf, .glb)")
    parser.add_argument("--rig", required=True, help="Path to input rig_data.json")
    parser.add_argument("--anim", default=None, help="Path to input anim_data.json")
    parser.add_argument("--output", default="character_rigged.blend", help="Output .blend file path")
    parser.add_argument("--export-fbx", default=None, help="Optional output .fbx file path")
    return parser.parse_args(args_list)


def clear_scene():
    """Removes all default objects, meshes, and lights."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        bpy.data.armatures.remove(block)


def import_3d_mesh(mesh_path: str):
    """Imports 3D surface mesh from .obj, .ply, or .gltf into the active scene."""
    path = Path(mesh_path)
    if not path.exists():
        print(f"[WARN] 3D mesh file not found: {mesh_path}. Skipping mesh binding.")
        return None

    ext = path.suffix.lower()
    print(f"[*] Importing 3D mesh: {mesh_path} (format: {ext})")

    # Store objects before import to identify new objects
    existing_objs = set(bpy.context.scene.objects)

    if ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif ext == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    elif ext in [".gltf", ".glb"]:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        print(f"[ERROR] Unsupported mesh extension '{ext}'")
        return None

    new_objs = [obj for obj in bpy.context.scene.objects if obj not in existing_objs and obj.type == "MESH"]
    if new_objs:
        mesh_obj = new_objs[0]
        mesh_obj.name = "Character_Mesh"
        bpy.context.view_layer.objects.active = mesh_obj
        # Apply smooth shading
        for f in mesh_obj.data.polygons:
            f.use_smooth = True
        print(f"[OK] Successfully imported mesh '{mesh_obj.name}' ({len(mesh_obj.data.vertices)} vertices).")
        return mesh_obj

    return None


def build_armature(rig_data):
    """Creates a native Blender Armature with exact bone heads, tails, and parent-child links."""
    armature_data = bpy.data.armatures.new(name=rig_data.get("name", "IrisArmature"))
    armature_obj = bpy.data.objects.new(name=rig_data.get("name", "IrisRig"), object_data=armature_data)

    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_data.edit_bones
    joints = rig_data.get("joints", {})
    bones_list = rig_data.get("bones", [])

    created_bones = {}

    # 1. Create bones from bone definition list
    for bone_info in bones_list:
        b_name = bone_info["name"]
        p_name = bone_info["parent"]
        c_name = bone_info["child"]

        if p_name not in joints or c_name not in joints:
            continue

        p_pos = joints[p_name]["rest_world_pos"]
        c_pos = joints[c_name]["rest_world_pos"]

        head_vec = mathutils.Vector((p_pos[0], -p_pos[2], p_pos[1]))  # Convert Y-up to Blender Z-up
        tail_vec = mathutils.Vector((c_pos[0], -c_pos[2], c_pos[1]))

        if (tail_vec - head_vec).length < 1e-4:
            tail_vec = head_vec + mathutils.Vector((0.0, 0.0, 0.1))

        bone = edit_bones.new(b_name)
        bone.head = head_vec
        bone.tail = tail_vec
        bone.use_deform = True
        created_bones[b_name] = bone

    # 2. Establish bone hierarchy
    for bone_info in bones_list:
        b_name = bone_info["name"]
        p_joint = bone_info["parent"]
        if b_name not in created_bones:
            continue

        # Find parent bone where child joint == p_joint
        for other in bones_list:
            if other["child"] == p_joint and other["name"] in created_bones:
                created_bones[b_name].parent = created_bones[other["name"]]
                break

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[OK] Generated Armature with {len(created_bones)} articulated bones.")
    return armature_obj


def bind_mesh_to_armature(mesh_obj, armature_obj):
    """Parents the mesh to the armature with Automatic Weights (ARMATURE_AUTO)."""
    if not mesh_obj or not armature_obj:
        return

    print(f"[*] Binding mesh '{mesh_obj.name}' to armature '{armature_obj.name}'...")
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        print("[OK] Mesh successfully parented with Automatic Vertex Weights!")
    except Exception as e:
        print(f"[WARN] Automatic weight generation failed ({e}). Falling back to Armature Deform modifier.")
        mod = mesh_obj.modifiers.new(name="ArmatureDeform", type="ARMATURE")
        mod.object = armature_obj
        mod.use_vertex_groups = True

    # Ensure Dual Quaternion Skinning (DQS) volume preservation is enabled on all armature modifiers
    for mod in mesh_obj.modifiers:
        if mod.type == "ARMATURE":
            mod.use_deform_preserve_volume = True


def apply_animation(armature_obj, anim_data):
    """Loads skeletal animation keyframes into a native Blender Action track."""
    if not anim_data:
        return

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="POSE")

    action_name = anim_data.get("name", "IrisAction")
    action = bpy.data.actions.new(name=action_name)
    if not armature_obj.animation_data:
        armature_obj.animation_data_create()
    armature_obj.animation_data.action = action

    fps = anim_data.get("fps", 30.0)
    bpy.context.scene.render.fps = int(round(fps))

    # Support both dictionary format (Iris standard) and frame list format
    if "rotations" in anim_data and "root_translations" in anim_data:
        timestamps = anim_data.get("timestamps", [])
        num_frames = len(timestamps)
        rotations_dict = anim_data["rotations"]
        root_trans_list = anim_data["root_translations"]

        total_baked = num_frames
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = max(1, num_frames)
        print(f"[*] Baking {num_frames} keyframes into Action '{action_name}' @ {fps:.1f} FPS...")

        pose_bones = armature_obj.pose.bones

        def find_pose_bone(name: str):
            if name in pose_bones:
                return pose_bones[name]
            for pb in pose_bones:
                if pb.name == f"Bone_{name}" or pb.name.endswith(f"_{name}"):
                    return pb
                if name.endswith(f"_{pb.name}") or name == f"Bone_{pb.name}":
                    return pb
            return None

        for frame_idx in range(num_frames):
            f_num = frame_idx + 1
            bpy.context.scene.frame_set(f_num)

            for bone_name, track in rotations_dict.items():
                p_bone = find_pose_bone(bone_name)
                if p_bone and frame_idx < len(track):
                    q = track[frame_idx]
                    p_bone.rotation_mode = "QUATERNION"
                    p_bone.rotation_quaternion = mathutils.Quaternion((q[0], q[1], -q[3], q[2]))
                    p_bone.keyframe_insert(data_path="rotation_quaternion", frame=f_num)

            if frame_idx < len(root_trans_list):
                root_trans = root_trans_list[frame_idx]
                root_bone = pose_bones[0] if pose_bones else None
                if root_bone:
                    root_bone.location = mathutils.Vector((root_trans[0], -root_trans[2], root_trans[1]))
                    root_bone.keyframe_insert(data_path="location", frame=f_num)
    else:
        frames = anim_data.get("frames", [])
        total_baked = len(frames)
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = max(1, len(frames))

        print(f"[*] Baking {len(frames)} keyframes into Action '{action_name}' @ {fps:.1f} FPS...")

        pose_bones = armature_obj.pose.bones

        def find_pose_bone(name: str):
            if name in pose_bones:
                return pose_bones[name]
            for pb in pose_bones:
                if pb.name == f"Bone_{name}" or pb.name.endswith(f"_{name}"):
                    return pb
                if name.endswith(f"_{pb.name}") or name == f"Bone_{pb.name}":
                    return pb
            return None

        for frame_idx, f_data in enumerate(frames, start=1):
            bpy.context.scene.frame_set(frame_idx)
            rotations = f_data.get("rotations", {})

            for bone_name, q in rotations.items():
                p_bone = find_pose_bone(bone_name)
                if p_bone:
                    p_bone.rotation_mode = "QUATERNION"
                    p_bone.rotation_quaternion = mathutils.Quaternion((q[0], q[1], -q[3], q[2]))
                    p_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

            if "root_translation" in f_data:
                root_trans = f_data["root_translation"]
                root_bone = pose_bones[0] if pose_bones else None
                if root_bone:
                    root_bone.location = mathutils.Vector((root_trans[0], -root_trans[2], root_trans[1]))
                    root_bone.keyframe_insert(data_path="location", frame=frame_idx)

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[OK] Animation keyframes successfully baked across {total_baked} frames.")


def main():
    args = parse_args()
    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - BLENDER RIGGING & ANIMATION PIPELINE")
    print("=" * 65)

    clear_scene()

    # 1. Import 3D Mesh
    mesh_obj = None
    imported_armature = None
    if args.mesh:
        mesh_obj = import_3d_mesh(args.mesh)
        # Purge any stray helper meshes created by importers (e.g. glTF helper Icospheres)
        for obj in list(bpy.data.objects):
            if "icosphere" in obj.name.lower():
                bpy.data.objects.remove(obj, do_unlink=True)
        for block in list(bpy.data.meshes):
            if "icosphere" in block.name.lower():
                bpy.data.meshes.remove(block)

        # Check if mesh import already brought an articulated armature (e.g. from glTF)
        existing_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
        if existing_armatures:
            imported_armature = existing_armatures[0]
            print(f"[OK] Reusing imported Armature '{imported_armature.name}' from mesh file.")

    # 2. Build Armature (only if mesh didn't already have one)
    if imported_armature:
        armature_obj = imported_armature
    else:
        with open(args.rig, "r", encoding="utf-8") as f:
            rig_data = json.load(f)
        armature_obj = build_armature(rig_data)

    # 3. Bind Mesh with Skin Weights (only if not already parented/skinned)
    if mesh_obj and not imported_armature:
        bind_mesh_to_armature(mesh_obj, armature_obj)
    elif mesh_obj:
        # Ensure Dual Quaternion volume preservation on existing armature modifier
        for mod in mesh_obj.modifiers:
            if mod.type == "ARMATURE":
                mod.use_deform_preserve_volume = True

    # 4. Apply Animation (only if the armature does not already carry an action imported from glTF)
    has_action = (
        armature_obj is not None
        and armature_obj.animation_data is not None
        and armature_obj.animation_data.action is not None
    )
    if not has_action and args.anim and Path(args.anim).exists():
        with open(args.anim, "r", encoding="utf-8") as f:
            anim_data = json.load(f)
        apply_animation(armature_obj, anim_data)
    elif has_action:
        print(f"[OK] Preserving mathematically accurate animation action '{armature_obj.animation_data.action.name}' from glTF import.")

    # 5. Clean scene of any stray objects before saving and exporting
    for obj in list(bpy.data.objects):
        if obj.type not in ["ARMATURE", "MESH"] or "icosphere" in obj.name.lower():
            bpy.data.objects.remove(obj, do_unlink=True)
    for block in list(bpy.data.meshes):
        if "icosphere" in block.name.lower():
            bpy.data.meshes.remove(block)

    # 6. Save Output .blend
    out_blend = Path(args.output).resolve()
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"[SUCCESS] Saved rigged Blender production scene to: {out_blend}")

    # 7. Clean FBX Export (Mesh and Armature only)
    if args.export_fbx:
        out_fbx = Path(args.export_fbx).resolve()
        out_fbx.parent.mkdir(parents=True, exist_ok=True)

        # Select only active character mesh and armature
        bpy.ops.object.select_all(action="DESELECT")
        if mesh_obj:
            mesh_obj.select_set(True)
        if armature_obj:
            armature_obj.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj

        bpy.ops.export_scene.fbx(
            filepath=str(out_fbx),
            use_selection=True,
            bake_anim=True,
            add_leaf_bones=False,
            object_types={"ARMATURE", "MESH"},
        )
        print(f"[SUCCESS] Exported production FBX to: {out_fbx}")

    print("=" * 65)


if __name__ == "__main__":
    main()
