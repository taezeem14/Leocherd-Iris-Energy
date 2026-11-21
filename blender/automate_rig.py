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
import struct
import sys
import traceback

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
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.armatures):
        bpy.data.armatures.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)


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


def retime_action_to_frame_range(action, target_start: float, target_end: float):
    """
    Linearly retimes all F-curve keyframes (and bezier handles) in `action`
    so the animation curves span exactly [target_start, target_end].
    """
    if not action or not action.fcurves:
        return
    src_start, src_end = float(action.frame_range[0]), float(action.frame_range[1])
    src_span = src_end - src_start
    dst_span = float(target_end) - float(target_start)
    if src_span <= 1e-6 or dst_span <= 1e-6:
        return

    scale = dst_span / src_span
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.co.x = target_start + (kp.co.x - src_start) * scale
            kp.handle_left.x = target_start + (kp.handle_left.x - src_start) * scale
            kp.handle_right.x = target_start + (kp.handle_right.x - src_start) * scale
        fc.update()

    new_start, new_end = float(action.frame_range[0]), float(action.frame_range[1])
    print(
        f"[OK] Retimed Action '{action.name}' F-curves: "
        f"[{src_start:.2f}, {src_end:.2f}] -> [{new_start:.2f}, {new_end:.2f}]"
    )


def ensure_packed_material_texture(mesh_obj, tex_file: Path):
    """
    Ensures `mesh_obj` has a Principled BSDF material whose Base Color image texture node
    references `tex_file` AND that the image is packed directly inside the .blend file.
    Preserves glTF-imported shader properties (such as cyan emissive tint) when present.
    """
    if not mesh_obj or not tex_file or not tex_file.exists():
        return

    resolved = tex_file.resolve()
    target_mat = mesh_obj.data.materials[0] if len(mesh_obj.data.materials) > 0 else None

    if target_mat is None:
        target_mat = bpy.data.materials.new(name="CreatureMaterial")
        target_mat.use_nodes = True
        mesh_obj.data.materials.append(target_mat)
    else:
        target_mat.use_nodes = True

    nodes = target_mat.node_tree.nodes
    links = target_mat.node_tree.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    tex_nodes = [n for n in nodes if n.type == "TEX_IMAGE"]
    if tex_nodes:
        tex_node = tex_nodes[0]
        if tex_node.image is None:
            tex_node.image = bpy.data.images.load(str(resolved), check_existing=True)
        else:
            # Ensure the image has its file bytes loaded and packed
            tex_node.image.filepath = str(resolved)
            try:
                tex_node.image.reload()
            except Exception:
                pass
    else:
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(str(resolved), check_existing=True)
        links.new(bsdf.inputs["Base Color"], tex_node.outputs["Color"])

    # Normalize image name (avoid .001 suffixes) and pack directly into the .blend file
    img = tex_node.image
    if img is not None:
        img.name = resolved.name
        if img.packed_file is None:
            img.pack()
        # Purge any duplicate unused images referencing the same filename
        for other in list(bpy.data.images):
            if other != img and other.users == 0:
                bpy.data.images.remove(other)

    # Pack all remaining external data so the .blend is 100% self-contained
    try:
        bpy.ops.file.pack_all()
    except Exception:
        pass

    packed_size = len(img.packed_file.data) if (img and img.packed_file) else 0
    print(
        f"[OK] Material '{target_mat.name}' bound to packed image '{img.name if img else 'None'}' "
        f"({packed_size:,} bytes packed inside .blend)."
    )


def patch_fbx_global_timespan(fbx_path: Path, frame_start: int, frame_end: int, fps: float):
    """
    Blender's FBX exporter hardcodes GlobalSettings.TimeSpanStop = 1.0s (FBX_KTIME) even when
    AnimationStack/Takes span the full animation. This performs a fixed-width in-place int64
    update of TimeSpanStart and TimeSpanStop so GlobalSettings matches AnimationStack & Takes.
    """
    FBX_KTIME = 46186158000  # FBX ticks per second
    start_ktime = int(round((float(frame_start) / float(fps)) * FBX_KTIME))
    stop_ktime = int(round((float(frame_end) / float(fps)) * FBX_KTIME))

    data = bytearray(fbx_path.read_bytes())
    patched = []
    for prop_name, val in [(b"TimeSpanStart", start_ktime), (b"TimeSpanStop", stop_ktime)]:
        sig = prop_name + b"S\x05\x00\x00\x00KTimeS\x04\x00\x00\x00TimeS\x00\x00\x00\x00L"
        idx = data.find(sig)
        if idx >= 0:
            val_off = idx + len(sig)
            data[val_off : val_off + 8] = struct.pack("<q", val)
            patched.append(f"{prop_name.decode()}={val / FBX_KTIME:.3f}s")
    if patched:
        fbx_path.write_bytes(data)
        print(f"[OK] Synchronized FBX GlobalSettings time span: {', '.join(patched)}")


def main():
    args = parse_args()
    print("=" * 65)
    print(" LEOCHERD IRIS ENERGY - BLENDER RIGGING & ANIMATION PIPELINE")
    print("=" * 65)

    clear_scene()

    # 0. Pre-read animation FPS & frame count BEFORE importing glTF so Blender's glTF importer
    #    samples timestamps at the true target FPS (30 FPS) instead of Blender's default 24 FPS!
    target_fps = 30.0
    target_frame_count = 0
    anim_data_cache = None
    if args.anim and Path(args.anim).exists():
        with open(args.anim, "r", encoding="utf-8") as f_a:
            anim_data_cache = json.load(f_a)
        target_fps = float(anim_data_cache.get("fps", 30.0))
        target_frame_count = len(anim_data_cache.get("timestamps", anim_data_cache.get("frames", [])))

    scene_fps = int(round(target_fps))
    bpy.context.scene.render.fps = scene_fps
    bpy.context.scene.render.fps_base = 1.0
    print(f"[OK] Pre-configured scene FPS: {scene_fps} FPS (source: {target_fps:.3f} FPS, {target_frame_count} keys)")

    # 1. Import 3D Mesh
    mesh_obj = None
    imported_armature = None
    if args.mesh:
        mesh_obj = import_3d_mesh(args.mesh)
        # Purge any stray helper meshes created by importers (e.g. glTF helper Icospheres)
        for obj in list(bpy.data.objects):
            if "icosphere" in obj.name.lower() or obj.name.lower().startswith("cube"):
                bpy.data.objects.remove(obj, do_unlink=True)
        for block in list(bpy.data.meshes):
            if "icosphere" in block.name.lower() or block.name.lower().startswith("cube"):
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

    # 3b. Ensure Creature Diffuse Texture is connected AND packed inside .blend
    tex_file = None
    if args.mesh:
        candidate_tex = Path(args.mesh).resolve().parent / "creature_diffuse.png"
        if candidate_tex.exists():
            tex_file = candidate_tex
    if tex_file and mesh_obj:
        ensure_packed_material_texture(mesh_obj, tex_file)

    # 4. Apply Animation (or retime glTF-imported Action to exact 1..N frame range at target FPS)
    has_action = (
        armature_obj is not None
        and armature_obj.animation_data is not None
        and armature_obj.animation_data.action is not None
    )
    if not has_action and anim_data_cache is not None:
        apply_animation(armature_obj, anim_data_cache)
    elif has_action:
        act = armature_obj.animation_data.action
        print(f"[OK] Preserving mathematically accurate animation action '{act.name}' from glTF import.")
        target_end = float(max(target_frame_count, int(round(act.frame_range[1])) + 1))
        retime_action_to_frame_range(act, 1.0, target_end)

    # 4b. Lock Scene Playback Range to match exact Action range
    target_frame_end = max(1, target_frame_count)
    if armature_obj and armature_obj.animation_data and armature_obj.animation_data.action:
        act = armature_obj.animation_data.action
        target_frame_end = max(target_frame_end, int(round(act.frame_range[1])))

    bpy.context.scene.render.fps = scene_fps
    bpy.context.scene.render.fps_base = 1.0
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = target_frame_end
    bpy.context.scene.frame_set(1)
    duration_sec = (target_frame_end - 1) / float(scene_fps) if target_frame_end > 1 else 0.0
    print(
        f"[OK] Scene playback range locked: 1..{target_frame_end} "
        f"({duration_sec:.2f}s @ {scene_fps} FPS)"
    )

    # 5. Exhaustive scene & orphan purge: keep ONLY Character_Mesh and Armature
    for obj in list(bpy.data.objects):
        if obj != mesh_obj and obj != armature_obj:
            bpy.data.objects.remove(obj, do_unlink=True)

    for block in list(bpy.data.meshes):
        if not mesh_obj or block != mesh_obj.data:
            bpy.data.meshes.remove(block)

    for block in list(bpy.data.armatures):
        if not armature_obj or block != armature_obj.data:
            bpy.data.armatures.remove(block)

    try:
        bpy.data.orphans_purge(do_recursive=True)
    except Exception:
        pass

    # 6. Save Output .blend (with packed texture, no .blend1 backup clutter)
    out_blend = Path(args.output).resolve()
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"[SUCCESS] Saved rigged Blender production scene to: {out_blend}")

    # 7. Clean FBX Export (Mesh and Armature only with full 1..N animation baked)
    if args.export_fbx:
        out_fbx = Path(args.export_fbx).resolve()
        out_fbx.parent.mkdir(parents=True, exist_ok=True)

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
            bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=False,
            bake_anim_step=1.0,
            bake_anim_simplify_factor=0.0,
            add_leaf_bones=False,
            path_mode="COPY",
            embed_textures=True,
            object_types={"ARMATURE", "MESH"},
        )
        patch_fbx_global_timespan(out_fbx, 1, target_frame_end, float(scene_fps))
        print(f"[SUCCESS] Exported production FBX to: {out_fbx}")

    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
