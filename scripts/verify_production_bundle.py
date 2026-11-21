"""
End-to-End Production Asset Verification Suite for Leocherd Iris Energy.

Verifies every deliverable in an export directory (glTF 2.0 via official Khronos
glTF-Validator + strict math checks, Blender .blend, Unreal/Unity .fbx via both
raw FBX binary parsing and clean-scene Blender import, OBJ/MTL/PNG, and BVH)
and emits machine-readable JSON reports.

Usage:
    python scripts/verify_production_bundle.py outputs/creature_dwarf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iris.core.config import get_settings
from scripts.validate_gltf_strict import validate_gltf_strict


BLENDER_AUDIT_SCRIPT = r"""
import json
import math
import sys
from pathlib import Path
import bpy

mode = sys.argv[sys.argv.index("--") + 1]
asset_path = Path(sys.argv[sys.argv.index("--") + 2]).resolve()
out_json = Path(sys.argv[sys.argv.index("--") + 3]).resolve()

def bone_pose_signature(arm_obj, frame: int):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    sig = {}
    for pb in arm_obj.pose.bones:
        m = pb.matrix
        sig[pb.name] = [m[r][c] for r in range(4) for c in range(4)]
    return sig

def pose_delta(sig_a, sig_b):
    total = 0.0
    for k in sig_a:
        va = sig_a[k]
        vb = sig_b[k]
        total += sum((a - b) ** 2 for a, b in zip(va, vb))
    return math.sqrt(total)

result = {"mode": mode, "asset": str(asset_path)}

if mode == "blend":
    bpy.ops.wm.open_mainfile(filepath=str(asset_path))
    scene = bpy.context.scene
    objs = [
        {
            "name": o.name,
            "type": o.type,
            "parent": o.parent.name if o.parent else None,
            "dimensions": [round(float(x), 4) for x in o.dimensions],
        }
        for o in bpy.data.objects
    ]
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    mesh = next((o for o in bpy.data.objects if o.type == "MESH"), None)
    bones = [b.name for b in arm.data.bones] if arm else []

    act = arm.animation_data.action if (arm and arm.animation_data) else None
    act_range = [round(float(act.frame_range[0]), 4), round(float(act.frame_range[1]), 4)] if act else None
    key_counts = sorted({len(fc.keyframe_points) for fc in act.fcurves}) if act else []

    # Check live motion across the entire timeline including late frames (330..412)
    checkpoints = [scene.frame_start, 100, 200, 300, 350, 390, scene.frame_end]
    checkpoints = sorted({max(scene.frame_start, min(scene.frame_end, c)) for c in checkpoints})
    motion_deltas = {}
    if arm and len(checkpoints) >= 2:
        prev_f = checkpoints[0]
        prev_sig = bone_pose_signature(arm, prev_f)
        for cur_f in checkpoints[1:]:
            cur_sig = bone_pose_signature(arm, cur_f)
            motion_deltas[f"{prev_f}->{cur_f}"] = round(pose_delta(prev_sig, cur_sig), 5)
            prev_f, prev_sig = cur_f, cur_sig

    # Check material texture nodes & packed image status
    mat_audit = []
    if mesh:
        for mat in mesh.data.materials:
            if mat is None:
                continue
            tex_info = []
            if mat.use_nodes and mat.node_tree:
                for n in mat.node_tree.nodes:
                    if n.type == "TEX_IMAGE":
                        img = n.image
                        tex_info.append({
                            "node": n.name,
                            "image_name": img.name if img else None,
                            "packed": bool(img and img.packed_file is not None),
                            "packed_bytes": len(img.packed_file.data) if (img and img.packed_file) else 0,
                            "size": list(img.size) if img else [0, 0],
                        })
            mat_audit.append({"material": mat.name, "texture_nodes": tex_info})

    result.update({
        "objects": objs,
        "cube_count": sum(1 for o in bpy.data.objects if "cube" in o.name.lower()),
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "fps": scene.render.fps,
        "fps_base": scene.render.fps_base,
        "duration_sec": round((scene.frame_end - scene.frame_start) / (scene.render.fps / scene.render.fps_base), 4),
        "bone_count": len(bones),
        "bones": bones,
        "action_name": act.name if act else None,
        "action_frame_range": act_range,
        "fcurve_count": len(act.fcurves) if act else 0,
        "keyframe_counts_per_fcurve": key_counts,
        "motion_deltas": motion_deltas,
        "materials": mat_audit,
    })

elif mode == "fbx":
    # 1. Raw FBX binary AST inspection via Blender's parse_fbx
    addons_dir = Path(bpy.utils.resource_path("LOCAL")) / "scripts" / "addons_core"
    sys.path.insert(0, str(addons_dir))
    from io_scene_fbx import parse_fbx

    root, fbx_ver = parse_fbx.parse(str(asset_path))
    KTIME = 46186158000.0
    raw_models = []
    raw_geometries = []
    global_times = {}
    stack_times = {}
    take_times = {}

    for elem in root.elems:
        if elem.id == b"GlobalSettings":
            for sub in elem.elems:
                for p in sub.elems:
                    if p.props and p.props[0] in (b"TimeSpanStart", b"TimeSpanStop", b"CustomFrameRate", b"TimeMode"):
                        k = p.props[0].decode()
                        v = p.props[4]
                        global_times[k] = round(v / KTIME, 4) if k.startswith("TimeSpan") else v
        elif elem.id == b"Objects":
            for sub in elem.elems:
                if sub.id == b"Geometry":
                    raw_geometries.append({"name": sub.props[1].split(b"\x00")[0].decode(), "subtype": sub.props[2].decode()})
                elif sub.id == b"Model":
                    raw_models.append({"name": sub.props[1].split(b"\x00")[0].decode(), "subtype": sub.props[2].decode()})
                elif sub.id == b"AnimationStack":
                    for p in sub.elems:
                        for pp in p.elems:
                            if pp.props and pp.props[0] in (b"LocalStart", b"LocalStop", b"ReferenceStart", b"ReferenceStop"):
                                stack_times[pp.props[0].decode()] = round(pp.props[4] / KTIME, 4)
        elif elem.id == b"Takes":
            for sub in elem.elems:
                if sub.id == b"Take":
                    for p in sub.elems:
                        if p.id in (b"LocalTime", b"ReferenceTime"):
                            take_times[p.id.decode()] = [round(x / KTIME, 4) for x in p.props]

    # 2. Clean-scene import into Blender (empty scene, no startup Cube!)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.fbx(filepath=str(asset_path), anim_offset=0.0)

    objs = [
        {
            "name": o.name,
            "type": o.type,
            "parent": o.parent.name if o.parent else None,
            "dimensions": [round(float(x), 4) for x in o.dimensions],
        }
        for o in bpy.data.objects
    ]
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    bones = [b.name for b in arm.data.bones] if arm else []
    act = arm.animation_data.action if (arm and arm.animation_data) else None
    act_range = [round(float(act.frame_range[0]), 4), round(float(act.frame_range[1]), 4)] if act else None

    f_start = int(round(act.frame_range[0])) if act else 1
    f_end = int(round(act.frame_range[1])) if act else 1
    bpy.context.scene.frame_start = f_start
    bpy.context.scene.frame_end = f_end

    checkpoints = sorted({f_start, min(100, f_end), min(200, f_end), min(300, f_end), min(350, f_end), min(390, f_end), f_end})
    motion_deltas = {}
    if arm and len(checkpoints) >= 2:
        prev_f = checkpoints[0]
        prev_sig = bone_pose_signature(arm, prev_f)
        for cur_f in checkpoints[1:]:
            cur_sig = bone_pose_signature(arm, cur_f)
            motion_deltas[f"{prev_f}->{cur_f}"] = round(pose_delta(prev_sig, cur_sig), 5)
            prev_f, prev_sig = cur_f, cur_sig

    result.update({
        "fbx_version": fbx_ver,
        "raw_geometries": raw_geometries,
        "raw_mesh_models": [m for m in raw_models if m["subtype"] == "Mesh"],
        "raw_bone_models": [m["name"] for m in raw_models if m["subtype"] == "LimbNode"],
        "raw_cube_count": sum(1 for m in raw_models if "cube" in m["name"].lower()),
        "global_settings_time": global_times,
        "animation_stack_time": stack_times,
        "take_time": take_times,
        "clean_import_objects": objs,
        "clean_import_cube_count": sum(1 for o in bpy.data.objects if "cube" in o.name.lower()),
        "bone_count": len(bones),
        "bones": bones,
        "action_name": act.name if act else None,
        "action_frame_range_offset0": act_range,
        "fcurve_count": len(act.fcurves) if act else 0,
        "motion_deltas": motion_deltas,
        "materials": [m.name for m in bpy.data.materials],
        "images": [{"name": i.name, "size": list(i.size), "packed": bool(i.packed_file)} for i in bpy.data.images],
    })

out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
"""


def run_blender_audit(blender_bin: Path, mode: str, asset_path: Path, tmp_dir: Path) -> Dict[str, Any]:
    script_file = tmp_dir / "_blender_audit_runner.py"
    out_json = tmp_dir / f"_audit_{mode}.json"
    script_file.write_text(BLENDER_AUDIT_SCRIPT, encoding="utf-8")
    cmd = [
        str(blender_bin),
        "--background",
        "--python",
        str(script_file),
        "--",
        mode,
        str(asset_path),
        str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"Blender {mode} audit failed:\n{proc.stdout}\n{proc.stderr}")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    script_file.unlink(missing_ok=True)
    out_json.unlink(missing_ok=True)
    return data


def verify_bundle(target_dir: Path) -> Dict[str, Any]:
    target_dir = target_dir.resolve()
    failures: list[str] = []
    report: Dict[str, Any] = {"directory": str(target_dir), "status": "PASSED", "checks": {}}

    gltf_path = target_dir / "creature_rigged.gltf"
    blend_path = target_dir / "creature_rigged.blend"
    fbx_path = target_dir / "creature_rigged.fbx"
    obj_path = target_dir / "creature_mesh.obj"
    mtl_path = target_dir / "creature_material.mtl"
    tex_path = target_dir / "creature_diffuse.png"
    bvh_path = target_dir / "creature_animation.bvh"
    khronos_report_path = target_dir / "gltf_validation_report.json"

    # 1. Official Khronos glTF-Validator
    validator_mjs = PROJECT_ROOT / "tools" / "gltf-validator" / "validate.mjs"
    if validator_mjs.exists():
        proc = subprocess.run(
            ["node", str(validator_mjs), str(gltf_path), "--report", str(khronos_report_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        k_data = json.loads(khronos_report_path.read_text(encoding="utf-8"))
        issues = k_data["issues"]
        report["checks"]["khronos_gltf_validator"] = {
            "validator_version": k_data.get("validatorVersion"),
            "errors": issues["numErrors"],
            "warnings": issues["numWarnings"],
            "infos": issues["numInfos"],
            "hints": issues["numHints"],
            "report_file": khronos_report_path.name,
        }
        if issues["numErrors"] != 0 or issues["numWarnings"] != 0:
            failures.append(
                f"Khronos glTF-Validator reported {issues['numErrors']} errors and {issues['numWarnings']} warnings"
            )
    else:
        failures.append("Khronos validator script tools/gltf-validator/validate.mjs not found")

    # 2. Strict internal math validator (IBM cancellation, monotonicity, weights)
    strict_res = validate_gltf_strict(gltf_path, check_ibms=True)
    report["checks"]["strict_gltf_math"] = {
        "valid": strict_res.is_valid,
        "errors": strict_res.errors,
        "warnings": strict_res.warnings,
        "stats": strict_res.stats,
    }
    if not strict_res.is_valid or len(strict_res.warnings) > 0:
        failures.append(f"Strict glTF validator: errors={strict_res.errors}, warnings={strict_res.warnings}")

    # 3. OBJ / MTL / Texture checks
    img = Image.open(tex_path)
    w, h = img.size
    is_pot = (w > 0 and (w & (w - 1)) == 0) and (h > 0 and (h & (h - 1)) == 0)
    mtl_text = mtl_path.read_text(encoding="utf-8") if mtl_path.exists() else ""
    obj_head = "\n".join(obj_path.read_text(encoding="utf-8").splitlines()[:10]) if obj_path.exists() else ""
    report["checks"]["obj_mtl_texture"] = {
        "obj_exists": obj_path.exists(),
        "obj_references_mtl": "mtllib creature_material.mtl" in obj_head,
        "mtl_exists": mtl_path.exists(),
        "mtl_references_png": "map_Kd creature_diffuse.png" in mtl_text,
        "texture_size": [w, h],
        "texture_is_power_of_two": is_pot,
    }
    if not (obj_path.exists() and mtl_path.exists() and "map_Kd creature_diffuse.png" in mtl_text and is_pot):
        failures.append("OBJ/MTL/PNG packaging or power-of-two check failed")

    # 4. BVH check
    bvh_text = bvh_path.read_text(encoding="utf-8")
    bvh_has_412 = "Frames: 412" in bvh_text
    bvh_has_shoulders = "LeftShoulder" in bvh_text and "RightShoulder" in bvh_text
    report["checks"]["bvh"] = {
        "exists": bvh_path.exists(),
        "has_412_frames": bvh_has_412,
        "has_shoulder_joints": bvh_has_shoulders,
    }
    if not (bvh_has_412 and bvh_has_shoulders):
        failures.append("BVH check failed (expected 412 frames and shoulder joints)")

    # 5. Blender .blend & .fbx audits
    settings = get_settings()
    blender_bin = Path(settings.blender_executable) if settings.blender_executable else None
    if blender_bin and blender_bin.exists():
        blend_audit = run_blender_audit(blender_bin, "blend", blend_path, target_dir)
        report["checks"]["blender_blend"] = blend_audit
        if blend_audit["cube_count"] != 0:
            failures.append(f".blend contains {blend_audit['cube_count']} stray Cube(s)")
        if blend_audit["frame_start"] != 1 or blend_audit["frame_end"] != 412 or blend_audit["fps"] != 30:
            failures.append(
                f".blend scene range mismatch: {blend_audit['frame_start']}..{blend_audit['frame_end']} @ {blend_audit['fps']} FPS"
            )
        if blend_audit["action_frame_range"] != [1.0, 412.0]:
            failures.append(f".blend action frame_range is {blend_audit['action_frame_range']}, expected [1.0, 412.0]")
        if any(v <= 1e-4 for v in blend_audit["motion_deltas"].values()):
            failures.append(f".blend has frozen motion interval(s): {blend_audit['motion_deltas']}")
        packed_ok = any(
            tn["packed"] and tn["packed_bytes"] > 0
            for m in blend_audit["materials"]
            for tn in m["texture_nodes"]
        )
        if not packed_ok:
            failures.append(".blend material texture node does NOT use a packed image")
        if "LeftShoulder" not in blend_audit["bones"] or "RightShoulder" not in blend_audit["bones"]:
            failures.append(".blend armature missing shoulder bones")

        fbx_audit = run_blender_audit(blender_bin, "fbx", fbx_path, target_dir)
        report["checks"]["blender_fbx"] = fbx_audit
        if fbx_audit["raw_cube_count"] != 0 or fbx_audit["clean_import_cube_count"] != 0:
            failures.append("FBX contains stray Cube mesh")
        if len(fbx_audit["raw_geometries"]) != 1 or len(fbx_audit["raw_mesh_models"]) != 1:
            failures.append(
                f"FBX should contain exactly 1 geometry/mesh, found {len(fbx_audit['raw_geometries'])}/{len(fbx_audit['raw_mesh_models'])}"
            )
        if fbx_audit["global_settings_time"].get("TimeSpanStop", 0) < 13.7:
            failures.append(f"FBX GlobalSettings.TimeSpanStop too short: {fbx_audit['global_settings_time']}")
        if fbx_audit["animation_stack_time"].get("LocalStop", 0) < 13.7:
            failures.append(f"FBX AnimationStack.LocalStop too short: {fbx_audit['animation_stack_time']}")
        if any(v <= 1e-4 for v in fbx_audit["motion_deltas"].values()):
            failures.append(f"FBX has frozen motion interval(s): {fbx_audit['motion_deltas']}")
    else:
        failures.append("Blender executable not configured/found for .blend and .fbx verification")

    report["failures"] = failures
    report["status"] = "PASSED" if not failures else "FAILED"
    out_report = target_dir / "verification_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all production assets in a creature export folder.")
    parser.add_argument("directory", nargs="?", default="outputs/creature_dwarf", help="Export directory to audit")
    args = parser.parse_args()

    report = verify_bundle(Path(args.directory))
    print("=" * 70)
    print(f" PRODUCTION BUNDLE VERIFICATION: {report['status']}")
    print("=" * 70)
    k = report["checks"].get("khronos_gltf_validator", {})
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
        print(f"   Motion deltas: {bl['motion_deltas']}")
        print(f"   Materials    : {bl['materials']}")
    fbx = report["checks"].get("blender_fbx", {})
    if fbx:
        print(
            f" Unreal/Unity .fbx: raw_meshes={len(fbx['raw_mesh_models'])} | raw_cubes={fbx['raw_cube_count']} | "
            f"clean_import_cubes={fbx['clean_import_cube_count']} | bones={fbx['bone_count']}"
        )
        print(f"   GlobalSettings TimeSpan: {fbx['global_settings_time']}")
        print(f"   AnimStack LocalTime    : {fbx['animation_stack_time']}")
        print(f"   Clean-import action    : {fbx['action_frame_range_offset0']} | deltas={fbx['motion_deltas']}")
    if report["failures"]:
        print("\nFAILURES:")
        for f in report["failures"]:
            print(f"  - {f}")
        return 1
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
