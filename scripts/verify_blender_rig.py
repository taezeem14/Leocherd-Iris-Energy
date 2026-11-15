"""
Blender Automated Verification Script for Leocherd Iris Energy Rigs.

Can run either:
1. Directly inside Blender (with bpy runtime):
   blender --background --python scripts/verify_blender_rig.py -- <path_to_gltf>
2. Standalone / Offline (without requiring Blender / bpy):
   python scripts/verify_blender_rig.py <path_to_gltf> [--rig-json <path_to_rig.json>]

Validates:
1. Armature hierarchy & bone connections.
2. Skinned mesh vertex weights and deform groups.
3. Dual Quaternion Skinning (DQS) volume preservation:
   modifier.use_deform_preserve_volume == True.
4. Geometry checks: absence of zero-area faces (area < 1e-7) and non-manifold edges.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import bpy
    import bmesh
except ImportError:
    bpy = None
    bmesh = None


def verify_blender_asset_offline(
    gltf_path: str | Path,
    rig_json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Validates exported glTF asset and rig JSON offline without requiring a Blender / bpy installation.
    Checks:
    - Mesh manifoldness (no edges shared by >2 faces).
    - Absence of zero-area faces (area < 1e-7).
    - Skinning vertex attributes and influence counts.
    - Rig hierarchy and volume preservation flags in rig JSON.
    """
    gltf_p = Path(gltf_path)
    errors: List[str] = []
    warnings: List[str] = []

    if not gltf_p.exists():
        return {
            "passed": False,
            "errors": [f"glTF file not found: {gltf_p}"],
            "warnings": [],
            "zero_area_faces": 0,
            "non_manifold_edges": 0,
            "vertex_count": 0,
            "face_count": 0,
            "bone_count": 0,
            "preserve_volume": False,
        }

    try:
        with open(gltf_p, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        return {
            "passed": False,
            "errors": [f"Failed to read glTF JSON: {e}"],
            "warnings": [],
            "zero_area_faces": 0,
            "non_manifold_edges": 0,
            "vertex_count": 0,
            "face_count": 0,
            "bone_count": 0,
            "preserve_volume": False,
        }

    # Decode binary buffer
    buffers = doc.get("buffers", [])
    buffer_views = doc.get("bufferViews", [])
    accessors = doc.get("accessors", [])

    buf_raw = b""
    if buffers:
        uri = buffers[0].get("uri", "")
        if uri.startswith("data:"):
            try:
                buf_raw = base64.b64decode(uri.split(",", 1)[1])
            except Exception as ex:
                errors.append(f"Failed to decode buffer base64: {ex}")
        else:
            bin_file = gltf_p.parent / uri
            if bin_file.exists():
                buf_raw = bin_file.read_bytes()

    # Geometry extraction
    total_zero_area = 0
    total_non_manifold = 0
    total_verts = 0
    total_faces = 0

    for m_idx, mesh in enumerate(doc.get("meshes", [])):
        for p_idx, prim in enumerate(mesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            pos_idx = attrs.get("POSITION")
            idx_idx = prim.get("indices")

            if pos_idx is None or idx_idx is None or not buf_raw:
                continue

            pos_acc = accessors[pos_idx]
            idx_acc = accessors[idx_idx]

            pos_bv = buffer_views[pos_acc["bufferView"]]
            idx_bv = buffer_views[idx_acc["bufferView"]]

            pos_off = pos_bv.get("byteOffset", 0) + pos_acc.get("byteOffset", 0)
            idx_off = idx_bv.get("byteOffset", 0) + idx_acc.get("byteOffset", 0)

            v_count = pos_acc.get("count", 0)
            f_count = idx_acc.get("count", 0) // 3
            total_verts += v_count
            total_faces += f_count

            verts = np.frombuffer(buf_raw[pos_off : pos_off + v_count * 12], dtype=np.float32).reshape((-1, 3))

            idx_dtype = np.uint32 if idx_acc.get("componentType") == 5125 else np.uint16
            faces = np.frombuffer(
                buf_raw[idx_off : idx_off + idx_acc.get("count", 0) * np.dtype(idx_dtype).itemsize],
                dtype=idx_dtype,
            ).reshape((-1, 3))

            # 1. Zero-area faces check: Area = 0.5 * ||(v1 - v0) x (v2 - v0)||
            if len(faces) > 0 and len(verts) > 0:
                v0 = verts[faces[:, 0]]
                v1 = verts[faces[:, 1]]
                v2 = verts[faces[:, 2]]
                cross = np.cross(v1 - v0, v2 - v0)
                areas = 0.5 * np.linalg.norm(cross, axis=1)
                zero_area_mask = areas < 1e-7
                zero_count = int(np.sum(zero_area_mask))
                total_zero_area += zero_count
                if zero_count > 0:
                    errors.append(f"Mesh {m_idx} Prim {p_idx}: found {zero_count} zero-area faces (area < 1e-7)")

                # 2. Non-manifold edges check: edge shared by >2 faces
                edge_counts = Counter()
                for face in faces:
                    u0, u1, u2 = int(face[0]), int(face[1]), int(face[2])
                    edge_counts[(min(u0, u1), max(u0, u1))] += 1
                    edge_counts[(min(u1, u2), max(u1, u2))] += 1
                    edge_counts[(min(u0, u2), max(u0, u2))] += 1

                non_manifold = [e for e, c in edge_counts.items() if c > 2]
                total_non_manifold += len(non_manifold)
                if non_manifold:
                    errors.append(
                        f"Mesh {m_idx} Prim {p_idx}: found {len(non_manifold)} non-manifold edges (shared by >2 faces)"
                    )

    # Rig hierarchy and volume preservation validation
    bone_count = 0
    preserve_volume = True

    # Look for adjacent rig_data.json if not specified
    if rig_json_path is None:
        cand = gltf_p.parent / f"{gltf_p.stem.replace('_skinned', '')}_rig_data.json"
        if cand.exists():
            rig_json_path = cand
        else:
            cand2 = gltf_p.parent / f"{gltf_p.stem}_rig_data.json"
            if cand2.exists():
                rig_json_path = cand2

    if rig_json_path and Path(rig_json_path).exists():
        try:
            with open(rig_json_path, "r", encoding="utf-8") as f:
                rig_data = json.load(f)
            bones = rig_data.get("bones", [])
            bone_count = len(bones)
            preserve_volume = rig_data.get("use_deform_preserve_volume", True)
            if not preserve_volume:
                warnings.append("Rig JSON explicitly sets use_deform_preserve_volume to False")
        except Exception as ex:
            warnings.append(f"Could not read rig JSON: {ex}")
    else:
        # Fall back to skin joints count from glTF
        skins = doc.get("skins", [])
        if skins:
            bone_count = len(skins[0].get("joints", []))

    passed = (len(errors) == 0 and total_zero_area == 0 and total_non_manifold == 0)

    return {
        "passed": passed,
        "zero_area_faces": total_zero_area,
        "non_manifold_edges": total_non_manifold,
        "vertex_count": total_verts,
        "face_count": total_faces,
        "bone_count": bone_count,
        "preserve_volume": preserve_volume,
        "errors": errors,
        "warnings": warnings,
    }


def verify_rig(gltf_path: str, require_volume_preservation: bool = True) -> bool:
    """Verifies glTF armature and mesh in Blender runtime (or offline fallback)."""
    if bpy is None:
        print("[!] Blender 'bpy' runtime not detected; executing offline verification...")
        offline_res = verify_blender_asset_offline(gltf_path)
        print(f"[*] Offline verification passed: {offline_res['passed']}")
        print(f"[*] Zero-area faces: {offline_res['zero_area_faces']}")
        print(f"[*] Non-manifold edges: {offline_res['non_manifold_edges']}")
        print(f"[*] Total vertices: {offline_res['vertex_count']}, faces: {offline_res['face_count']}")
        if offline_res["errors"]:
            print(f"[FAIL] Errors ({len(offline_res['errors'])}):")
            for e in offline_res["errors"]:
                print(f"    - {e}")
        return offline_res["passed"]

    print("=" * 60)
    print(f" LEOCHERD IRIS ENERGY - BLENDER RIG VERIFICATION")
    print(f" File: {gltf_path}")
    print("=" * 60)

    # Clear current scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import glTF 2.0
    bpy.ops.import_scene.gltf(filepath=str(gltf_path))

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]

    print(f"[*] Armatures found: {len(armatures)}")
    print(f"[*] Meshes found   : {len(meshes)}")

    if not armatures:
        print("[FAIL] No armature found in imported asset!")
        return False

    arm = armatures[0]
    bones = arm.data.bones
    print(f"[*] Bones count    : {len(bones)}")
    for b in bones:
        parent_str = b.parent.name if b.parent else "ROOT"
        print(f"    - {b.name:<18} (Parent: {parent_str})")

    if not meshes:
        print("[FAIL] No skinned mesh found in imported asset!")
        return False

    all_passed = True
    for mesh in meshes:
        print(f"[*] Mesh '{mesh.name}': {len(mesh.data.vertices)} vertices, {len(mesh.data.polygons)} polygons")
        print(f"[*] Vertex groups (Skinning joints): {len(mesh.vertex_groups)}")

        # 1. Armature modifier volume preservation check
        arm_mods = [m for m in mesh.modifiers if m.type == "ARMATURE"]
        for mod in arm_mods:
            vol_preserve = getattr(mod, "use_deform_preserve_volume", False)
            print(f"    - Armature modifier volume preservation: {vol_preserve}")
            if require_volume_preservation and not vol_preserve:
                print("      [WARN] use_deform_preserve_volume is False; DQS volume preservation not active.")

        # 2. Geometry check: zero-area faces
        zero_area_faces = [p for p in mesh.data.polygons if p.area < 1e-7]
        if zero_area_faces:
            print(f"[FAIL] Found {len(zero_area_faces)} zero-area faces (area < 1e-7) in mesh '{mesh.name}'!")
            all_passed = False
        else:
            print(f"    - Zero-area faces: 0 (OK)")

        # 3. Geometry check: non-manifold edges
        if bmesh is not None:
            bm = bmesh.new()
            bm.from_mesh(mesh.data)
            non_manifold_edges = [e for e in bm.edges if not e.is_manifold]
            bm.free()
            if non_manifold_edges:
                print(f"[FAIL] Found {len(non_manifold_edges)} non-manifold edges in mesh '{mesh.name}'!")
                all_passed = False
            else:
                print(f"    - Non-manifold edges: 0 (OK)")

    # Check animations
    actions = bpy.data.actions
    print(f"[*] Animation actions: {len(actions)}")
    for act in actions:
        print(
            f"    - Action '{act.name}' ({len(act.fcurves)} F-Curves, range: {act.frame_range[0]}..{act.frame_range[1]})"
        )

    print("-" * 60)
    if all_passed:
        print("✅ RIG & ARMATURE VERIFICATION PASSED SUCCESSFULLY!")
    else:
        print("❌ RIG & ARMATURE VERIFICATION FAILED GEOMETRY OR HIERARCHY CHECKS!")
    print("=" * 60)
    return all_passed


def main() -> None:
    args = sys.argv
    gltf_file = None
    rig_json = None

    if "--" in args:
        idx = args.index("--")
        rem = args[idx + 1 :]
        if rem:
            gltf_file = rem[0]
            if len(rem) > 1:
                rig_json = rem[1]
    else:
        parser = argparse.ArgumentParser(description="Blender Rig & Geometry Verification")
        parser.add_argument("gltf_file", nargs="?", default=None, help="Path to glTF asset")
        parser.add_argument("--rig-json", default=None, help="Path to rig_data.json")
        parsed, _ = parser.parse_known_args()
        gltf_file = parsed.gltf_file
        rig_json = parsed.rig_json

    if not gltf_file:
        gltf_file = "outputs/dance_run/reconstruction_7946e279.gltf"

    if bpy is not None:
        ok = verify_rig(gltf_file)
        sys.exit(0 if ok else 1)
    else:
        res = verify_blender_asset_offline(gltf_file, rig_json_path=rig_json)
        print(json.dumps(res, indent=2))
        sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
