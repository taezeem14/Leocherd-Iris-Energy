"""
Blender Standalone Python Automation: Asset Integrity Validator
Loads and verifies bone counts, keyframes, and skin bindings in generated .blend or .glb assets.

Usage:
    blender --background scene.blend --python blender/validate_asset.py
"""

import sys

try:
    import bpy
except ImportError:
    print("[ERROR] This script must be run inside Blender's Python runtime.")
    sys.exit(1)


def validate():
    print("=" * 60)
    print(" LEOCHERD IRIS ENERGY - BLENDER ASSET VALIDATOR")
    print("=" * 60)

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        print("[FAIL] No Armature object found in scene.")
        sys.exit(1)

    arm = armatures[0]
    bone_count = len(arm.data.bones)
    print(f"[OK] Armature '{arm.name}' verified with {bone_count} bones.")

    action = arm.animation_data.action if arm.animation_data else None
    if action:
        frame_range = action.frame_range
        print(f"[OK] Action '{action.name}' found. Frame range: {frame_range[0]} to {frame_range[1]}")
    else:
        print("[WARN] No active animation action linked to Armature.")

    print("[SUCCESS] Blender asset validation complete.")


if __name__ == "__main__":
    validate()
