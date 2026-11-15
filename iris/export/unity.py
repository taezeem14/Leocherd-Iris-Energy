"""
Unity-Compatible Generic Rig Profile & Verification.

Validates that exported glTF 2.0 assets strictly conform to Unity's Generic Rig specifications:
1. Single root joint in the skin hierarchy.
2. Uniform scale [1.0, 1.0, 1.0] across all bones (prevents SkinnedMeshRenderer shear distortion).
3. Maximum 4 bone influences per vertex (JOINTS_0, WEIGHTS_0 VEC4).
4. Strictly normalized, descending sorted non-negative skin weights summing to 1.0 +- 1e-4.
5. Zero-weight joint index sanitization (w == 0 => j == 0).
6. Non-singular, invertible 4x4 bindposes (Mesh.bindposes).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class UnityValidationResult:
    """Validation report for Unity Generic Rig profile conformity."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Unity Generic Rig Conformity: {status}",
            f"Errors ({len(self.errors)}):",
            *[f"  - [ERROR] {e}" for e in self.errors],
            f"Warnings ({len(self.warnings)}):",
            *[f"  - [WARN]  {w}" for w in self.warnings],
            f"Stats: {json.dumps(self.stats, indent=2)}",
        ]
        return "\n".join(lines)


class UnityRigProfile:
    """Unity-compatible Generic Rig profile verifier."""

    @classmethod
    def validate(cls, gltf_path: str | Path) -> UnityValidationResult:
        path = Path(gltf_path)
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {}

        if not path.exists():
            return UnityValidationResult(
                is_valid=False,
                errors=[f"File does not exist: {path}"],
                warnings=[],
                stats={},
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            return UnityValidationResult(
                is_valid=False,
                errors=[f"Failed to parse glTF JSON: {e}"],
                warnings=[],
                stats={},
            )

        nodes = doc.get("nodes", [])
        skins = doc.get("skins", [])
        accessors = doc.get("accessors", [])
        buffer_views = doc.get("bufferViews", [])
        buffers = doc.get("buffers", [])

        stats["node_count"] = len(nodes)
        stats["skin_count"] = len(skins)

        if not skins:
            errors.append("Unity Generic Rig requires at least one skin definition in glTF asset.")
            return UnityValidationResult(is_valid=False, errors=errors, warnings=warnings, stats=stats)

        # Decode buffer for data access
        if not buffers:
            errors.append("No buffers found in glTF asset.")
            return UnityValidationResult(is_valid=False, errors=errors, warnings=warnings, stats=stats)

        buf_raw = b""
        uri = buffers[0].get("uri", "")
        if uri.startswith("data:"):
            try:
                buf_raw = base64.b64decode(uri.split(",", 1)[1])
            except Exception as ex:
                errors.append(f"Failed to decode buffer base64: {ex}")
        else:
            bin_file = path.parent / uri
            if bin_file.exists():
                buf_raw = bin_file.read_bytes()
            else:
                errors.append(f"Referenced binary file not found: {bin_file}")

        skin = skins[0]
        joint_indices = skin.get("joints", [])
        stats["joint_count"] = len(joint_indices)

        if not joint_indices:
            errors.append("Skin contains no joints.")
            return UnityValidationResult(is_valid=False, errors=errors, warnings=warnings, stats=stats)

        # 1. Single Root Joint Check
        # Build child-to-parent mapping among skin joints
        joint_set = set(joint_indices)
        joint_parents: Dict[int, int] = {}
        node_names = set()

        for j_idx in joint_indices:
            if j_idx < 0 or j_idx >= len(nodes):
                errors.append(f"Joint node index {j_idx} is out of bounds (total nodes: {len(nodes)}).")
                continue
            n = nodes[j_idx]
            name = n.get("name", "")
            if not name:
                warnings.append(f"Joint node {j_idx} has empty or missing name.")
            elif name in node_names:
                warnings.append(f"Duplicate joint node name '{name}' detected.")
            node_names.add(name)

            for c_idx in n.get("children", []):
                if c_idx in joint_set:
                    joint_parents[c_idx] = j_idx

        # Roots are joints without parent within the skin joint set
        skin_roots = [j for j in joint_indices if j not in joint_parents]
        if len(skin_roots) == 0:
            errors.append("Skin hierarchy contains cycles: no root joint found.")
        elif len(skin_roots) > 1:
            errors.append(
                f"Unity Generic Rig requires exactly 1 root joint, but found {len(skin_roots)}: "
                f"{[nodes[r].get('name', r) for r in skin_roots]}"
            )
        else:
            stats["root_joint"] = nodes[skin_roots[0]].get("name", skin_roots[0])

        # 2. Uniform Scale Constraint [1.0, 1.0, 1.0]
        for j_idx in joint_indices:
            if j_idx >= len(nodes):
                continue
            n = nodes[j_idx]
            scale = n.get("scale", [1.0, 1.0, 1.0])
            if not np.allclose(scale, [1.0, 1.0, 1.0], atol=1e-4):
                errors.append(
                    f"Joint node '{n.get('name', j_idx)}' has non-uniform scale {scale}; "
                    f"Unity Generic Rig requires uniform [1.0, 1.0, 1.0] scale."
                )

        # 3. Valid Bindposes (Mesh.bindposes)
        ibm_idx = skin.get("inverseBindMatrices")
        if ibm_idx is None or ibm_idx >= len(accessors):
            errors.append("Skin missing inverseBindMatrices accessor.")
        else:
            ibm_acc = accessors[ibm_idx]
            if ibm_acc.get("type") != "MAT4":
                errors.append(f"IBM accessor type is '{ibm_acc.get('type')}', expected 'MAT4'.")
            if ibm_acc.get("componentType") != 5126:
                errors.append(f"IBM accessor componentType is {ibm_acc.get('componentType')}, expected 5126 (FLOAT).")

            if ibm_acc.get("count") != len(joint_indices):
                errors.append(
                    f"IBM count {ibm_acc.get('count')} does not match joints count {len(joint_indices)}."
                )

            # Read matrices
            bv_idx = ibm_acc.get("bufferView")
            if bv_idx is not None and bv_idx < len(buffer_views) and buf_raw:
                bv = buffer_views[bv_idx]
                offset = bv.get("byteOffset", 0) + ibm_acc.get("byteOffset", 0)
                ibm_bytes = buf_raw[offset : offset + len(joint_indices) * 64]
                ibm_mats = np.frombuffer(ibm_bytes, dtype=np.float32).reshape((len(joint_indices), 4, 4))

                for j_pos, node_idx in enumerate(joint_indices):
                    mat = ibm_mats[j_pos].T  # glTF column-major
                    if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
                        errors.append(f"Joint '{nodes[node_idx].get('name', node_idx)}' bindpose contains NaN or Inf.")
                        continue
                    det = float(np.linalg.det(mat))
                    if abs(det) < 1e-6:
                        errors.append(
                            f"Joint '{nodes[node_idx].get('name', node_idx)}' bindpose is singular (det={det:.2e})."
                        )

        # 4. Max 4 Influences per Vertex & Sanitization
        meshes = doc.get("meshes", [])
        total_verts = 0
        for m_idx, mesh in enumerate(meshes):
            for p_idx, prim in enumerate(mesh.get("primitives", [])):
                attrs = prim.get("attributes", {})
                pos_idx = attrs.get("POSITION")
                if pos_idx is not None and pos_idx < len(accessors):
                    v_count = accessors[pos_idx].get("count", 0)
                    total_verts += v_count

                if "JOINTS_1" in attrs or "WEIGHTS_1" in attrs:
                    warnings.append(
                        f"Mesh {m_idx} Prim {p_idx} defines JOINTS_1/WEIGHTS_1 (>4 influences). "
                        f"Unity standard SkinnedMeshRenderer clamps to 4 influences."
                    )

                j_idx = attrs.get("JOINTS_0")
                w_idx = attrs.get("WEIGHTS_0")
                if j_idx is None or w_idx is None:
                    continue

                j_acc = accessors[j_idx]
                w_acc = accessors[w_idx]

                if j_acc.get("type") != "VEC4" or w_acc.get("type") != "VEC4":
                    errors.append(
                        f"Mesh {m_idx} Prim {p_idx}: JOINTS_0/WEIGHTS_0 must be VEC4 (max 4 influences)."
                    )

                if buf_raw:
                    j_bv = buffer_views[j_acc["bufferView"]]
                    w_bv = buffer_views[w_acc["bufferView"]]
                    j_off = j_bv.get("byteOffset", 0) + j_acc.get("byteOffset", 0)
                    w_off = w_bv.get("byteOffset", 0) + w_acc.get("byteOffset", 0)

                    j_dtype = np.uint8 if j_acc.get("componentType") == 5121 else np.uint16
                    j_data = np.frombuffer(
                        buf_raw[j_off : j_off + j_acc["count"] * 4 * np.dtype(j_dtype).itemsize],
                        dtype=j_dtype,
                    ).reshape((-1, 4))

                    w_data = np.frombuffer(
                        buf_raw[w_off : w_off + w_acc["count"] * 16],
                        dtype=np.float32,
                    ).reshape((-1, 4))

                    # Non-negativity
                    if np.any(w_data < -1e-6):
                        errors.append(f"Mesh {m_idx} Prim {p_idx}: negative skin weights detected.")

                    # Sum to 1.0 +- 1e-4
                    sums = np.sum(w_data, axis=1)
                    max_err = float(np.max(np.abs(sums - 1.0)))
                    if max_err > 1e-4:
                        errors.append(
                            f"Mesh {m_idx} Prim {p_idx}: weights do not sum to 1.0 (max deviation: {max_err:.2e})."
                        )

                    # Descending sort
                    for k in range(3):
                        if np.any(w_data[:, k] < w_data[:, k + 1] - 1e-5):
                            warnings.append(
                                f"Mesh {m_idx} Prim {p_idx}: weights at slot {k} are not sorted descending."
                            )

                    # Zero weight => joint index 0
                    zero_w = w_data == 0.0
                    bad_zero = (j_data != 0) & zero_w
                    if np.sum(bad_zero) > 0:
                        warnings.append(
                            f"Mesh {m_idx} Prim {p_idx}: {int(np.sum(bad_zero))} zero-weight influences have non-zero joint index."
                        )

        stats["total_vertices"] = total_verts
        is_valid = len(errors) == 0

        return UnityValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )


def validate_unity_generic_rig(gltf_path: str | Path) -> UnityValidationResult:
    """Convenience function validating glTF asset against Unity Generic Rig requirements."""
    return UnityRigProfile.validate(gltf_path)
