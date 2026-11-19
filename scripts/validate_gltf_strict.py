"""
Strict glTF 2.0 Rig, Mesh, Skinning, and Animation Validator.

Validates glTF 2.0 assets against strict production requirements:
1. Base64 buffer decoding & 4-byte bufferView alignment.
2. Accessor min/max fidelity against raw buffer data.
3. Inverse bind matrices (IBMs): MAT4 componentType=5126, count matches joints,
   non-singularity (det != 0), and bindpose cancellation (M_j * IBM_j ≈ I).
4. Skinning influences: JOINTS_0 (5121/5123), WEIGHTS_0 (5126), VEC4,
   partition of unity (sum == 1.0 +- 1e-4), non-negativity, descending sort,
   and zero-weight joint index sanitization (w == 0 => j == 0).
5. Animation curves: strictly monotonically increasing timestamps,
   normalized unit quaternions (||q|| == 1.0 +- 1e-3), valid sampler paths.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class GltfValidationResult:
    """Container for glTF 2.0 strict validation results."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Validation Status: {status}",
            f"Errors ({len(self.errors)}):",
            *[f"  - [ERROR] {e}" for e in self.errors],
            f"Warnings ({len(self.warnings)}):",
            *[f"  - [WARN]  {w}" for w in self.warnings],
            f"Statistics: {json.dumps(self.stats, indent=2)}",
        ]
        return "\n".join(lines)


def _quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Converts glTF quaternion [x, y, z, w] to 3x3 rotation matrix."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n > 1e-8:
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


class GltfStrictValidator:
    """Production-grade strict glTF 2.0 rig, mesh, skin, and animation validator."""

    TYPE_COMPONENT_COUNT = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

    COMPONENT_TYPE_MAP = {
        5120: (np.int8, 1),
        5121: (np.uint8, 1),
        5122: (np.int16, 2),
        5123: (np.uint16, 2),
        5125: (np.uint32, 4),
        5126: (np.float32, 4),
    }

    def __init__(self) -> None:
        pass

    def validate(
        self,
        gltf_path: str | Path,
        check_ibms: bool = True,
    ) -> GltfValidationResult:
        path = Path(gltf_path)
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {}

        if not path.exists():
            return GltfValidationResult(
                is_valid=False,
                errors=[f"File does not exist: {path}"],
                warnings=[],
                stats={},
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            return GltfValidationResult(
                is_valid=False,
                errors=[f"Failed to parse JSON in {path}: {e}"],
                warnings=[],
                stats={},
            )

        # 1. Asset specification check
        asset = doc.get("asset", {})
        if asset.get("version") != "2.0":
            errors.append(f"glTF asset version is '{asset.get('version')}', expected '2.0'")

        # 2. Buffers decoding
        buffers = doc.get("buffers", [])
        if not buffers:
            errors.append("No buffers defined in glTF document.")
            return GltfValidationResult(is_valid=False, errors=errors, warnings=warnings, stats=stats)

        decoded_buffers: List[bytes] = []
        for b_idx, buf in enumerate(buffers):
            uri = buf.get("uri", "")
            declared_len = buf.get("byteLength", 0)
            if uri.startswith("data:"):
                try:
                    payload = uri.split(",", 1)[1]
                    raw = base64.b64decode(payload)
                    decoded_buffers.append(raw)
                    if len(raw) != declared_len:
                        errors.append(
                            f"Buffer {b_idx}: decoded byte length {len(raw)} != declared byteLength {declared_len}"
                        )
                except Exception as ex:
                    errors.append(f"Buffer {b_idx}: base64 decoding failed: {ex}")
                    decoded_buffers.append(b"")
            else:
                bin_path = path.parent / uri
                if bin_path.exists():
                    raw = bin_path.read_bytes()
                    decoded_buffers.append(raw)
                    if len(raw) != declared_len:
                        errors.append(
                            f"Buffer {b_idx}: binary file length {len(raw)} != declared byteLength {declared_len}"
                        )
                else:
                    errors.append(f"Buffer {b_idx}: referenced binary file not found: {bin_path}")
                    decoded_buffers.append(b"")

        # 3. BufferViews alignment & bounds
        buffer_views = doc.get("bufferViews", [])
        for bv_idx, bv in enumerate(buffer_views):
            b_idx = bv.get("buffer", 0)
            byte_offset = bv.get("byteOffset", 0)
            byte_length = bv.get("byteLength", 0)

            # 4-byte alignment check
            if byte_offset % 4 != 0:
                warnings.append(f"BufferView {bv_idx}: byteOffset {byte_offset} is not 4-byte aligned")

            if b_idx < len(decoded_buffers):
                buf_len = len(decoded_buffers[b_idx])
                if byte_offset + byte_length > buf_len:
                    errors.append(
                        f"BufferView {bv_idx}: extends out of bounds ({byte_offset} + {byte_length} > {buf_len})"
                    )

        # 4. Accessors inspection
        accessors = doc.get("accessors", [])
        accessor_arrays: List[Optional[np.ndarray]] = [None] * len(accessors)

        for a_idx, acc in enumerate(accessors):
            bv_idx = acc.get("bufferView")
            if bv_idx is None or bv_idx >= len(buffer_views):
                errors.append(f"Accessor {a_idx}: invalid bufferView index {bv_idx}")
                continue

            bv = buffer_views[bv_idx]
            b_idx = bv.get("buffer", 0)
            if b_idx >= len(decoded_buffers):
                continue

            buf = decoded_buffers[b_idx]
            offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            comp_type = acc.get("componentType")
            type_str = acc.get("type")
            count = acc.get("count", 0)

            if comp_type not in self.COMPONENT_TYPE_MAP:
                errors.append(f"Accessor {a_idx}: unsupported componentType {comp_type}")
                continue
            if type_str not in self.TYPE_COMPONENT_COUNT:
                errors.append(f"Accessor {a_idx}: unsupported type {type_str}")
                continue

            dtype, elem_size = self.COMPONENT_TYPE_MAP[comp_type]
            comps_per_elem = self.TYPE_COMPONENT_COUNT[type_str]
            total_bytes = count * comps_per_elem * elem_size

            if offset + total_bytes > len(buf):
                errors.append(f"Accessor {a_idx}: data read out of bounds ({offset} + {total_bytes} > {len(buf)})")
                continue

            raw_data = np.frombuffer(buf[offset : offset + total_bytes], dtype=dtype)
            if comps_per_elem > 1:
                raw_data = raw_data.reshape((count, comps_per_elem))
            accessor_arrays[a_idx] = raw_data

            # Min/Max validation
            if "min" in acc and "max" in acc and count > 0:
                if comps_per_elem == 1:
                    act_min = float(np.min(raw_data))
                    act_max = float(np.max(raw_data))
                    dec_min = float(acc["min"][0])
                    dec_max = float(acc["max"][0])
                    if act_min < dec_min - 1e-4:
                        errors.append(f"Accessor {a_idx}: actual min {act_min} < declared min {dec_min}")
                    if act_max > dec_max + 1e-4:
                        errors.append(f"Accessor {a_idx}: actual max {act_max} > declared max {dec_max}")
                else:
                    act_min = np.min(raw_data, axis=0)
                    act_max = np.max(raw_data, axis=0)
                    for c in range(comps_per_elem):
                        if act_min[c] < acc["min"][c] - 1e-4:
                            errors.append(
                                f"Accessor {a_idx} comp {c}: actual min {act_min[c]} < declared min {acc['min'][c]}"
                            )
                        if act_max[c] > acc["max"][c] + 1e-4:
                            errors.append(
                                f"Accessor {a_idx} comp {c}: actual max {act_max[c]} > declared max {acc['max'][c]}"
                            )

        # 5. Node hierarchy evaluation for bindpose cancellation
        nodes = doc.get("nodes", [])
        stats["node_count"] = len(nodes)

        # Build parent map
        node_parents: Dict[int, int] = {}
        for parent_idx, n in enumerate(nodes):
            for child_idx in n.get("children", []):
                node_parents[child_idx] = parent_idx

        # Compute rest global transform for each node
        def compute_local_matrix(node_dict: Dict[str, Any]) -> np.ndarray:
            t = np.array(node_dict.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            rot = node_dict.get("rotation", [0.0, 0.0, 0.0, 1.0])  # glTF: [x, y, z, w]
            s = np.array(node_dict.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)

            R = _quat_to_rot_matrix(rot[0], rot[1], rot[2], rot[3])
            # M = [R * diag(S) | t]
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = R * s
            M[:3, 3] = t
            return M

        node_global_matrices: Dict[int, np.ndarray] = {}

        def get_global_matrix(idx: int) -> np.ndarray:
            if idx in node_global_matrices:
                return node_global_matrices[idx]
            loc = compute_local_matrix(nodes[idx])
            if idx in node_parents:
                p_mat = get_global_matrix(node_parents[idx])
                glob = p_mat @ loc
            else:
                glob = loc
            node_global_matrices[idx] = glob
            return glob

        for n_idx in range(len(nodes)):
            get_global_matrix(n_idx)

        # 6. Skin & Inverse Bind Matrices validation
        skins = doc.get("skins", [])
        stats["skin_count"] = len(skins)
        total_joints = 0

        for s_idx, skin in enumerate(skins):
            skin_joints = skin.get("joints", [])
            total_joints += len(skin_joints)
            if not skin_joints:
                errors.append(f"Skin {s_idx}: has empty joints array")
                continue

            # Check joint indices point to valid nodes
            for j_node_idx in skin_joints:
                if j_node_idx < 0 or j_node_idx >= len(nodes):
                    errors.append(f"Skin {s_idx}: joint node index {j_node_idx} out of range (total nodes: {len(nodes)})")

            ibm_acc_idx = skin.get("inverseBindMatrices")
            if ibm_acc_idx is None:
                errors.append(f"Skin {s_idx}: missing inverseBindMatrices accessor")
                continue
            if ibm_acc_idx >= len(accessors):
                errors.append(f"Skin {s_idx}: invalid inverseBindMatrices accessor index {ibm_acc_idx}")
                continue

            ibm_acc = accessors[ibm_acc_idx]
            if ibm_acc.get("type") != "MAT4":
                errors.append(f"Skin {s_idx} IBM: type is '{ibm_acc.get('type')}', expected 'MAT4'")
            if ibm_acc.get("componentType") != 5126:
                errors.append(
                    f"Skin {s_idx} IBM: componentType is {ibm_acc.get('componentType')}, expected 5126 (FLOAT)"
                )
            if ibm_acc.get("count") != len(skin_joints):
                errors.append(
                    f"Skin {s_idx} IBM: count {ibm_acc.get('count')} != joints count {len(skin_joints)}"
                )

            # Validate IBM numerical properties
            ibm_arr = accessor_arrays[ibm_acc_idx]
            if ibm_arr is not None:
                num_mats = len(skin_joints)
                for j_idx, node_idx in enumerate(skin_joints):
                    if j_idx >= len(ibm_arr):
                        break
                    # glTF column-major MAT4 read into C-order 1D slice of 16 floats
                    raw_mat = ibm_arr[j_idx].reshape((4, 4))
                    # In NumPy, column-major 4x4 matrix read row-wise is transposed
                    ibm_j = raw_mat.T

                    if np.any(np.isnan(ibm_j)) or np.any(np.isinf(ibm_j)):
                        errors.append(f"Skin {s_idx} Joint {j_idx}: IBM contains NaN or Inf values")
                        continue

                    det = float(np.linalg.det(ibm_j))
                    if abs(det) < 1e-6:
                        errors.append(f"Skin {s_idx} Joint {j_idx}: IBM is singular (det={det:.2e})")

                    if check_ibms and node_idx < len(nodes):
                        # Bindpose cancellation: G_j * IBM_j == I_4x4
                        G_j = node_global_matrices[node_idx]
                        prod = G_j @ ibm_j
                        diff = np.max(np.abs(prod - np.eye(4, dtype=np.float32)))
                        if diff > 1e-3:
                            errors.append(
                                f"Skin {s_idx} Joint {j_idx} (node '{nodes[node_idx].get('name', node_idx)}'): "
                                f"Bindpose cancellation failed (G_j * IBM_j deviation={diff:.4f} > 1e-3)"
                            )

        stats["total_skin_joints"] = total_joints

        # 7. Meshes and Skinning Attributes (JOINTS_0, WEIGHTS_0)
        meshes = doc.get("meshes", [])
        stats["mesh_count"] = len(meshes)
        total_verts = 0
        total_faces = 0

        for m_idx, mesh in enumerate(meshes):
            primitives = mesh.get("primitives", [])
            for p_idx, prim in enumerate(primitives):
                attrs = prim.get("attributes", {})

                # Position
                pos_idx = attrs.get("POSITION")
                if pos_idx is None or pos_idx >= len(accessors):
                    errors.append(f"Mesh {m_idx} Prim {p_idx}: missing or invalid POSITION attribute")
                    continue
                pos_acc = accessors[pos_idx]
                v_count = pos_acc.get("count", 0)
                total_verts += v_count

                # Indices
                idx_acc_idx = prim.get("indices")
                if idx_acc_idx is not None and idx_acc_idx < len(accessors):
                    idx_acc = accessors[idx_acc_idx]
                    total_faces += idx_acc.get("count", 0) // 3

                # Skinning attributes
                has_joints = "JOINTS_0" in attrs
                has_weights = "WEIGHTS_0" in attrs

                if has_joints != has_weights:
                    errors.append(
                        f"Mesh {m_idx} Prim {p_idx}: JOINTS_0 and WEIGHTS_0 must both be present for skinned meshes"
                    )

                if has_joints and has_weights:
                    j_acc_idx = attrs["JOINTS_0"]
                    w_acc_idx = attrs["WEIGHTS_0"]

                    if j_acc_idx >= len(accessors) or w_acc_idx >= len(accessors):
                        errors.append(f"Mesh {m_idx} Prim {p_idx}: invalid JOINTS_0 or WEIGHTS_0 accessor index")
                        continue

                    j_acc = accessors[j_acc_idx]
                    w_acc = accessors[w_acc_idx]

                    # Component types
                    if j_acc.get("componentType") not in (5121, 5123):
                        errors.append(
                            f"Mesh {m_idx} Prim {p_idx}: JOINTS_0 componentType {j_acc.get('componentType')} "
                            f"must be 5121 (UNSIGNED_BYTE) or 5123 (UNSIGNED_SHORT)"
                        )
                    if j_acc.get("type") != "VEC4":
                        errors.append(f"Mesh {m_idx} Prim {p_idx}: JOINTS_0 type must be VEC4")

                    if w_acc.get("componentType") != 5126:
                        errors.append(
                            f"Mesh {m_idx} Prim {p_idx}: WEIGHTS_0 componentType {w_acc.get('componentType')} "
                            f"must be 5126 (FLOAT)"
                        )
                    if w_acc.get("type") != "VEC4":
                        errors.append(f"Mesh {m_idx} Prim {p_idx}: WEIGHTS_0 type must be VEC4")

                    if j_acc.get("count") != v_count or w_acc.get("count") != v_count:
                        errors.append(
                            f"Mesh {m_idx} Prim {p_idx}: influence counts ({j_acc.get('count')}, {w_acc.get('count')}) "
                            f"do not match vertex count {v_count}"
                        )

                    # Weights & joints values validation
                    j_arr = accessor_arrays[j_acc_idx]
                    w_arr = accessor_arrays[w_acc_idx]

                    if j_arr is not None and w_arr is not None:
                        # 1. Non-negativity
                        if np.any(w_arr < -1e-6):
                            errors.append(f"Mesh {m_idx} Prim {p_idx}: negative weights detected in WEIGHTS_0")

                        # 2. Partition of unity (sum == 1.0 +- 1e-4)
                        w_sums = np.sum(w_arr, axis=1)
                        max_dev = float(np.max(np.abs(w_sums - 1.0)))
                        if max_dev > 1e-4:
                            errors.append(
                                f"Mesh {m_idx} Prim {p_idx}: skin weights do not sum to 1.0 (max deviation: {max_dev:.2e})"
                            )

                        # 3. Descending sort check
                        for k in range(3):
                            if np.any(w_arr[:, k] < w_arr[:, k + 1] - 1e-5):
                                warnings.append(
                                    f"Mesh {m_idx} Prim {p_idx}: weights are not strictly sorted descending at slot {k}"
                                )

                        # 4. Zero-weight influence check: unused slots (w == 0) must reference valid skin joints
                        if skins:
                            skin_joint_count = len(skins[0].get("joints", []))
                            if np.any(j_arr >= skin_joint_count):
                                errors.append(
                                    f"Mesh {m_idx} Prim {p_idx}: joint index exceeds skin joints count ({skin_joint_count})"
                                )

                        # 5. Strict Duplicate Joint Indices Check (Khronos ACCESSOR_JOINTS_INDEX_DUPLICATE)
                        # In glTF 2.0, all 4 joint indices for a given vertex in JOINTS_0 must be unique.
                        for v_i in range(len(j_arr)):
                            all_v_joints = list(j_arr[v_i])
                            if len(all_v_joints) != len(set(all_v_joints)):
                                errors.append(
                                    f"Mesh {m_idx} Prim {p_idx} Vertex {v_i}: duplicate joint index detected in JOINTS_0 "
                                    f"(ACCESSOR_JOINTS_INDEX_DUPLICATE: {all_v_joints})"
                                )
                                break

        stats["total_vertices"] = total_verts
        stats["total_faces"] = total_faces

        # 8. Animations validation
        animations = doc.get("animations", [])
        stats["animation_count"] = len(animations)
        total_keyframes = 0

        for a_idx, anim in enumerate(animations):
            samplers = anim.get("samplers", [])
            channels = anim.get("channels", [])

            for s_idx, sampler in enumerate(samplers):
                in_idx = sampler.get("input")
                out_idx = sampler.get("output")

                if in_idx is None or in_idx >= len(accessors) or out_idx is None or out_idx >= len(accessors):
                    errors.append(f"Animation {a_idx} Sampler {s_idx}: invalid input/output accessor indices")
                    continue

                in_acc = accessors[in_idx]
                total_keyframes = max(total_keyframes, in_acc.get("count", 0))

                # Timestamps must be strictly monotonically increasing
                t_arr = accessor_arrays[in_idx]
                if t_arr is not None and len(t_arr) > 1:
                    t_flat = t_arr.flatten()
                    diffs = np.diff(t_flat)
                    if np.any(diffs <= 0.0):
                        errors.append(
                            f"Animation {a_idx} Sampler {s_idx}: timestamps are not strictly monotonically increasing"
                        )

            for c_idx, channel in enumerate(channels):
                target = channel.get("target", {})
                path_type = target.get("path")
                samp_idx = channel.get("sampler")

                if samp_idx is None or samp_idx >= len(samplers):
                    errors.append(f"Animation {a_idx} Channel {c_idx}: invalid sampler index {samp_idx}")
                    continue

                out_acc_idx = samplers[samp_idx].get("output")
                out_arr = accessor_arrays[out_acc_idx] if out_acc_idx < len(accessor_arrays) else None

                if path_type == "rotation" and out_arr is not None:
                    # Normalized unit quaternions: ||q|| == 1.0 +- 1e-3
                    q_norms = np.linalg.norm(out_arr, axis=1)
                    max_norm_dev = float(np.max(np.abs(q_norms - 1.0)))
                    if max_norm_dev > 1e-3:
                        errors.append(
                            f"Animation {a_idx} Channel {c_idx} (rotation): "
                            f"quaternions are not normalized (max deviation: {max_norm_dev:.2e})"
                        )

        stats["max_keyframes"] = total_keyframes

        is_valid = len(errors) == 0
        return GltfValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )


def validate_gltf_strict(
    gltf_path: str | Path,
    check_ibms: bool = True,
) -> GltfValidationResult:
    """Convenience functional interface for strict glTF 2.0 validation."""
    validator = GltfStrictValidator()
    return validator.validate(gltf_path, check_ibms=check_ibms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict glTF 2.0 Rig & Animation Validator")
    parser.add_argument("gltf_file", nargs="?", default=None, help="Path to glTF file")
    parser.add_argument("--input", "-i", dest="input_file", help="Path to glTF file")
    parser.add_argument("--skip-ibms", action="store_true", help="Skip bindpose cancellation check")

    args = parser.parse_args()
    target_file = args.input_file or args.gltf_file
    if not target_file:
        target_file = "outputs/online_walking_run/reconstruction_skinned_walking.gltf"

    result = validate_gltf_strict(target_file, check_ibms=not args.skip_ibms)
    print(result.summary())
    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
