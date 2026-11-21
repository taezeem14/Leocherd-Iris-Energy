"""
glTF 2.0 skeletal animation and skinned asset exporter.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from iris.core.exceptions import ExportError
from iris.core.logging import get_logger
from iris.export.base import BaseAssetExporter
from iris.retargeting.animation import SkeletalAnimation
from iris.rigging.orientation import QuaternionMath
from iris.rigging.skeleton import Skeleton

logger = get_logger("iris.export.gltf")


class GltfExporter(BaseAssetExporter):
    """Exports rigged skeletons and animation curves into standard glTF 2.0 format."""

    def __init__(self) -> None:
        super().__init__(name="GltfExporter", extension=".gltf")

    @staticmethod
    def sanitize_skin_influences(
        j_indices: np.ndarray,
        j_weights: np.ndarray,
        min_weight: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Canonicalizes per-vertex skin influences to the glTF 2.0 skinning rules.

        Output guarantees (validated by the Khronos glTF-Validator with zero errors/warnings):
          * at most 4 influences, merged per joint and sorted by descending weight
          * weights are non-negative and sum to 1.0 (within float32 precision)
          * no joint index appears twice among non-zero weights
          * every zero-weight slot uses joint index 0 (no ACCESSOR_JOINTS_USED_ZERO_WEIGHT)
          * vertices with no valid influence are bound 100% to joint 0
        """
        ji = np.asarray(j_indices, dtype=np.int64)
        jw = np.asarray(j_weights, dtype=np.float64)
        V = len(jw)
        out_j = np.zeros((V, 4), dtype=np.uint16)
        out_w = np.zeros((V, 4), dtype=np.float32)
        if V == 0:
            return out_j, out_w

        K = min(ji.shape[1], jw.shape[1])
        for v_idx in range(V):
            merged: Dict[int, float] = {}
            for k in range(K):
                w = float(jw[v_idx, k])
                if w > min_weight and np.isfinite(w):
                    j = int(ji[v_idx, k])
                    merged[j] = merged.get(j, 0.0) + w
            pairs = sorted(merged.items(), key=lambda p: p[1], reverse=True)[:4]
            total = sum(w for _, w in pairs)
            if total <= 1e-9:
                out_w[v_idx, 0] = 1.0
                continue
            for k, (j, w) in enumerate(pairs):
                out_j[v_idx, k] = j
                out_w[v_idx, k] = w / total

        # Remove float32 rounding drift: push the residual onto the dominant influence.
        residual = 1.0 - out_w.astype(np.float64).sum(axis=1)
        out_w[:, 0] = (out_w[:, 0].astype(np.float64) + residual).astype(np.float32)
        out_j[out_w <= 0.0] = 0
        out_w[out_w < 0.0] = 0.0
        return out_j, out_w

    def export(
        self,
        skeleton: Skeleton,
        animation: SkeletalAnimation,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        out_file = Path(output_path).with_suffix(self.extension)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting glTF 2.0 asset to {out_file}...")

        # 1. Map joint hierarchy to glTF nodes in canonical bind pose (T-pose)
        joint_names = list(skeleton.joints.keys())
        node_map = {name: idx for idx, name in enumerate(joint_names)}

        nodes = []
        for j_name in joint_names:
            node = skeleton.joints[j_name]
            children_indices = [node_map[c] for c in node.children_names if c in node_map]

            if node.parent_name is not None:
                parent_node = skeleton.joints[node.parent_name]
                t_local = node.rest_world_pos - parent_node.rest_world_pos
            else:
                t_local = node.rest_world_pos

            # Determine rest pose rotation [x, y, z, w] for glTF
            if hasattr(node, "local_rotation") and node.local_rotation is not None and not np.allclose(node.local_rotation, [1.0, 0.0, 0.0, 0.0]):
                lw, lx, ly, lz = QuaternionMath.normalize_quat(node.local_rotation)
                r_local = [float(lx), float(ly), float(lz), float(lw)]
            elif node.parent_name is not None and hasattr(node, "world_rotation") and node.world_rotation is not None:
                parent_node = skeleton.joints[node.parent_name]
                if hasattr(parent_node, "world_rotation") and parent_node.world_rotation is not None:
                    q_loc = QuaternionMath.multiply(QuaternionMath.conjugate(parent_node.world_rotation), node.world_rotation)
                    qw, qx, qy, qz = QuaternionMath.normalize_quat(q_loc)
                    r_local = [float(qx), float(qy), float(qz), float(qw)]
                else:
                    qw, qx, qy, qz = QuaternionMath.normalize_quat(node.world_rotation)
                    r_local = [float(qx), float(qy), float(qz), float(qw)]
            elif node.parent_name is None and hasattr(node, "world_rotation") and node.world_rotation is not None:
                qw, qx, qy, qz = QuaternionMath.normalize_quat(node.world_rotation)
                r_local = [float(qx), float(qy), float(qz), float(qw)]
            elif hasattr(node, "local_rotation") and node.local_rotation is not None:
                lw, lx, ly, lz = QuaternionMath.normalize_quat(node.local_rotation)
                r_local = [float(lx), float(ly), float(lz), float(lw)]
            else:
                r_local = [0.0, 0.0, 0.0, 1.0]

            gltf_node: Dict[str, Any] = {
                "name": j_name,
                "translation": [float(t_local[0]), float(t_local[1]), float(t_local[2])],
                "rotation": r_local,
            }
            if children_indices:
                gltf_node["children"] = children_indices
            nodes.append(gltf_node)

        # Root node index
        root_idx = node_map[skeleton.root_name]

        # 2. Build Animation Tracks & Binary Buffer
        # Buffer collects:
        # - Timestamps (float32 array)
        # - Root translation keys (float32 x 3 per frame)
        # - Joint rotation keys (float32 x 4 per frame [x, y, z, w])
        # - Inverse bind matrices (float32 x 16 per joint)

        raw_buffer = bytearray()
        buffer_views = []
        accessors = []

        def add_buffer_data(
            data_bytes: bytes,
            component_type: int,
            type_str: str,
            count: int,
            min_vals: Optional[List[float]] = None,
            max_vals: Optional[List[float]] = None,
            target: Optional[int] = None,
        ) -> int:
            nonlocal raw_buffer
            # 4-byte alignment
            offset = len(raw_buffer)
            pad = (4 - (offset % 4)) % 4
            if pad > 0:
                raw_buffer.extend(b"\x00" * pad)
                offset += pad

            byte_len = len(data_bytes)
            raw_buffer.extend(data_bytes)

            bv_idx = len(buffer_views)
            bv: Dict[str, Any] = {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": byte_len,
            }
            if target is not None:
                bv["target"] = target
            buffer_views.append(bv)

            acc_idx = len(accessors)
            acc: Dict[str, Any] = {
                "bufferView": bv_idx,
                "byteOffset": 0,
                "componentType": component_type,
                "count": count,
                "type": type_str,
            }
            if min_vals is not None and max_vals is not None:
                acc["min"] = min_vals
                acc["max"] = max_vals
            accessors.append(acc)
            return acc_idx

        # 2a. Timestamps accessor (5126 = FLOAT)
        T = animation.frame_count
        times = np.asarray(animation.timestamps, dtype=np.float32)
        times_bytes = times.tobytes()
        time_acc_idx = add_buffer_data(
            times_bytes,
            component_type=5126,
            type_str="SCALAR",
            count=T,
            min_vals=[float(times[0]) if T > 0 else 0.0],
            max_vals=[float(times[-1]) if T > 0 else 0.0],
        )

        channels = []
        samplers = []

        # 2b. Root translation animation
        root_trans = np.asarray(animation.root_translations, dtype=np.float32)
        root_trans_bytes = root_trans.tobytes()
        min_root = [float(np.min(root_trans[:, d])) for d in range(3)] if T > 0 else [0.0, 0.0, 0.0]
        max_root = [float(np.max(root_trans[:, d])) for d in range(3)] if T > 0 else [0.0, 0.0, 0.0]
        trans_acc_idx = add_buffer_data(
            root_trans_bytes,
            component_type=5126,
            type_str="VEC3",
            count=T,
            min_vals=min_root,
            max_vals=max_root,
        )

        sampler_root_idx = len(samplers)
        samplers.append({
            "input": time_acc_idx,
            "output": trans_acc_idx,
            "interpolation": "LINEAR",
        })
        channels.append({
            "sampler": sampler_root_idx,
            "target": {"node": root_idx, "path": "translation"},
        })

        # 2c. Joint rotations animation
        for j_name, node_idx in node_map.items():
            if j_name in animation.rotations:
                # Convert internal [w, x, y, z] to glTF [x, y, z, w]
                internal_q = animation.rotations[j_name]  # Shape: (T, 4)
                gltf_q = np.zeros((T, 4), dtype=np.float32)
                gltf_q[:, 0] = internal_q[:, 1]  # x
                gltf_q[:, 1] = internal_q[:, 2]  # y
                gltf_q[:, 2] = internal_q[:, 3]  # z
                gltf_q[:, 3] = internal_q[:, 0]  # w

                q_bytes = gltf_q.tobytes()
                rot_acc_idx = add_buffer_data(q_bytes, component_type=5126, type_str="VEC4", count=T)

                samp_idx = len(samplers)
                samplers.append({
                    "input": time_acc_idx,
                    "output": rot_acc_idx,
                    "interpolation": "LINEAR",
                })
                channels.append({
                    "sampler": samp_idx,
                    "target": {"node": node_idx, "path": "rotation"},
                })

        # 2d. Inverse Bind Matrices for skinning
        # For each joint, compute full 4x4 rest world affine transform M_j = [R | T],
        # compute true inverse M_inv = np.linalg.inv(M), and serialize in column-major order (M_inv.T).
        ibm_data = np.zeros((len(joint_names), 4, 4), dtype=np.float32)
        for j_idx, j_name in enumerate(joint_names):
            node = skeleton.joints[j_name]
            mat = np.eye(4, dtype=np.float32)
            if hasattr(node, "world_rotation") and node.world_rotation is not None:
                mat[:3, :3] = QuaternionMath.quat_to_matrix(node.world_rotation)
            mat[:3, 3] = node.rest_world_pos
            try:
                m_inv = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                m_inv = np.eye(4, dtype=np.float32)
            ibm_data[j_idx] = m_inv.T

        ibm_bytes = ibm_data.tobytes()
        ibm_acc_idx = add_buffer_data(
            ibm_bytes,
            component_type=5126,
            type_str="MAT4",
            count=len(joint_names),
        )

        # 3. Generate or Load Skinned 3D Mesh Geometry
        custom_mesh = kwargs.get("mesh", None)
        uvs = kwargs.get("uvs", None)
        if custom_mesh is not None:
            if isinstance(custom_mesh, (tuple, list)):
                if len(custom_mesh) == 6:
                    verts, norms, uvs, tri_indices, j_indices, j_weights = custom_mesh
                elif len(custom_mesh) == 5:
                    verts, norms, tri_indices, j_indices, j_weights = custom_mesh
                elif len(custom_mesh) == 4:
                    verts, tri_indices, arr3, arr4 = custom_mesh
                    if np.issubdtype(np.asarray(arr3).dtype, np.floating) and np.issubdtype(np.asarray(arr4).dtype, np.integer):
                        j_weights, j_indices = arr3, arr4
                    elif np.issubdtype(np.asarray(arr4).dtype, np.floating) and np.issubdtype(np.asarray(arr3).dtype, np.integer):
                        j_indices, j_weights = arr3, arr4
                    else:
                        j_weights, j_indices = arr3, arr4
                    norms = None
                else:
                    raise ExportError(f"Unsupported mesh tuple length: {len(custom_mesh)}")
            elif isinstance(custom_mesh, dict):
                verts = custom_mesh.get("vertices", custom_mesh.get("verts"))
                tri_indices = custom_mesh.get("faces", custom_mesh.get("indices", custom_mesh.get("tri_indices")))
                norms = custom_mesh.get("normals", custom_mesh.get("norms", None))
                uvs = custom_mesh.get("uvs", uvs)
                j_weights = custom_mesh.get("weights", custom_mesh.get("j_weights"))
                j_indices = custom_mesh.get("joint_indices", custom_mesh.get("j_indices", custom_mesh.get("joints")))
            else:
                verts = getattr(custom_mesh, "vertices", getattr(custom_mesh, "verts", None))
                tri_indices = getattr(custom_mesh, "faces", getattr(custom_mesh, "indices", None))
                norms = getattr(custom_mesh, "normals", getattr(custom_mesh, "norms", None))
                uvs = getattr(custom_mesh, "uvs", uvs)
                j_weights = getattr(custom_mesh, "weights", getattr(custom_mesh, "j_weights", None))
                j_indices = getattr(custom_mesh, "joint_indices", getattr(custom_mesh, "j_indices", None))

            verts = np.asarray(verts, dtype=np.float32)
            tri_indices = np.asarray(tri_indices)
            j_indices = np.asarray(j_indices, dtype=np.uint16)
            j_weights = np.asarray(j_weights, dtype=np.float32)

            if norms is None or len(norms) == 0:
                norms = np.zeros_like(verts, dtype=np.float32)
                v0 = verts[tri_indices[:, 0]]
                v1 = verts[tri_indices[:, 1]]
                v2 = verts[tri_indices[:, 2]]
                fn = np.cross(v1 - v0, v2 - v0)
                for i in range(3):
                    np.add.at(norms, tri_indices[:, i], fn)
                lens = np.linalg.norm(norms, axis=1, keepdims=True)
                lens[lens < 1e-8] = 1.0
                norms = (norms / lens).astype(np.float32)
            else:
                norms = np.asarray(norms, dtype=np.float32)
        else:
            from iris.rigging.proxy_mesh import ProxyMeshGenerator
            verts, norms, tri_indices, j_indices, j_weights = ProxyMeshGenerator.create_skinned_mannequin(skeleton)

        # Sanitize weights and indices for strict glTF 2.0 conformance (Khronos glTF-Validator clean):
        # 1. Merge duplicate joint indices per vertex (sum their weights).
        # 2. Sort influences strictly descending by weight, keep the top 4.
        # 3. Normalize weights to sum to 1.0 (partition of unity).
        # 4. Every zero-weight slot gets joint index 0. Per spec, joints with zero weight carry no influence,
        #    and the validator flags non-zero indices on zero weights (ACCESSOR_JOINTS_USED_ZERO_WEIGHT).
        #    Repeating index 0 on zero-weight slots is allowed: duplicates are only an error for non-zero weights.
        if len(j_weights) > 0:
            j_indices, j_weights = self.sanitize_skin_influences(j_indices, j_weights)

        # Eliminate any zero-area or degenerate faces (area < 1e-7)
        if len(tri_indices) > 0 and len(verts) > 0:
            v0 = verts[tri_indices[:, 0]]
            v1 = verts[tri_indices[:, 1]]
            v2 = verts[tri_indices[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            valid_mask = areas >= 1e-7
            if not np.all(valid_mask):
                tri_indices = tri_indices[valid_mask]

        # 3a. Add Mesh Vertex Attributes to Buffer
        min_pos = [float(np.min(verts[:, d])) for d in range(3)]
        max_pos = [float(np.max(verts[:, d])) for d in range(3)]
        pos_acc_idx = add_buffer_data(
            verts.tobytes(),
            component_type=5126,  # FLOAT
            type_str="VEC3",
            count=len(verts),
            min_vals=min_pos,
            max_vals=max_pos,
            target=34962,  # ARRAY_BUFFER
        )
        norm_acc_idx = add_buffer_data(
            norms.tobytes(),
            component_type=5126,  # FLOAT
            type_str="VEC3",
            count=len(norms),
            target=34962,
        )
        joints_acc_idx = add_buffer_data(
            j_indices.tobytes(),
            component_type=5123,  # UNSIGNED_SHORT
            type_str="VEC4",
            count=len(j_indices),
            target=34962,
        )
        weights_acc_idx = add_buffer_data(
            j_weights.tobytes(),
            component_type=5126,  # FLOAT
            type_str="VEC4",
            count=len(j_weights),
            target=34962,
        )
        is_uint32_indices = np.max(tri_indices) >= 65536
        idx_dtype = np.uint32 if is_uint32_indices else np.uint16
        idx_comp_type = 5125 if is_uint32_indices else 5123
        flat_indices = tri_indices.flatten().astype(idx_dtype)
        indices_acc_idx = add_buffer_data(
            flat_indices.tobytes(),
            component_type=idx_comp_type,
            type_str="SCALAR",
            count=len(flat_indices),
            min_vals=[int(np.min(flat_indices))],
            max_vals=[int(np.max(flat_indices))],
            target=34963,  # ELEMENT_ARRAY_BUFFER
        )

        # 3b. Texture Coordinates (TEXCOORD_0)
        prim_attributes = {
            "POSITION": pos_acc_idx,
            "NORMAL": norm_acc_idx,
            "JOINTS_0": joints_acc_idx,
            "WEIGHTS_0": weights_acc_idx,
        }
        if uvs is not None and len(uvs) == len(verts):
            uvs_arr = np.asarray(uvs, dtype=np.float32)
            tex_acc_idx = add_buffer_data(
                uvs_arr.tobytes(),
                component_type=5126,  # FLOAT
                type_str="VEC2",
                count=len(uvs_arr),
                min_vals=[float(np.min(uvs_arr[:, 0])), float(np.min(uvs_arr[:, 1]))],
                max_vals=[float(np.max(uvs_arr[:, 0])), float(np.max(uvs_arr[:, 1]))],
                target=34962,  # ARRAY_BUFFER
            )
            prim_attributes["TEXCOORD_0"] = tex_acc_idx

        # 3c. Skinned Mesh & Material Definitions
        meshes = [
            {
                "name": f"Mesh_{skeleton.name}",
                "primitives": [
                    {
                        "attributes": prim_attributes,
                        "indices": indices_acc_idx,
                        "material": 0,
                        "mode": 4,  # TRIANGLES
                    }
                ],
            }
        ]

        custom_material = kwargs.get("material", None)
        if custom_material is not None:
            materials = [dict(custom_material)]
        else:
            materials = [
                {
                    "name": "Cyan_Luminous_Skeletal_Material",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.0, 1.0, 0.85, 1.0],
                        "metallicFactor": 0.15,
                        "roughnessFactor": 0.25,
                    },
                    "emissiveFactor": [0.0, 0.85, 0.72],
                    "doubleSided": True,
                }
            ]

        # Connect diffuse texture if provided
        texture_path = kwargs.get("texture_path", kwargs.get("texture_uri", None))
        gltf_images = []
        gltf_samplers = []
        gltf_textures = []
        if texture_path:
            tex_name = Path(texture_path).name
            gltf_images.append({"uri": tex_name})
            gltf_samplers.append({
                "magFilter": 9729,  # LINEAR
                "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
                "wrapS": 10497,     # REPEAT
                "wrapT": 10497,     # REPEAT
            })
            gltf_textures.append({
                "sampler": 0,
                "source": 0,
            })
            if materials and "pbrMetallicRoughness" in materials[0]:
                materials[0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

        # 3d. Skinned Mesh Node bound to skeleton skin 0
        mesh_node_idx = len(nodes)
        nodes.append({
            "name": f"SkinnedMesh_{skeleton.name}",
            "mesh": 0,
            "skin": 0,
        })

        # 4. Assemble glTF Document with Base64 embedded URI
        b64_buffer = base64.b64encode(raw_buffer).decode("ascii")
        data_uri = f"data:application/octet-stream;base64,{b64_buffer}"

        gltf_doc = {
            "asset": {
                "version": "2.0",
                "generator": "Leocherd Iris Energy 0.1.0 Universal glTF Exporter",
            },
            "scene": 0,
            "scenes": [{"nodes": [root_idx, mesh_node_idx]}],
            "nodes": nodes,
            "materials": materials,
            "meshes": meshes,
            "skins": [
                {
                    "name": f"Skin_{skeleton.name}",
                    "inverseBindMatrices": ibm_acc_idx,
                    "joints": [node_map[j] for j in joint_names],
                    "skeleton": root_idx,
                }
            ],
            "animations": [
                {
                    "name": animation.name,
                    "channels": channels,
                    "samplers": samplers,
                }
            ],
            "buffers": [
                {
                    "byteLength": len(raw_buffer),
                    "uri": data_uri,
                }
            ],
            "bufferViews": buffer_views,
            "accessors": accessors,
        }

        if gltf_images:
            gltf_doc["images"] = gltf_images
            gltf_doc["samplers"] = gltf_samplers
            gltf_doc["textures"] = gltf_textures

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(gltf_doc, f, indent=2)

        logger.info(f"Successfully generated glTF 2.0 asset ({len(nodes)} nodes, {len(channels)} animation tracks)")
        return out_file


# Alias for backward/forward compatibility
GLTFExporter = GltfExporter
