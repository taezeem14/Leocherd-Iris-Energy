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

        # 1. Map joint hierarchy to glTF nodes
        joint_names = list(skeleton.joints.keys())
        node_map = {name: idx for idx, name in enumerate(joint_names)}

        nodes = []
        for j_name in joint_names:
            node = skeleton.joints[j_name]
            children_indices = [node_map[c] for c in node.children_names if c in node_map]

            # glTF quaternion convention: [x, y, z, w]
            q_w, q_x, q_y, q_z = node.local_rotation
            gltf_node: Dict[str, Any] = {
                "name": j_name,
                "translation": [round(float(v), 5) for v in node.local_pos],
                "rotation": [round(float(q_x), 5), round(float(q_y), 5), round(float(q_z), 5), round(float(q_w), 5)],
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
        times = animation.timestamps.astype(np.float32)
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
        root_trans = animation.root_translations.astype(np.float32)
        root_trans_bytes = root_trans.tobytes()
        min_pos = [float(np.min(root_trans[:, d])) for d in range(3)] if T > 0 else [0.0, 0.0, 0.0]
        max_pos = [float(np.max(root_trans[:, d])) for d in range(3)] if T > 0 else [0.0, 0.0, 0.0]
        trans_acc_idx = add_buffer_data(
            root_trans_bytes,
            component_type=5126,
            type_str="VEC3",
            count=T,
            min_vals=min_pos,
            max_vals=max_pos,
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
        # For each joint, compute 4x4 inverse bind matrix: translation(-rest_pos)
        ibm_data = np.zeros((len(joint_names), 4, 4), dtype=np.float32)
        for j_idx, j_name in enumerate(joint_names):
            node = skeleton.joints[j_name]
            # Column-major identity with negated translation in 4th column
            mat = np.eye(4, dtype=np.float32)
            mat[3, 0] = -node.rest_world_pos[0]
            mat[3, 1] = -node.rest_world_pos[1]
            mat[3, 2] = -node.rest_world_pos[2]
            ibm_data[j_idx] = mat

        ibm_bytes = ibm_data.tobytes()
        ibm_acc_idx = add_buffer_data(
            ibm_bytes,
            component_type=5126,
            type_str="MAT4",
            count=len(joint_names),
        )

        # 3. Generate Skinned 3D Proxy Mannequin Geometry
        from iris.rigging.proxy_mesh import ProxyMeshGenerator

        verts, norms, tri_indices, j_indices, j_weights = ProxyMeshGenerator.create_skinned_mannequin(skeleton)

        # 3a. Add Mesh Vertex Attributes to Buffer
        min_pos = [round(float(v), 5) for v in np.min(verts, axis=0)]
        max_pos = [round(float(v), 5) for v in np.max(verts, axis=0)]
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
        flat_indices = tri_indices.flatten().astype(np.uint16)
        indices_acc_idx = add_buffer_data(
            flat_indices.tobytes(),
            component_type=5123,  # UNSIGNED_SHORT
            type_str="SCALAR",
            count=len(flat_indices),
            min_vals=[int(np.min(flat_indices))],
            max_vals=[int(np.max(flat_indices))],
            target=34963,  # ELEMENT_ARRAY_BUFFER
        )

        # 3b. Skinned Mesh & Material Definitions
        meshes = [
            {
                "name": f"Mesh_{skeleton.name}",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": pos_acc_idx,
                            "NORMAL": norm_acc_idx,
                            "JOINTS_0": joints_acc_idx,
                            "WEIGHTS_0": weights_acc_idx,
                        },
                        "indices": indices_acc_idx,
                        "material": 0,
                        "mode": 4,  # TRIANGLES
                    }
                ],
            }
        ]

        materials = [
            {
                "name": "Iris_Mannequin_Material",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.22, 0.65, 0.95, 1.0],  # Iris Blue/Cyan
                    "metallicFactor": 0.15,
                    "roughnessFactor": 0.45,
                },
                "doubleSided": True,
            }
        ]

        # 3c. Skinned Mesh Node bound to skeleton skin 0
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

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(gltf_doc, f, indent=2)

        logger.info(f"Successfully generated glTF 2.0 asset ({len(nodes)} nodes, {len(channels)} animation tracks)")
        return out_file
