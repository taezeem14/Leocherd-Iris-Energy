#!/usr/bin/env python3
"""
Leocherd Iris Energy - 3D Rigging & Asset Reconstruction Benchmarking Tool.

Validates glTF 2.0 and BVH motion capture assets against industry standards,
audits specification compliance, and computes statistical metrics on bone
rotations (quaternion normalization |q| ≈ 1.0) and root motion bounding boxes.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# -----------------------------------------------------------------------------
# Data Models & Results
# -----------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: str  # "ERROR" or "WARNING"
    component: str  # "glTF", "BVH", "Quaternion", "RootMotion"
    message: str


@dataclass
class QuaternionStats:
    total_samples: int
    mean_norm: float
    min_norm: float
    max_norm: float
    std_norm: float
    max_deviation: float
    normalized_ratio_1e3: float
    normalized_ratio_1e4: float
    is_valid: bool
    per_joint: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class RootMotionStats:
    source: str
    frame_count: int
    fps: float
    min_bounds: List[float]
    max_bounds: List[float]
    extents: List[float]
    center_pos: List[float]
    std_pos: List[float]
    total_trajectory_distance: float
    max_frame_displacement: float
    is_physically_plausible: bool


@dataclass
class GltfSpecValidation:
    valid: bool
    asset_version: str
    generator: str
    scenes_count: int
    nodes_count: int
    skins_count: int
    animations_count: int
    channels_count: int
    samplers_count: int
    buffers_count: int
    buffer_views_count: int
    accessors_count: int
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class BvhSpecValidation:
    valid: bool
    root_name: str
    joint_count: int
    total_channels: int
    frame_count: int
    frame_time: float
    fps: float
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    overall_status: str  # "PASS" or "FAIL"
    gltf_file: str
    bvh_file: str
    gltf_spec: GltfSpecValidation
    bvh_spec: BvhSpecValidation
    quaternion_stats: QuaternionStats
    gltf_root_motion: Optional[RootMotionStats]
    bvh_root_motion: Optional[RootMotionStats]
    cross_format_discrepancy: Optional[Dict[str, float]]
    issues: List[ValidationIssue] = field(default_factory=list)


# -----------------------------------------------------------------------------
# glTF 2.0 Specification Validator & Asset Parser
# -----------------------------------------------------------------------------

class GltfValidator:
    """Validates glTF 2.0 specification compliance and extracts animation data."""

    COMPONENT_TYPE_SIZES = {
        5120: 1,  # BYTE
        5121: 1,  # UNSIGNED_BYTE
        5122: 2,  # SHORT
        5123: 2,  # UNSIGNED_SHORT
        5125: 4,  # UNSIGNED_INT
        5126: 4,  # FLOAT
    }

    TYPE_NUM_COMPONENTS = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.doc: Dict[str, Any] = {}
        self.buffers_data: List[bytes] = []
        self.issues: List[ValidationIssue] = []

    def load_and_validate(self) -> GltfSpecValidation:
        if not self.path.exists():
            self.issues.append(ValidationIssue("ERROR", "glTF", f"File not found: {self.path}"))
            return GltfSpecValidation(
                valid=False,
                asset_version="N/A",
                generator="N/A",
                scenes_count=0,
                nodes_count=0,
                skins_count=0,
                animations_count=0,
                channels_count=0,
                samplers_count=0,
                buffers_count=0,
                buffer_views_count=0,
                accessors_count=0,
                issues=self.issues,
            )

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.doc = json.load(f)
        except Exception as e:
            self.issues.append(ValidationIssue("ERROR", "glTF", f"Failed to parse glTF JSON: {e}"))
            return GltfSpecValidation(
                valid=False,
                asset_version="N/A",
                generator="N/A",
                scenes_count=0,
                nodes_count=0,
                skins_count=0,
                animations_count=0,
                channels_count=0,
                samplers_count=0,
                buffers_count=0,
                buffer_views_count=0,
                accessors_count=0,
                issues=self.issues,
            )

        # 1. Validate 'asset'
        asset = self.doc.get("asset")
        if not isinstance(asset, dict):
            self.issues.append(ValidationIssue("ERROR", "glTF", "Missing or invalid 'asset' dictionary"))
            asset_version = "UNKNOWN"
            generator = "UNKNOWN"
        else:
            asset_version = asset.get("version", "")
            generator = asset.get("generator", "")
            if asset_version != "2.0":
                self.issues.append(
                    ValidationIssue("ERROR", "glTF", f"Expected asset.version '2.0', found '{asset_version}'")
                )

        # 2. Validate 'scene' and 'scenes'
        scenes = self.doc.get("scenes", [])
        if not isinstance(scenes, list) or len(scenes) == 0:
            self.issues.append(ValidationIssue("ERROR", "glTF", "'scenes' must be a non-empty list"))
        active_scene = self.doc.get("scene")
        if active_scene is not None:
            if not isinstance(active_scene, int) or active_scene < 0 or active_scene >= len(scenes):
                self.issues.append(ValidationIssue("ERROR", "glTF", f"Default 'scene' index {active_scene} is out of bounds"))

        # 3. Validate 'nodes'
        nodes = self.doc.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) == 0:
            self.issues.append(ValidationIssue("ERROR", "glTF", "'nodes' must be a non-empty list"))
        else:
            for idx, node in enumerate(nodes):
                if not isinstance(node, dict):
                    self.issues.append(ValidationIssue("ERROR", "glTF", f"Node at index {idx} is not an object"))
                    continue
                if "translation" in node:
                    t = node["translation"]
                    if not isinstance(t, list) or len(t) != 3 or not all(isinstance(v, (int, float)) for v in t):
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"Node {idx} has invalid 'translation'"))
                if "rotation" in node:
                    r = node["rotation"]
                    if not isinstance(r, list) or len(r) != 4 or not all(isinstance(v, (int, float)) for v in r):
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"Node {idx} has invalid 'rotation'"))
                if "children" in node:
                    for c in node["children"]:
                        if not isinstance(c, int) or c < 0 or c >= len(nodes):
                            self.issues.append(ValidationIssue("ERROR", "glTF", f"Node {idx} references invalid child index {c}"))
                        elif c == idx:
                            self.issues.append(ValidationIssue("ERROR", "glTF", f"Node {idx} has self-referencing child"))

        # 4. Validate 'skins'
        skins = self.doc.get("skins", [])
        if not isinstance(skins, list):
            self.issues.append(ValidationIssue("ERROR", "glTF", "'skins' must be a list"))
        else:
            for s_idx, skin in enumerate(skins):
                if "joints" not in skin or not isinstance(skin["joints"], list) or len(skin["joints"]) == 0:
                    self.issues.append(ValidationIssue("ERROR", "glTF", f"Skin {s_idx} missing non-empty 'joints' list"))
                else:
                    for j in skin["joints"]:
                        if j < 0 or j >= len(nodes):
                            self.issues.append(ValidationIssue("ERROR", "glTF", f"Skin {s_idx} references invalid joint node {j}"))
                if "skeleton" in skin and (skin["skeleton"] < 0 or skin["skeleton"] >= len(nodes)):
                    self.issues.append(ValidationIssue("ERROR", "glTF", f"Skin {s_idx} invalid 'skeleton' node index"))

        # 5. Validate 'buffers' & decode data
        buffers = self.doc.get("buffers", [])
        if not isinstance(buffers, list) or len(buffers) == 0:
            self.issues.append(ValidationIssue("ERROR", "glTF", "'buffers' must be a non-empty list"))
        else:
            for b_idx, buf in enumerate(buffers):
                byte_length = buf.get("byteLength", 0)
                if byte_length <= 0:
                    self.issues.append(ValidationIssue("ERROR", "glTF", f"Buffer {b_idx} has invalid byteLength {byte_length}"))
                uri = buf.get("uri", "")
                if uri.startswith("data:"):
                    try:
                        raw_b64 = uri.split(",", 1)[1]
                        decoded = base64.b64decode(raw_b64)
                        self.buffers_data.append(decoded)
                        if len(decoded) < byte_length:
                            self.issues.append(ValidationIssue("ERROR", "glTF", f"Buffer {b_idx} data is smaller than byteLength"))
                    except Exception as e:
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"Buffer {b_idx} base64 decode failed: {e}"))
                elif uri:
                    bin_path = self.path.parent / uri
                    if bin_path.exists():
                        self.buffers_data.append(bin_path.read_bytes())
                    else:
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"External buffer file not found: {bin_path}"))
                else:
                    self.issues.append(ValidationIssue("ERROR", "glTF", f"Buffer {b_idx} has no valid URI"))

        # 6. Validate 'bufferViews'
        buffer_views = self.doc.get("bufferViews", [])
        for bv_idx, bv in enumerate(buffer_views):
            b_idx = bv.get("buffer")
            if b_idx is None or b_idx < 0 or b_idx >= len(buffers):
                self.issues.append(ValidationIssue("ERROR", "glTF", f"bufferView {bv_idx} references invalid buffer {b_idx}"))
                continue
            offset = bv.get("byteOffset", 0)
            length = bv.get("byteLength", 0)
            if b_idx < len(self.buffers_data):
                if offset + length > len(self.buffers_data[b_idx]):
                    self.issues.append(ValidationIssue("ERROR", "glTF", f"bufferView {bv_idx} out of buffer bounds"))

        # 7. Validate 'accessors'
        accessors = self.doc.get("accessors", [])
        for a_idx, acc in enumerate(accessors):
            bv_idx = acc.get("bufferView")
            if bv_idx is not None and (bv_idx < 0 or bv_idx >= len(buffer_views)):
                self.issues.append(ValidationIssue("ERROR", "glTF", f"accessor {a_idx} references invalid bufferView {bv_idx}"))
            comp_type = acc.get("componentType")
            if comp_type not in self.COMPONENT_TYPE_SIZES:
                self.issues.append(ValidationIssue("ERROR", "glTF", f"accessor {a_idx} has unknown componentType {comp_type}"))
            acc_type = acc.get("type")
            if acc_type not in self.TYPE_NUM_COMPONENTS:
                self.issues.append(ValidationIssue("ERROR", "glTF", f"accessor {a_idx} has unknown type {acc_type}"))

        # 8. Validate 'animations'
        animations = self.doc.get("animations", [])
        total_channels = 0
        total_samplers = 0
        if not isinstance(animations, list) or len(animations) == 0:
            self.issues.append(ValidationIssue("ERROR", "glTF", "'animations' must be a non-empty list"))
        else:
            for anim_idx, anim in enumerate(animations):
                samplers = anim.get("samplers", [])
                channels = anim.get("channels", [])
                total_samplers += len(samplers)
                total_channels += len(channels)

                if len(samplers) == 0:
                    self.issues.append(ValidationIssue("ERROR", "glTF", f"Animation {anim_idx} has no samplers"))
                if len(channels) == 0:
                    self.issues.append(ValidationIssue("ERROR", "glTF", f"Animation {anim_idx} has no channels"))

                for s_idx, samp in enumerate(samplers):
                    inp = samp.get("input")
                    out = samp.get("output")
                    if inp is None or inp < 0 or inp >= len(accessors):
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"Sampler {s_idx} has invalid input accessor {inp}"))
                    if out is None or out < 0 or out >= len(accessors):
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"Sampler {s_idx} has invalid output accessor {out}"))

                for c_idx, ch in enumerate(channels):
                    s_ref = ch.get("sampler")
                    if s_ref is None or s_ref < 0 or s_ref >= len(samplers):
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"Channel {c_idx} references invalid sampler {s_ref}"))
                    target = ch.get("target", {})
                    t_node = target.get("node")
                    t_path = target.get("path")
                    if t_node is None or t_node < 0 or t_node >= len(nodes):
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"Channel {c_idx} references invalid target node {t_node}"))
                    if t_path not in ("translation", "rotation", "scale", "weights"):
                        self.issues.append(ValidationIssue("ERROR", "glTF", f"Channel {c_idx} has invalid target path '{t_path}'"))

        has_errors = any(i.severity == "ERROR" for i in self.issues)
        return GltfSpecValidation(
            valid=not has_errors,
            asset_version=asset_version,
            generator=generator,
            scenes_count=len(scenes),
            nodes_count=len(nodes),
            skins_count=len(skins),
            animations_count=len(animations),
            channels_count=total_channels,
            samplers_count=total_samplers,
            buffers_count=len(buffers),
            buffer_views_count=len(buffer_views),
            accessors_count=len(accessors),
            issues=self.issues,
        )

    def extract_accessor_array(self, accessor_idx: int) -> Optional[np.ndarray]:
        """Decodes an accessor from raw binary buffer data into a typed NumPy array."""
        accessors = self.doc.get("accessors", [])
        if accessor_idx < 0 or accessor_idx >= len(accessors):
            return None
        acc = accessors[accessor_idx]
        bv_idx = acc.get("bufferView")
        if bv_idx is None:
            return None

        buffer_views = self.doc.get("bufferViews", [])
        if bv_idx < 0 or bv_idx >= len(buffer_views):
            return None
        bv = buffer_views[bv_idx]

        b_idx = bv.get("buffer", 0)
        if b_idx < 0 or b_idx >= len(self.buffers_data):
            return None

        raw_buffer = self.buffers_data[b_idx]
        offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc.get("count", 0)
        comp_type = acc.get("componentType")
        type_str = acc.get("type")

        num_components = self.TYPE_NUM_COMPONENTS.get(type_str, 1)

        # Assuming FLOAT (5126) for animation curves
        if comp_type == 5126:
            dtype = np.float32
        elif comp_type in (5120, 5121):
            dtype = np.uint8 if comp_type == 5121 else np.int8
        elif comp_type in (5122, 5123):
            dtype = np.uint16 if comp_type == 5123 else np.int16
        elif comp_type == 5125:
            dtype = np.uint32
        else:
            return None

        total_elements = count * num_components
        try:
            arr = np.frombuffer(raw_buffer, dtype=dtype, count=total_elements, offset=offset)
            if num_components > 1:
                arr = arr.reshape(count, num_components)
            return arr
        except Exception:
            return None

    def get_animation_quaternions(self) -> Dict[str, np.ndarray]:
        """Extracts quaternion rotations for each animated node from the glTF asset."""
        nodes = self.doc.get("nodes", [])
        animations = self.doc.get("animations", [])
        rotations: Dict[str, np.ndarray] = {}

        for anim in animations:
            samplers = anim.get("samplers", [])
            for ch in anim.get("channels", []):
                target = ch.get("target", {})
                if target.get("path") == "rotation":
                    node_idx = target.get("node")
                    sampler_idx = ch.get("sampler")
                    if node_idx is not None and sampler_idx is not None:
                        samp = samplers[sampler_idx]
                        arr = self.extract_accessor_array(samp["output"])
                        if arr is not None and arr.shape[1] == 4:
                            node_name = nodes[node_idx].get("name", f"Node_{node_idx}")
                            rotations[node_name] = arr

        # Also fallback/supplement with node rest rotations if present
        for idx, node in enumerate(nodes):
            name = node.get("name", f"Node_{idx}")
            if name not in rotations and "rotation" in node:
                r = np.array(node["rotation"], dtype=np.float32).reshape(1, 4)
                rotations[name] = r

        return rotations

    def get_root_translations(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Extracts root translation track from glTF animations."""
        nodes = self.doc.get("nodes", [])
        animations = self.doc.get("animations", [])

        # Check animation channels for translation
        for anim in animations:
            samplers = anim.get("samplers", [])
            for ch in anim.get("channels", []):
                target = ch.get("target", {})
                if target.get("path") == "translation":
                    node_idx = target.get("node")
                    sampler_idx = ch.get("sampler")
                    if node_idx is not None and sampler_idx is not None:
                        samp = samplers[sampler_idx]
                        arr = self.extract_accessor_array(samp["output"])
                        if arr is not None and arr.shape[1] == 3:
                            node_name = nodes[node_idx].get("name", f"Node_{node_idx}")
                            return node_name, arr

        # Fallback to skin skeleton root
        skins = self.doc.get("skins", [])
        if skins and "skeleton" in skins[0]:
            root_idx = skins[0]["skeleton"]
            if root_idx < len(nodes):
                name = nodes[root_idx].get("name", f"Node_{root_idx}")
                trans = nodes[root_idx].get("translation", [0.0, 0.0, 0.0])
                return name, np.array(trans, dtype=np.float32).reshape(1, 3)

        return None, None


# -----------------------------------------------------------------------------
# BVH Specification Validator & Asset Parser
# -----------------------------------------------------------------------------

class BvhValidator:
    """Validates Biovision Hierarchy (BVH) files and extracts motion data."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.issues: List[ValidationIssue] = []
        self.root_name: str = ""
        self.joints: List[str] = []
        self.total_channels: int = 0
        self.frame_count: int = 0
        self.frame_time: float = 0.0
        self.motion_data: Optional[np.ndarray] = None

    def load_and_validate(self) -> BvhSpecValidation:
        if not self.path.exists():
            self.issues.append(ValidationIssue("ERROR", "BVH", f"File not found: {self.path}"))
            return BvhSpecValidation(
                valid=False,
                root_name="N/A",
                joint_count=0,
                total_channels=0,
                frame_count=0,
                frame_time=0.0,
                fps=0.0,
                issues=self.issues,
            )

        try:
            content = self.path.read_text(encoding="utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as e:
            self.issues.append(ValidationIssue("ERROR", "BVH", f"Could not read BVH file: {e}"))
            return BvhSpecValidation(
                valid=False,
                root_name="N/A",
                joint_count=0,
                total_channels=0,
                frame_count=0,
                frame_time=0.0,
                fps=0.0,
                issues=self.issues,
            )

        if not lines:
            self.issues.append(ValidationIssue("ERROR", "BVH", "File is empty"))
            return BvhSpecValidation(
                valid=False,
                root_name="N/A",
                joint_count=0,
                total_channels=0,
                frame_count=0,
                frame_time=0.0,
                fps=0.0,
                issues=self.issues,
            )

        # Check HIERARCHY header
        if lines[0] != "HIERARCHY":
            self.issues.append(ValidationIssue("ERROR", "BVH", f"First line must be 'HIERARCHY', found '{lines[0]}'"))

        # Find MOTION section index
        motion_idx = -1
        for idx, line in enumerate(lines):
            if line.startswith("MOTION"):
                motion_idx = idx
                break

        if motion_idx == -1:
            self.issues.append(ValidationIssue("ERROR", "BVH", "Missing 'MOTION' block in BVH file"))
            return BvhSpecValidation(
                valid=False,
                root_name="N/A",
                joint_count=0,
                total_channels=0,
                frame_count=0,
                frame_time=0.0,
                fps=0.0,
                issues=self.issues,
            )

        # 1. Parse HIERARCHY
        hierarchy_lines = lines[1:motion_idx]
        brace_balance = 0
        total_channels = 0
        joints = []
        root_name = ""

        for line in hierarchy_lines:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "ROOT":
                root_name = tokens[1] if len(tokens) > 1 else "Root"
                joints.append(root_name)
            elif tokens[0] == "JOINT":
                j_name = tokens[1] if len(tokens) > 1 else f"Joint_{len(joints)}"
                joints.append(j_name)
            elif tokens[0] == "CHANNELS":
                try:
                    num_ch = int(tokens[1])
                    total_channels += num_ch
                except (ValueError, IndexError):
                    self.issues.append(ValidationIssue("ERROR", "BVH", f"Malformed CHANNELS line: '{line}'"))
            elif tokens[0] == "{":
                brace_balance += 1
            elif tokens[0] == "}":
                brace_balance -= 1

        if brace_balance != 0:
            self.issues.append(
                ValidationIssue("ERROR", "BVH", f"Unbalanced braces in HIERARCHY (final delta: {brace_balance})")
            )

        if not root_name:
            self.issues.append(ValidationIssue("ERROR", "BVH", "No ROOT joint found in HIERARCHY"))

        self.root_name = root_name
        self.joints = joints
        self.total_channels = total_channels

        # 2. Parse MOTION block
        motion_lines = lines[motion_idx + 1 :]
        if len(motion_lines) < 2:
            self.issues.append(ValidationIssue("ERROR", "BVH", "MOTION block missing Frames and Frame Time headers"))
            frame_count = 0
            frame_time = 0.0
        else:
            frames_line = motion_lines[0]
            time_line = motion_lines[1]

            frame_count = 0
            frame_time = 0.0

            if frames_line.startswith("Frames:"):
                try:
                    frame_count = int(frames_line.split(":", 1)[1].strip())
                    if frame_count <= 0:
                        self.issues.append(ValidationIssue("ERROR", "BVH", f"Non-positive frame count: {frame_count}"))
                except ValueError:
                    self.issues.append(ValidationIssue("ERROR", "BVH", f"Invalid Frames value: '{frames_line}'"))
            else:
                self.issues.append(ValidationIssue("ERROR", "BVH", f"Expected 'Frames:' line, got '{frames_line}'"))

            if time_line.startswith("Frame Time:"):
                try:
                    frame_time = float(time_line.split(":", 1)[1].strip())
                    if frame_time <= 0:
                        self.issues.append(ValidationIssue("ERROR", "BVH", f"Non-positive frame time: {frame_time}"))
                except ValueError:
                    self.issues.append(ValidationIssue("ERROR", "BVH", f"Invalid Frame Time value: '{time_line}'"))
            else:
                self.issues.append(ValidationIssue("ERROR", "BVH", f"Expected 'Frame Time:' line, got '{time_line}'"))

            data_lines = motion_lines[2:]
            if len(data_lines) != frame_count:
                self.issues.append(
                    ValidationIssue(
                        "ERROR",
                        "BVH",
                        f"Frame count mismatch: header specifies {frame_count}, but found {len(data_lines)} data rows",
                    )
                )

            # Parse motion numerical values
            parsed_rows = []
            for row_idx, r_line in enumerate(data_lines):
                parts = r_line.split()
                if len(parts) != total_channels:
                    self.issues.append(
                        ValidationIssue(
                            "ERROR",
                            "BVH",
                            f"Row {row_idx} channels count mismatch: expected {total_channels}, got {len(parts)}",
                        )
                    )
                    break
                try:
                    vals = [float(p) for p in parts]
                    parsed_rows.append(vals)
                except ValueError:
                    self.issues.append(ValidationIssue("ERROR", "BVH", f"Non-numeric value in motion row {row_idx}"))
                    break

            if len(parsed_rows) == len(data_lines) and len(parsed_rows) > 0:
                self.motion_data = np.array(parsed_rows, dtype=np.float32)
                if np.isnan(self.motion_data).any():
                    self.issues.append(ValidationIssue("ERROR", "BVH", "Motion data contains NaN values"))
                if np.isinf(self.motion_data).any():
                    self.issues.append(ValidationIssue("ERROR", "BVH", "Motion data contains Inf values"))

        self.frame_count = frame_count
        self.frame_time = frame_time
        fps = 1.0 / frame_time if frame_time > 0 else 0.0

        has_errors = any(i.severity == "ERROR" for i in self.issues)
        return BvhSpecValidation(
            valid=not has_errors,
            root_name=self.root_name,
            joint_count=len(self.joints),
            total_channels=self.total_channels,
            frame_count=self.frame_count,
            frame_time=self.frame_time,
            fps=fps,
            issues=self.issues,
        )

    def get_root_translations(self) -> Optional[np.ndarray]:
        """Extracts root [X, Y, Z] translations from the first 3 channels of motion data."""
        if self.motion_data is not None and self.motion_data.shape[1] >= 3:
            return self.motion_data[:, :3]
        return None


# -----------------------------------------------------------------------------
# Statistical Metrics Computation
# -----------------------------------------------------------------------------

def compute_quaternion_metrics(
    rotations: Dict[str, np.ndarray],
    tolerance: float = 1e-3,
) -> QuaternionStats:
    """
    Computes rigorous statistical metrics on bone rotations to verify that
    all quaternions are strictly normalized (|q| ≈ 1.0).
    """
    all_norms: List[float] = []
    per_joint: Dict[str, Dict[str, float]] = {}

    for joint_name, quat_arr in rotations.items():
        if quat_arr.ndim != 2 or quat_arr.shape[1] != 4:
            continue

        norms = np.linalg.norm(quat_arr, axis=1)
        all_norms.extend(norms.tolist())

        devs = np.abs(norms - 1.0)
        per_joint[joint_name] = {
            "samples": float(len(norms)),
            "mean_norm": float(np.mean(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "max_deviation": float(np.max(devs)),
        }

    if not all_norms:
        return QuaternionStats(
            total_samples=0,
            mean_norm=0.0,
            min_norm=0.0,
            max_norm=0.0,
            std_norm=0.0,
            max_deviation=0.0,
            normalized_ratio_1e3=0.0,
            normalized_ratio_1e4=0.0,
            is_valid=False,
            per_joint={},
        )

    arr_norms = np.array(all_norms, dtype=np.float64)
    deviations = np.abs(arr_norms - 1.0)

    ratio_1e3 = float(np.mean(deviations <= 1e-3))
    ratio_1e4 = float(np.mean(deviations <= 1e-4))
    max_dev = float(np.max(deviations))

    is_valid = bool(ratio_1e3 >= 0.999 and max_dev <= 0.05)

    return QuaternionStats(
        total_samples=len(all_norms),
        mean_norm=float(np.mean(arr_norms)),
        min_norm=float(np.min(arr_norms)),
        max_norm=float(np.max(arr_norms)),
        std_norm=float(np.std(arr_norms)),
        max_deviation=max_dev,
        normalized_ratio_1e3=ratio_1e3,
        normalized_ratio_1e4=ratio_1e4,
        is_valid=is_valid,
        per_joint=per_joint,
    )


def compute_root_motion_metrics(
    translations: np.ndarray,
    fps: float,
    source_name: str,
    max_jump_threshold: float = 5.0,
) -> RootMotionStats:
    """
    Computes spatial bounds and kinetic metrics for root translation paths.
    """
    if translations.ndim != 2 or translations.shape[1] != 3 or len(translations) == 0:
        return RootMotionStats(
            source=source_name,
            frame_count=0,
            fps=fps,
            min_bounds=[0.0, 0.0, 0.0],
            max_bounds=[0.0, 0.0, 0.0],
            extents=[0.0, 0.0, 0.0],
            center_pos=[0.0, 0.0, 0.0],
            std_pos=[0.0, 0.0, 0.0],
            total_trajectory_distance=0.0,
            max_frame_displacement=0.0,
            is_physically_plausible=False,
        )

    min_b = np.min(translations, axis=0)
    max_b = np.max(translations, axis=0)
    extents = max_b - min_b
    mean_p = np.mean(translations, axis=0)
    std_p = np.std(translations, axis=0)

    if len(translations) > 1:
        step_diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        total_dist = float(np.sum(step_diffs))
        max_step = float(np.max(step_diffs))
    else:
        total_dist = 0.0
        max_step = 0.0

    is_plausible = bool(
        not np.isnan(translations).any()
        and not np.isinf(translations).any()
        and max_step <= max_jump_threshold
        and np.all(extents < 100.0)
    )

    return RootMotionStats(
        source=source_name,
        frame_count=len(translations),
        fps=fps,
        min_bounds=[round(float(v), 5) for v in min_b],
        max_bounds=[round(float(v), 5) for v in max_b],
        extents=[round(float(v), 5) for v in extents],
        center_pos=[round(float(v), 5) for v in mean_p],
        std_pos=[round(float(v), 5) for v in std_p],
        total_trajectory_distance=round(total_dist, 5),
        max_frame_displacement=round(max_step, 5),
        is_physically_plausible=is_plausible,
    )


# -----------------------------------------------------------------------------
# End-to-End Benchmarking Orchestration
# -----------------------------------------------------------------------------

def benchmark_reconstruction(
    gltf_path: Path | str,
    bvh_path: Path | str,
    tolerance: float = 1e-3,
    max_jump_threshold: float = 5.0,
) -> BenchmarkReport:
    """
    Executes comprehensive verification across both glTF and BVH assets,
    evaluating standard specs, quaternion normalization, and root motion kinetics.
    """
    gltf_val = GltfValidator(gltf_path)
    gltf_spec = gltf_val.load_and_validate()

    bvh_val = BvhValidator(bvh_path)
    bvh_spec = bvh_val.load_and_validate()

    issues: List[ValidationIssue] = list(gltf_spec.issues) + list(bvh_spec.issues)

    # 1. Compute Quaternion Norm Statistics (from glTF)
    rotations = gltf_val.get_animation_quaternions()
    quat_stats = compute_quaternion_metrics(rotations, tolerance=tolerance)
    if not quat_stats.is_valid:
        issues.append(
            ValidationIssue(
                "ERROR",
                "Quaternion",
                f"Quaternions are not normalized (|q| max dev: {quat_stats.max_deviation:.6f}, normalized ratio: {quat_stats.normalized_ratio_1e3:.2%})",
            )
        )

    # 2. Compute Root Motion Bounds (glTF)
    _, gltf_trans = gltf_val.get_root_translations()
    gltf_root_stats = (
        compute_root_motion_metrics(
            gltf_trans,
            fps=bvh_spec.fps or 30.0,
            source_name="glTF 2.0",
            max_jump_threshold=max_jump_threshold,
        )
        if gltf_trans is not None
        else None
    )
    if gltf_root_stats and not gltf_root_stats.is_physically_plausible:
        issues.append(
            ValidationIssue("ERROR", "RootMotion", "glTF root motion bounds violated physical plausibility limits")
        )

    # 3. Compute Root Motion Bounds (BVH)
    bvh_trans = bvh_val.get_root_translations()
    bvh_root_stats = (
        compute_root_motion_metrics(
            bvh_trans,
            fps=bvh_spec.fps,
            source_name="BVH",
            max_jump_threshold=max_jump_threshold,
        )
        if bvh_trans is not None
        else None
    )
    if bvh_root_stats and not bvh_root_stats.is_physically_plausible:
        issues.append(
            ValidationIssue("ERROR", "RootMotion", "BVH root motion bounds violated physical plausibility limits")
        )

    # 4. Cross-format consistency check
    discrepancy = None
    if gltf_trans is not None and bvh_trans is not None:
        if len(gltf_trans) == len(bvh_trans):
            abs_diff = np.abs(gltf_trans - bvh_trans)
            max_diff = float(np.max(abs_diff))
            mean_diff = float(np.mean(abs_diff))
            discrepancy = {
                "max_absolute_error_m": round(max_diff, 6),
                "mean_absolute_error_m": round(mean_diff, 6),
            }
            if max_diff > 1e-3:
                issues.append(
                    ValidationIssue(
                        "WARNING",
                        "CrossFormat",
                        f"glTF and BVH root motions differ by max {max_diff:.5f} m",
                    )
                )
        else:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "CrossFormat",
                    f"Frame count mismatch between glTF ({len(gltf_trans)}) and BVH ({len(bvh_trans)})",
                )
            )

    has_errors = any(i.severity == "ERROR" for i in issues)
    overall_status = "FAIL" if has_errors else "PASS"

    return BenchmarkReport(
        overall_status=overall_status,
        gltf_file=str(gltf_path),
        bvh_file=str(bvh_path),
        gltf_spec=gltf_spec,
        bvh_spec=bvh_spec,
        quaternion_stats=quat_stats,
        gltf_root_motion=gltf_root_stats,
        bvh_root_motion=bvh_root_stats,
        cross_format_discrepancy=discrepancy,
        issues=issues,
    )


# -----------------------------------------------------------------------------
# Reporting & CLI
# -----------------------------------------------------------------------------

def print_benchmark_report(report: BenchmarkReport) -> None:
    """Formats and prints an ASCII summary table for the terminal."""
    banner = "=" * 80
    print(f"\n{banner}")
    print(f"  LEOCHERD IRIS ENERGY - 3D RIGGING & RECONSTRUCTION BENCHMARK")
    print(f"{banner}")
    status_symbol = "[+] PASS" if report.overall_status == "PASS" else "[-] FAIL"
    print(f"  Overall Validation Status: {status_symbol}")
    print(f"  glTF Asset: {report.gltf_file}")
    print(f"  BVH Asset:  {report.bvh_file}")
    print(f"{banner}\n")

    # 1. glTF Specification
    print(" [1] glTF 2.0 Specification Audit")
    print(" -------------------------------------------------------------------------")
    g_spec = report.gltf_spec
    print(f"  * Specification Compliant:    {'YES' if g_spec.valid else 'NO'}")
    print(f"  * glTF Asset Version:         {g_spec.asset_version}")
    print(f"  * Asset Generator:            {g_spec.generator}")
    print(f"  * Scene Hierarchy:            {g_spec.scenes_count} scene(s), {g_spec.nodes_count} node(s)")
    print(f"  * Skeletal Skins:             {g_spec.skins_count} skin(s)")
    print(f"  * Animation Tracks:           {g_spec.animations_count} anim(s), {g_spec.channels_count} channel(s), {g_spec.samplers_count} sampler(s)")
    print(f"  * Binary Architecture:        {g_spec.buffers_count} buffer(s), {g_spec.buffer_views_count} bufferView(s), {g_spec.accessors_count} accessor(s)")
    print()

    # 2. BVH Specification
    print(" [2] BVH Motion Capture Specification Audit")
    print(" -------------------------------------------------------------------------")
    b_spec = report.bvh_spec
    print(f"  * Specification Compliant:    {'YES' if b_spec.valid else 'NO'}")
    print(f"  * Root Joint Name:            {b_spec.root_name}")
    print(f"  * Articulated Skeleton:       {b_spec.joint_count} joint(s), {b_spec.total_channels} active channel(s)")
    print(f"  * Motion Curve:               {b_spec.frame_count} frames @ {b_spec.fps:.2f} FPS (frame_time={b_spec.frame_time:.6f}s)")
    print()

    # 3. Bone Quaternion Statistics
    print(" [3] Bone Rotation Transformation Metrics (Quaternion Normalization)")
    print(" -------------------------------------------------------------------------")
    q = report.quaternion_stats
    print(f"  * Evaluated Rotations:        {q.total_samples:,} samples across {len(q.per_joint)} joints")
    print(f"  * Mean Norm |q|:              {q.mean_norm:.8f} (Ideal: 1.00000000)")
    print(f"  * Min Norm / Max Norm:        {q.min_norm:.8f} / {q.max_norm:.8f}")
    print(f"  * Norm Std Deviation:         {q.std_norm:.2e}")
    print(f"  * Max Deviation (| |q| - 1 |):{q.max_deviation:.2e}")
    print(f"  * Samples within 1e-3 tol:    {q.normalized_ratio_1e3:.2%}")
    print(f"  * Samples within 1e-4 tol:    {q.normalized_ratio_1e4:.2%}")
    print(f"  * Normalization Verified:     {'PASS' if q.is_valid else 'FAIL'}")
    print()

    # 4. Root Motion Bounds
    print(" [4] Root Motion Kinematic Bounds & Trajectory")
    print(" -------------------------------------------------------------------------")
    for r_stat in [report.gltf_root_motion, report.bvh_root_motion]:
        if r_stat is None:
            continue
        print(f"  Source: {r_stat.source}")
        print(f"  * Frame Count:                {r_stat.frame_count} frames")
        print(f"  * Min Bounding Coords [X,Y,Z]:[{r_stat.min_bounds[0]:.4f}, {r_stat.min_bounds[1]:.4f}, {r_stat.min_bounds[2]:.4f}] m")
        print(f"  * Max Bounding Coords [X,Y,Z]:[{r_stat.max_bounds[0]:.4f}, {r_stat.max_bounds[1]:.4f}, {r_stat.max_bounds[2]:.4f}] m")
        print(f"  * Bounding Extents (Span):    [{r_stat.extents[0]:.4f}, {r_stat.extents[1]:.4f}, {r_stat.extents[2]:.4f}] m")
        print(f"  * Center Position:            [{r_stat.center_pos[0]:.4f}, {r_stat.center_pos[1]:.4f}, {r_stat.center_pos[2]:.4f}] m")
        print(f"  * Total Trajectory Distance:  {r_stat.total_trajectory_distance:.4f} m")
        print(f"  * Max Frame-to-Frame Delta:   {r_stat.max_frame_displacement:.4f} m")
        print(f"  * Plausibility Check:         {'PASS' if r_stat.is_physically_plausible else 'FAIL'}")
        print()

    if report.cross_format_discrepancy:
        print(" [5] Cross-Format Kinematic Consistency")
        print(" -------------------------------------------------------------------------")
        print(f"  * Max Absolute Root Delta:    {report.cross_format_discrepancy['max_absolute_error_m']:.6f} m")
        print(f"  * Mean Absolute Root Delta:   {report.cross_format_discrepancy['mean_absolute_error_m']:.6f} m")
        print()

    # 5. Issues & Findings
    if report.issues:
        print(" [!] Validation Notices & Findings")
        print(" -------------------------------------------------------------------------")
        for iss in report.issues:
            sym = "[!]" if iss.severity == "WARNING" else "[X]"
            print(f"  {sym} [{iss.severity}] [{iss.component}] {iss.message}")
        print()

    print(f"{banner}\n")


def auto_discover_assets() -> Tuple[Optional[Path], Optional[Path]]:
    """Discovers reconstructed glTF and BVH pairs in the outputs/ directory."""
    root = Path(__file__).resolve().parent.parent
    outputs_dir = root / "outputs"
    if not outputs_dir.exists():
        return None, None

    gltf_files = sorted(outputs_dir.rglob("*.gltf"))
    bvh_files = sorted(outputs_dir.rglob("*.bvh"))

    # Try matching stems (e.g. reconstruction_33de54ad)
    for g in gltf_files:
        matching_bvh = [b for b in bvh_files if b.stem == g.stem]
        if matching_bvh:
            return g, matching_bvh[0]

    if gltf_files and bvh_files:
        return gltf_files[0], bvh_files[0]

    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leocherd Iris Energy 3D Rigging & Asset Reconstruction Benchmarking Tool"
    )
    parser.add_argument("--gltf", type=str, help="Path to reconstructed glTF 2.0 asset")
    parser.add_argument("--bvh", type=str, help="Path to reconstructed BVH asset")
    parser.add_argument("--json", type=str, help="Optional output path to write JSON report")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance threshold for quaternion normalization (default: 1e-3)",
    )
    parser.add_argument(
        "--max-jump",
        type=float,
        default=5.0,
        help="Maximum allowed frame-to-frame displacement in meters (default: 5.0)",
    )
    args = parser.parse_args()

    gltf_path = Path(args.gltf) if args.gltf else None
    bvh_path = Path(args.bvh) if args.bvh else None

    if not gltf_path or not bvh_path:
        disc_gltf, disc_bvh = auto_discover_assets()
        if disc_gltf and disc_bvh:
            gltf_path = gltf_path or disc_gltf
            bvh_path = bvh_path or disc_bvh
            print(f"[*] Auto-discovered reconstructed assets:\n    glTF: {gltf_path}\n    BVH:  {bvh_path}")
        else:
            print("[-] Error: Could not auto-discover reconstructed assets in outputs/.")
            print("    Please provide --gltf and --bvh arguments.")
            return 1

    report = benchmark_reconstruction(
        gltf_path,
        bvh_path,
        tolerance=args.tolerance,
        max_jump_threshold=args.max_jump,
    )

    print_benchmark_report(report)

    if args.json:
        out_json = Path(args.json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        # Convert dataclasses to dict
        data = asdict(report)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[*] Detailed JSON report written to: {out_json}")

    return 0 if report.overall_status == "PASS" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
