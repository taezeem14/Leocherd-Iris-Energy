"""
Procedural 3D Creature Trajectory Fixtures.

Generates mathematically rigorous, kinematically valid procedural motion trajectories
for all five core creature morphologies:
1. Human / Biped (upright vertical spine, arm swing, bipedal foot contacts)
2. Quadruped (horizontal spine, trotting gait, four limb contacts, articulated tail)
3. Serpentine (undulating traveling sinusoidal wave, constant vertebral lengths, ground contact)
4. Avian (wing flapping, S-neck, bipedal leg stepping/perching, tail)
5. Tentacular (central body hub with M radial undulating tentacle arms)

All generators support noise injection (noise_std) and guarantee zero-variance
bone rigidity invariants when noise_std == 0.0.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from iris.pose.base import Joint3D, PoseFrame


class TrajectoryResult(np.ndarray):
    """
    Subclass of np.ndarray of shape (T, N, 3) providing joint naming metadata,
    dictionary conversion, pose frame serialization, and rigidity validation.
    """

    joint_names: List[str]
    fps: float
    morphology_name: str
    bone_pairs: List[Tuple[str, str]]

    def __new__(
        cls,
        input_array: np.ndarray,
        joint_names: Optional[List[str]] = None,
        fps: float = 15.0,
        morphology_name: str = "",
        bone_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> "TrajectoryResult":
        obj = np.asarray(input_array, dtype=np.float32).view(cls)
        if obj.ndim == 2:
            obj = obj[np.newaxis, ...]  # (1, N, 3)
        if obj.ndim != 3 or obj.shape[-1] != 3:
            raise ValueError(f"Expected trajectory array of shape (T, N, 3), got {obj.shape}")

        N = obj.shape[1]
        obj.joint_names = list(joint_names) if joint_names is not None else [f"Joint_{i}" for i in range(N)]
        if len(obj.joint_names) != N:
            raise ValueError(f"Joint names count ({len(obj.joint_names)}) does not match joints count ({N})")

        obj.fps = float(fps)
        obj.morphology_name = morphology_name
        obj.bone_pairs = list(bone_pairs) if bone_pairs is not None else []
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.joint_names = getattr(obj, "joint_names", [])
        self.fps = getattr(obj, "fps", 15.0)
        self.morphology_name = getattr(obj, "morphology_name", "")
        self.bone_pairs = getattr(obj, "bone_pairs", [])

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Converts (T, N, 3) array into Dict[str, np.ndarray] of shape (T, 3)."""
        return {name: np.array(self[:, i, :], dtype=np.float32) for i, name in enumerate(self.joint_names)}

    def to_pose_frames(self) -> List[PoseFrame]:
        """Converts trajectory frames to a list of typed PoseFrame objects."""
        T = self.shape[0]
        poses: List[PoseFrame] = []
        dt = 1.0 / max(1.0, self.fps)
        for t in range(T):
            joints_dict: Dict[str, Joint3D] = {}
            for i, name in enumerate(self.joint_names):
                pos = self[t, i, :]
                joints_dict[name] = Joint3D(
                    name=name,
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    confidence=1.0,
                    visibility=1.0,
                )
            poses.append(
                PoseFrame(
                    frame_index=t,
                    timestamp_sec=float(t * dt),
                    subject_id=1,
                    morphology=self.morphology_name or "custom",
                    joints=joints_dict,
                )
            )
        return poses

    def compute_rigidity_invariants(self) -> Dict[Tuple[str, str], float]:
        """Computes distance variance for all registered parent-child bone pairs."""
        return compute_rigidity_invariants(self, self.bone_pairs, self.joint_names)

    def verify_bone_rigidity(self, max_variance: float = 1e-4) -> bool:
        """Verifies that all registered bone pairs have distance variance below threshold."""
        return verify_bone_rigidity(self, self.bone_pairs, max_variance=max_variance, joint_names=self.joint_names)


def _apply_noise(coords: np.ndarray, noise_std: float) -> np.ndarray:
    """Applies isotropic zero-mean Gaussian tracking noise to trajectory coordinates."""
    if noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=coords.shape).astype(np.float32)
        return coords + noise
    return coords


def get_morphology_bone_pairs(morphology_name: str, **kwargs: Any) -> List[Tuple[str, str]]:
    """Returns canonical parent-child joint pairs for the specified morphology."""
    name = morphology_name.lower()
    if name in ("human", "biped"):
        return [
            ("Hips", "Spine"),
            ("Spine", "Chest"),
            ("Chest", "Neck"),
            ("Neck", "Head"),
            ("Chest", "LeftShoulder"),
            ("LeftShoulder", "LeftArm"),
            ("LeftArm", "LeftForeArm"),
            ("LeftForeArm", "LeftHand"),
            ("Chest", "RightShoulder"),
            ("RightShoulder", "RightArm"),
            ("RightArm", "RightForeArm"),
            ("RightForeArm", "RightHand"),
            ("Hips", "LeftUpLeg"),
            ("LeftUpLeg", "LeftLeg"),
            ("LeftLeg", "LeftFoot"),
            ("Hips", "RightUpLeg"),
            ("RightUpLeg", "RightLeg"),
            ("RightLeg", "RightFoot"),
        ]
    elif name == "quadruped":
        return [
            ("Pelvis", "Spine_Lumbar"),
            ("Spine_Lumbar", "Spine_Thoracic"),
            ("Spine_Thoracic", "Neck"),
            ("Neck", "Head"),
            ("Pelvis", "Tail_Base"),
            ("Tail_Base", "Tail_Mid"),
            ("Tail_Mid", "Tail_Tip"),
            ("Spine_Thoracic", "FrontLeft_Scapula"),
            ("FrontLeft_Scapula", "FrontLeft_Knee"),
            ("FrontLeft_Knee", "FrontLeft_Ankle"),
            ("FrontLeft_Ankle", "FrontLeft_Paw"),
            ("Spine_Thoracic", "FrontRight_Scapula"),
            ("FrontRight_Scapula", "FrontRight_Knee"),
            ("FrontRight_Knee", "FrontRight_Ankle"),
            ("FrontRight_Ankle", "FrontRight_Paw"),
            ("Pelvis", "BackLeft_Hip"),
            ("BackLeft_Hip", "BackLeft_Knee"),
            ("BackLeft_Knee", "BackLeft_Hock"),
            ("BackLeft_Hock", "BackLeft_Paw"),
            ("Pelvis", "BackRight_Hip"),
            ("BackRight_Hip", "BackRight_Knee"),
            ("BackRight_Knee", "BackRight_Hock"),
            ("BackRight_Hock", "BackRight_Paw"),
        ]
    elif name == "serpentine":
        num_verts = kwargs.get("num_vertebrae", 12)
        return [(f"Vertebra_{i}", f"Vertebra_{i+1}") for i in range(num_verts - 1)]
    elif name == "avian":
        pairs = [
            ("Hips", "Spine"),
            ("Spine", "Chest"),
            ("Chest", "Neck"),
            ("Neck", "Head"),
            ("Chest", "LeftWing_Base"),
            ("LeftWing_Base", "LeftWing_Mid"),
            ("LeftWing_Mid", "LeftWing_Tip"),
            ("Chest", "RightWing_Base"),
            ("RightWing_Base", "RightWing_Mid"),
            ("RightWing_Mid", "RightWing_Tip"),
            ("Hips", "LeftThigh"),
            ("LeftThigh", "LeftKnee"),
            ("LeftKnee", "LeftAnkle"),
            ("LeftAnkle", "LeftFoot"),
            ("Hips", "RightThigh"),
            ("RightThigh", "RightKnee"),
            ("RightKnee", "RightAnkle"),
            ("RightAnkle", "RightFoot"),
        ]
        if kwargs.get("with_tail", True):
            pairs.extend([
                ("Hips", "Tail_1"),
                ("Tail_1", "Tail_2"),
                ("Tail_2", "Tail_3"),
            ])
        return pairs
    elif name == "tentacular":
        num_arms = kwargs.get("num_arms", 6)
        segs_per_arm = kwargs.get("segs_per_arm", 4)
        pairs = []
        for a in range(num_arms):
            pairs.append(("CentralBody", f"Tentacle_{a}_Seg_0"))
            for s in range(segs_per_arm - 1):
                pairs.append((f"Tentacle_{a}_Seg_{s}", f"Tentacle_{a}_Seg_{s+1}"))
        return pairs
    elif name == "branching_tree":
        levels = kwargs.get("levels", 3)
        branch_factor = kwargs.get("branch_factor", 2)
        pairs = []
        current_level = ["Root"]
        for lvl in range(1, levels):
            next_level = []
            for parent_name in current_level:
                for b in range(branch_factor):
                    node_name = f"Node_L{lvl}_{parent_name}_B{b}"
                    pairs.append((parent_name, node_name))
                    next_level.append(node_name)
            current_level = next_level
        return pairs
    return []


# =============================================================================
# 1. Human / Biped Trajectory Generator
# =============================================================================

def generate_human_trajectory(
    frames: int = 30,
    fps: float = 15.0,
    noise_std: float = 0.0,
    as_dict: bool = False,
    return_dict: bool = False,
) -> Union[TrajectoryResult, Dict[str, np.ndarray]]:
    """
    Generates a procedural walking human trajectory with 19 joints.
    Upright vertical spine, arm-leg antiphase swing, ground foot contacts.
    """
    joint_names = [
        "Hips", "Spine", "Chest", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftUpLeg", "LeftLeg", "LeftFoot",
        "RightUpLeg", "RightLeg", "RightFoot",
    ]
    bone_pairs = get_morphology_bone_pairs("human")

    T = max(2, frames)
    times = np.linspace(0.0, (T - 1) / max(1.0, fps), T, dtype=np.float32)
    omega = 2.0 * math.pi * 1.0  # 1.0 Hz stride frequency
    v_forward = 0.35  # Forward walking speed (m/s)

    coords = np.zeros((T, len(joint_names), 3), dtype=np.float32)
    j_idx = {name: i for i, name in enumerate(joint_names)}

    # Rest proportions
    l_spine = 0.20
    l_chest = 0.25
    l_neck = 0.13
    l_head = 0.14
    l_sh_x = 0.10
    l_arm = 0.28
    l_forearm = 0.27
    l_thigh = 0.45
    l_shin = 0.40

    for t_step, t in enumerate(times):
        # 1. Root / Hips: forward progression + vertical bounce + lateral sway
        sway = 0.02 * math.sin(omega * t)
        bounce = 0.02 * math.cos(2.0 * omega * t)
        z_root = v_forward * t
        hips = np.array([sway, 1.00 + bounce, z_root], dtype=np.float32)
        coords[t_step, j_idx["Hips"]] = hips

        # 2. Spine Chain: strictly vertical offsets from root
        spine = hips + np.array([0.0, l_spine, 0.0], dtype=np.float32)
        chest = spine + np.array([0.0, l_chest, 0.0], dtype=np.float32)
        neck = chest + np.array([0.0, l_neck, 0.0], dtype=np.float32)
        head = neck + np.array([0.0, l_head, 0.0], dtype=np.float32)

        coords[t_step, j_idx["Spine"]] = spine
        coords[t_step, j_idx["Chest"]] = chest
        coords[t_step, j_idx["Neck"]] = neck
        coords[t_step, j_idx["Head"]] = head

        # 3. Arms: Antiphase swing relative to stride
        arm_swing = 0.35 * math.sin(omega * t)
        l_sh = chest + np.array([-l_sh_x, 0.0, 0.0], dtype=np.float32)
        r_sh = chest + np.array([l_sh_x, 0.0, 0.0], dtype=np.float32)
        coords[t_step, j_idx["LeftShoulder"]] = l_sh
        coords[t_step, j_idx["RightShoulder"]] = r_sh

        # Left Arm: swings forward when sin(omega*t) > 0
        u_l_arm = np.array([0.0, -math.cos(arm_swing), math.sin(arm_swing)], dtype=np.float32)
        u_l_arm /= np.linalg.norm(u_l_arm)
        l_arm_pos = l_sh + l_arm * u_l_arm
        coords[t_step, j_idx["LeftArm"]] = l_arm_pos

        forearm_angle = arm_swing + 0.15 * max(0.0, math.sin(omega * t))
        u_l_fa = np.array([0.0, -math.cos(forearm_angle), math.sin(forearm_angle)], dtype=np.float32)
        u_l_fa /= np.linalg.norm(u_l_fa)
        l_fa_pos = l_arm_pos + l_forearm * u_l_fa
        coords[t_step, j_idx["LeftForeArm"]] = l_fa_pos
        coords[t_step, j_idx["LeftHand"]] = l_fa_pos + 0.10 * u_l_fa

        # Right Arm: opposite phase
        u_r_arm = np.array([0.0, -math.cos(-arm_swing), math.sin(-arm_swing)], dtype=np.float32)
        u_r_arm /= np.linalg.norm(u_r_arm)
        r_arm_pos = r_sh + l_arm * u_r_arm
        coords[t_step, j_idx["RightArm"]] = r_arm_pos

        r_fa_angle = -arm_swing + 0.15 * max(0.0, math.sin(-arm_swing))
        u_r_fa = np.array([0.0, -math.cos(r_fa_angle), math.sin(r_fa_angle)], dtype=np.float32)
        u_r_fa /= np.linalg.norm(u_r_fa)
        r_fa_pos = r_arm_pos + l_forearm * u_r_fa
        coords[t_step, j_idx["RightForeArm"]] = r_fa_pos
        coords[t_step, j_idx["RightHand"]] = r_fa_pos + 0.10 * u_r_fa

        # 4. Legs: Stride walking kinematics with exact bone length preservation
        leg_swing_l = -0.35 * math.sin(omega * t)
        l_up_leg = hips + np.array([-0.10, -0.05, 0.0], dtype=np.float32)
        r_up_leg = hips + np.array([0.10, -0.05, 0.0], dtype=np.float32)
        coords[t_step, j_idx["LeftUpLeg"]] = l_up_leg
        coords[t_step, j_idx["RightUpLeg"]] = r_up_leg

        # Left Leg
        u_thigh_l = np.array([0.0, -math.cos(leg_swing_l), math.sin(leg_swing_l)], dtype=np.float32)
        u_thigh_l /= np.linalg.norm(u_thigh_l)
        l_knee = l_up_leg + l_thigh * u_thigh_l
        coords[t_step, j_idx["LeftLeg"]] = l_knee

        knee_flex_l = leg_swing_l - 0.25 * max(0.0, math.sin(omega * t))
        u_shin_l = np.array([0.0, -math.cos(knee_flex_l), math.sin(knee_flex_l)], dtype=np.float32)
        u_shin_l /= np.linalg.norm(u_shin_l)
        l_foot = l_knee + l_shin * u_shin_l
        coords[t_step, j_idx["LeftFoot"]] = l_foot

        # Right Leg (opposite phase)
        leg_swing_r = -leg_swing_l
        u_thigh_r = np.array([0.0, -math.cos(leg_swing_r), math.sin(leg_swing_r)], dtype=np.float32)
        u_thigh_r /= np.linalg.norm(u_thigh_r)
        r_knee = r_up_leg + l_thigh * u_thigh_r
        coords[t_step, j_idx["RightLeg"]] = r_knee

        knee_flex_r = leg_swing_r - 0.25 * max(0.0, math.sin(-omega * t))
        u_shin_r = np.array([0.0, -math.cos(knee_flex_r), math.sin(knee_flex_r)], dtype=np.float32)
        u_shin_r /= np.linalg.norm(u_shin_r)
        r_foot = r_knee + l_shin * u_shin_r
        coords[t_step, j_idx["RightFoot"]] = r_foot

    coords = _apply_noise(coords, noise_std)
    result = TrajectoryResult(
        coords,
        joint_names=joint_names,
        fps=fps,
        morphology_name="human",
        bone_pairs=bone_pairs,
    )
    if as_dict or return_dict:
        return result.as_dict()
    return result


# =============================================================================
# 2. Quadruped Trajectory Generator
# =============================================================================

def generate_quadruped_trajectory(
    frames: int = 30,
    fps: float = 15.0,
    noise_std: float = 0.0,
    as_dict: bool = False,
    return_dict: bool = False,
) -> Union[TrajectoryResult, Dict[str, np.ndarray]]:
    """
    Generates a procedural trotting quadruped trajectory with 24 joints.
    Horizontal spine, diagonal gait synchronization (FL+BR vs FR+BL), wagging tail.
    """
    joint_names = [
        "Pelvis", "Spine_Lumbar", "Spine_Thoracic", "Neck", "Head",
        "Tail_Base", "Tail_Mid", "Tail_Tip",
        "FrontLeft_Scapula", "FrontLeft_Knee", "FrontLeft_Ankle", "FrontLeft_Paw",
        "FrontRight_Scapula", "FrontRight_Knee", "FrontRight_Ankle", "FrontRight_Paw",
        "BackLeft_Hip", "BackLeft_Knee", "BackLeft_Hock", "BackLeft_Paw",
        "BackRight_Hip", "BackRight_Knee", "BackRight_Hock", "BackRight_Paw",
    ]
    bone_pairs = get_morphology_bone_pairs("quadruped")

    T = max(2, frames)
    times = np.linspace(0.0, (T - 1) / max(1.0, fps), T, dtype=np.float32)
    omega = 2.0 * math.pi * 1.2  # 1.2 Hz trotting gait
    v_forward = 0.40  # Forward progression (m/s)

    coords = np.zeros((T, len(joint_names), 3), dtype=np.float32)
    j_idx = {name: i for i, name in enumerate(joint_names)}

    l_lumbar = 0.20
    l_thoracic = 0.25
    l_neck = 0.32
    l_head = 0.22
    l_tail_base = 0.15
    l_tail_mid = 0.20
    l_tail_tip = 0.20

    l_scapula_y = 0.27
    l_knee_y = 0.25
    l_ankle_y = 0.15

    for t_step, t in enumerate(times):
        # 1. Root / Pelvis (horizontal spine starting at -Z, heading toward +Z)
        bounce = 0.015 * math.sin(2.0 * omega * t)
        z_pelvis = v_forward * t - 0.40
        pelvis = np.array([0.0, 0.70 + bounce, z_pelvis], dtype=np.float32)
        coords[t_step, j_idx["Pelvis"]] = pelvis

        # 2. Horizontal Spine
        lumbar = pelvis + np.array([0.0, 0.02, l_lumbar], dtype=np.float32)
        thoracic = lumbar + np.array([0.0, 0.03, l_thoracic], dtype=np.float32)

        # Neck extends up and forward (+Y, +Z)
        u_neck = np.array([0.0, 0.30, 0.95], dtype=np.float32)
        u_neck /= np.linalg.norm(u_neck)
        neck = thoracic + l_neck * u_neck

        # Head extends forward and slightly up
        u_head = np.array([0.0, 0.35, 0.94], dtype=np.float32)
        u_head /= np.linalg.norm(u_head)
        head = neck + l_head * u_head

        coords[t_step, j_idx["Spine_Lumbar"]] = lumbar
        coords[t_step, j_idx["Spine_Thoracic"]] = thoracic
        coords[t_step, j_idx["Neck"]] = neck
        coords[t_step, j_idx["Head"]] = head

        # 3. Tail: trailing dynamic pendulum (lagging wave in X)
        tail_wave = 0.15 * math.sin(omega * t)
        u_tail1 = np.array([tail_wave, -0.3, -0.9], dtype=np.float32)
        u_tail1 /= np.linalg.norm(u_tail1)
        tail_base = pelvis + l_tail_base * u_tail1

        u_tail2 = np.array([1.5 * tail_wave, -0.4, -0.9], dtype=np.float32)
        u_tail2 /= np.linalg.norm(u_tail2)
        tail_mid = tail_base + l_tail_mid * u_tail2

        u_tail3 = np.array([2.0 * tail_wave, -0.5, -0.8], dtype=np.float32)
        u_tail3 /= np.linalg.norm(u_tail3)
        tail_tip = tail_mid + l_tail_tip * u_tail3

        coords[t_step, j_idx["Tail_Base"]] = tail_base
        coords[t_step, j_idx["Tail_Mid"]] = tail_mid
        coords[t_step, j_idx["Tail_Tip"]] = tail_tip

        # 4. Limbs: Trotting gait (diagonal pairs: FL+BR at phase 0, FR+BL at phase pi)
        phi_fl = omega * t
        phi_fr = omega * t + math.pi
        phi_bl = phi_fr
        phi_br = phi_fl

        def build_limb(anchor: np.ndarray, phase: float, is_front: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            swing = 0.30 * math.sin(phase)
            u_upper = np.array([0.0, -math.cos(swing), math.sin(swing)], dtype=np.float32)
            u_upper /= np.linalg.norm(u_upper)
            knee = anchor + l_scapula_y * u_upper

            knee_angle = swing - 0.20 * max(0.0, math.sin(phase))
            u_mid = np.array([0.0, -math.cos(knee_angle), math.sin(knee_angle)], dtype=np.float32)
            u_mid /= np.linalg.norm(u_mid)
            ankle = knee + l_knee_y * u_mid

            paw = ankle + l_ankle_y * np.array([0.0, -0.95, 0.3], dtype=np.float32) / math.sqrt(0.95**2 + 0.3**2)
            return knee, ankle, paw

        # FrontLeft
        fl_scap = thoracic + np.array([-0.15, -0.03, 0.05], dtype=np.float32)
        coords[t_step, j_idx["FrontLeft_Scapula"]] = fl_scap
        fl_knee, fl_ankle, fl_paw = build_limb(fl_scap, phi_fl, True)
        coords[t_step, j_idx["FrontLeft_Knee"]] = fl_knee
        coords[t_step, j_idx["FrontLeft_Ankle"]] = fl_ankle
        coords[t_step, j_idx["FrontLeft_Paw"]] = fl_paw

        # FrontRight
        fr_scap = thoracic + np.array([0.15, -0.03, 0.05], dtype=np.float32)
        coords[t_step, j_idx["FrontRight_Scapula"]] = fr_scap
        fr_knee, fr_ankle, fr_paw = build_limb(fr_scap, phi_fr, True)
        coords[t_step, j_idx["FrontRight_Knee"]] = fr_knee
        coords[t_step, j_idx["FrontRight_Ankle"]] = fr_ankle
        coords[t_step, j_idx["FrontRight_Paw"]] = fr_paw

        # BackLeft
        bl_hip = pelvis + np.array([-0.15, -0.02, 0.02], dtype=np.float32)
        coords[t_step, j_idx["BackLeft_Hip"]] = bl_hip
        bl_knee, bl_hock, bl_paw = build_limb(bl_hip, phi_bl, False)
        coords[t_step, j_idx["BackLeft_Knee"]] = bl_knee
        coords[t_step, j_idx["BackLeft_Hock"]] = bl_hock
        coords[t_step, j_idx["BackLeft_Paw"]] = bl_paw

        # BackRight
        br_hip = pelvis + np.array([0.15, -0.02, 0.02], dtype=np.float32)
        coords[t_step, j_idx["BackRight_Hip"]] = br_hip
        br_knee, br_hock, br_paw = build_limb(br_hip, phi_br, False)
        coords[t_step, j_idx["BackRight_Knee"]] = br_knee
        coords[t_step, j_idx["BackRight_Hock"]] = br_hock
        coords[t_step, j_idx["BackRight_Paw"]] = br_paw

    coords = _apply_noise(coords, noise_std)
    result = TrajectoryResult(
        coords,
        joint_names=joint_names,
        fps=fps,
        morphology_name="quadruped",
        bone_pairs=bone_pairs,
    )
    if as_dict or return_dict:
        return result.as_dict()
    return result


# =============================================================================
# 3. Serpentine Trajectory Generator
# =============================================================================

def generate_serpentine_trajectory(
    frames: int = 30,
    fps: float = 15.0,
    num_vertebrae: int = 12,
    noise_std: float = 0.0,
    as_dict: bool = False,
    return_dict: bool = False,
) -> Union[TrajectoryResult, Dict[str, np.ndarray]]:
    """
    Generates a procedural serpentine locomotion trajectory with N vertebrae.
    Propagates a lateral traveling sinusoidal wave on ground (Y ≈ 0.05).
    Guarantees mathematically exact vertebral segment lengths.
    """
    N = max(4, num_vertebrae)
    joint_names = [f"Vertebra_{i}" for i in range(N)]
    bone_pairs = get_morphology_bone_pairs("serpentine", num_vertebrae=N)

    T = max(2, frames)
    times = np.linspace(0.0, (T - 1) / max(1.0, fps), T, dtype=np.float32)
    step_length = 2.0 / (N - 1)  # 2.0 meters total rest length
    speed = 2.0  # Wave temporal speed
    wavelength = 1.4  # Spatial wavelength in meters
    max_turn_angle = 0.45  # Peak articulation angle in radians

    coords = np.zeros((T, N, 3), dtype=np.float32)

    for t_step, t in enumerate(times):
        # Head / Root: Vertebra_0 advances forward along Z
        z_head = 0.3 * t
        coords[t_step, 0] = np.array([0.0, 0.05, z_head], dtype=np.float32)

        # Forward kinematics for vertebral chain ensuring strictly unit direction vectors
        for i in range(N - 1):
            s = i * step_length
            # Tangent angle for segment i: traveling sine wave theta(s, t)
            theta = max_turn_angle * math.sin((2.0 * math.pi / wavelength) * s - speed * t)
            # Unit direction vector: [sin(theta), 0.0, cos(theta)] has norm == 1.0 identically
            u_dir = np.array([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float32)
            coords[t_step, i + 1] = coords[t_step, i] + step_length * u_dir

    coords = _apply_noise(coords, noise_std)
    result = TrajectoryResult(
        coords,
        joint_names=joint_names,
        fps=fps,
        morphology_name="serpentine",
        bone_pairs=bone_pairs,
    )
    if as_dict or return_dict:
        return result.as_dict()
    return result


# =============================================================================
# 4. Avian Trajectory Generator
# =============================================================================

def generate_avian_trajectory(
    frames: int = 30,
    fps: float = 15.0,
    noise_std: float = 0.0,
    as_dict: bool = False,
    return_dict: bool = False,
) -> Union[TrajectoryResult, Dict[str, np.ndarray]]:
    """
    Generates a procedural avian flight & perch trajectory with 21 joints.
    Chest wing flapping with wing folding, digitigrade legs, and stabilizing tail.
    """
    joint_names = [
        "Hips", "Spine", "Chest", "Neck", "Head",
        "LeftWing_Base", "LeftWing_Mid", "LeftWing_Tip",
        "RightWing_Base", "RightWing_Mid", "RightWing_Tip",
        "LeftThigh", "LeftKnee", "LeftAnkle", "LeftFoot",
        "RightThigh", "RightKnee", "RightAnkle", "RightFoot",
        "Tail_1", "Tail_2", "Tail_3",
    ]
    bone_pairs = get_morphology_bone_pairs("avian", with_tail=True)

    T = max(2, frames)
    times = np.linspace(0.0, (T - 1) / max(1.0, fps), T, dtype=np.float32)
    omega_flap = 2.0 * math.pi * 1.5  # 1.5 Hz wing flap frequency
    v_forward = 0.50  # Airspeed forward progression

    coords = np.zeros((T, len(joint_names), 3), dtype=np.float32)
    j_idx = {name: i for i, name in enumerate(joint_names)}

    l_spine = 0.20
    l_chest = 0.25
    l_neck = 0.20
    l_head = 0.15
    l_wing_mid = 0.35
    l_wing_tip = 0.35
    l_thigh = 0.40
    l_shin = 0.35
    l_foot_len = 0.15
    l_tail_seg = 0.25

    for t_step, t in enumerate(times):
        # 1. Body & Spine
        flap_bounce = 0.04 * math.sin(omega_flap * t)
        z_hips = v_forward * t - 0.10
        hips = np.array([0.0, 1.10 + flap_bounce, z_hips], dtype=np.float32)
        coords[t_step, j_idx["Hips"]] = hips

        spine = hips + np.array([0.0, l_spine * 0.8, l_spine * 0.6], dtype=np.float32)
        chest = spine + np.array([0.0, l_chest * 0.7, l_chest * 0.7], dtype=np.float32)
        neck = chest + np.array([0.0, l_neck * 0.75, l_neck * 0.65], dtype=np.float32)
        head = neck + np.array([0.0, l_head * 0.5, l_head * 0.85], dtype=np.float32)

        coords[t_step, j_idx["Spine"]] = spine
        coords[t_step, j_idx["Chest"]] = chest
        coords[t_step, j_idx["Neck"]] = neck
        coords[t_step, j_idx["Head"]] = head

        # 2. Wings: Flapping kinematics with folding on upstroke
        flap_angle = 0.55 * math.sin(omega_flap * t)
        tip_fold = flap_angle + 0.30 * math.cos(omega_flap * t)

        lw_base = chest + np.array([-0.15, 0.0, 0.0], dtype=np.float32)
        rw_base = chest + np.array([0.15, 0.0, 0.0], dtype=np.float32)
        coords[t_step, j_idx["LeftWing_Base"]] = lw_base
        coords[t_step, j_idx["RightWing_Base"]] = rw_base

        # Left Wing (extends along -X and flaps in Y)
        u_l_mid = np.array([-math.cos(flap_angle), math.sin(flap_angle), -0.1], dtype=np.float32)
        u_l_mid /= np.linalg.norm(u_l_mid)
        lw_mid = lw_base + l_wing_mid * u_l_mid
        coords[t_step, j_idx["LeftWing_Mid"]] = lw_mid

        u_l_tip = np.array([-math.cos(tip_fold), math.sin(tip_fold), -0.2], dtype=np.float32)
        u_l_tip /= np.linalg.norm(u_l_tip)
        coords[t_step, j_idx["LeftWing_Tip"]] = lw_mid + l_wing_tip * u_l_tip

        # Right Wing (mirror in +X)
        u_r_mid = np.array([math.cos(flap_angle), math.sin(flap_angle), -0.1], dtype=np.float32)
        u_r_mid /= np.linalg.norm(u_r_mid)
        rw_mid = rw_base + l_wing_mid * u_r_mid
        coords[t_step, j_idx["RightWing_Mid"]] = rw_mid

        u_r_tip = np.array([math.cos(tip_fold), math.sin(tip_fold), -0.2], dtype=np.float32)
        u_r_tip /= np.linalg.norm(u_r_tip)
        coords[t_step, j_idx["RightWing_Tip"]] = rw_mid + l_wing_tip * u_r_tip

        # 3. Digitigrade Legs: retracted in flight with gentle rhythm
        leg_tuck = 0.15 * math.sin(omega_flap * t)
        l_thigh_pt = hips + np.array([-0.12, -0.05, 0.0], dtype=np.float32)
        r_thigh_pt = hips + np.array([0.12, -0.05, 0.0], dtype=np.float32)
        coords[t_step, j_idx["LeftThigh"]] = l_thigh_pt
        coords[t_step, j_idx["RightThigh"]] = r_thigh_pt

        u_thigh = np.array([0.0, -0.85, 0.5 + leg_tuck], dtype=np.float32)
        u_thigh /= np.linalg.norm(u_thigh)
        l_knee = l_thigh_pt + l_thigh * u_thigh
        r_knee = r_thigh_pt + l_thigh * u_thigh
        coords[t_step, j_idx["LeftKnee"]] = l_knee
        coords[t_step, j_idx["RightKnee"]] = r_knee

        u_shin = np.array([0.0, -0.70, -0.70], dtype=np.float32)
        u_shin /= np.linalg.norm(u_shin)
        l_ankle = l_knee + l_shin * u_shin
        r_ankle = r_knee + l_shin * u_shin
        coords[t_step, j_idx["LeftAnkle"]] = l_ankle
        coords[t_step, j_idx["RightAnkle"]] = r_ankle

        u_foot = np.array([0.0, -0.2, 0.98], dtype=np.float32)
        u_foot /= np.linalg.norm(u_foot)
        l_foot = l_ankle + l_foot_len * u_foot
        r_foot = r_ankle + l_foot_len * u_foot
        coords[t_step, j_idx["LeftFoot"]] = l_foot
        coords[t_step, j_idx["RightFoot"]] = r_foot

        # 4. Tail
        u_tail = np.array([0.0, -0.2, -0.98], dtype=np.float32)
        u_tail /= np.linalg.norm(u_tail)
        tail_1 = hips + l_tail_seg * u_tail
        tail_2 = tail_1 + l_tail_seg * u_tail
        tail_3 = tail_2 + l_tail_seg * u_tail
        coords[t_step, j_idx["Tail_1"]] = tail_1
        coords[t_step, j_idx["Tail_2"]] = tail_2
        coords[t_step, j_idx["Tail_3"]] = tail_3

    coords = _apply_noise(coords, noise_std)
    result = TrajectoryResult(
        coords,
        joint_names=joint_names,
        fps=fps,
        morphology_name="avian",
        bone_pairs=bone_pairs,
    )
    if as_dict or return_dict:
        return result.as_dict()
    return result


# =============================================================================
# 5. Tentacular Trajectory Generator
# =============================================================================

def generate_tentacular_trajectory(
    frames: int = 30,
    fps: float = 15.0,
    num_arms: int = 6,
    segs_per_arm: int = 4,
    noise_std: float = 0.0,
    as_dict: bool = False,
    return_dict: bool = False,
) -> Union[TrajectoryResult, Dict[str, np.ndarray]]:
    """
    Generates a procedural tentacular monster trajectory with M radial arms and K segments each.
    Central body hub with undulating traveling waves radiating outwards.
    """
    M = max(3, num_arms)
    K = max(2, segs_per_arm)

    joint_names = ["CentralBody"]
    for a in range(M):
        for s in range(K):
            joint_names.append(f"Tentacle_{a}_Seg_{s}")

    bone_pairs = get_morphology_bone_pairs("tentacular", num_arms=M, segs_per_arm=K)

    T = max(2, frames)
    times = np.linspace(0.0, (T - 1) / max(1.0, fps), T, dtype=np.float32)
    omega = 2.0 * math.pi * 0.8  # 0.8 Hz undulation
    l_seg = 0.25

    coords = np.zeros((T, len(joint_names), 3), dtype=np.float32)
    j_idx = {name: i for i, name in enumerate(joint_names)}

    angle_step = (2.0 * math.pi) / M

    for t_step, t in enumerate(times):
        # 1. Central Body hub hovering with vertical pulse
        y_pulse = 0.05 * math.sin(omega * t)
        center = np.array([0.0, 1.00 + y_pulse, 0.0], dtype=np.float32)
        coords[t_step, j_idx["CentralBody"]] = center

        # 2. Radial Arms
        for a in range(M):
            azimuth = a * angle_step
            parent_pos = center

            for s in range(K):
                # Elevation wave (draping down and undulating)
                phi = -0.35 - 0.25 * math.sin(omega * t - 0.7 * s)
                # Azimuth oscillation
                psi = azimuth + 0.15 * math.cos(omega * t - 0.7 * s)

                # Normalized unit direction vector
                cos_phi = math.cos(phi)
                u_dir = np.array([
                    cos_phi * math.cos(psi),
                    math.sin(phi),
                    cos_phi * math.sin(psi),
                ], dtype=np.float32)
                u_dir /= np.linalg.norm(u_dir)

                seg_pos = parent_pos + l_seg * u_dir
                name = f"Tentacle_{a}_Seg_{s}"
                coords[t_step, j_idx[name]] = seg_pos
                parent_pos = seg_pos

    coords = _apply_noise(coords, noise_std)
    result = TrajectoryResult(
        coords,
        joint_names=joint_names,
        fps=fps,
        morphology_name="tentacular",
        bone_pairs=bone_pairs,
    )
    if as_dict or return_dict:
        return result.as_dict()
    return result


# =============================================================================
# 6. Branching Tree Trajectory Generator (Alien Tree Wind Sway)
# =============================================================================

def generate_branching_tree_trajectory(
    frames: int = 30,
    fps: float = 15.0,
    levels: int = 3,
    branch_factor: int = 2,
    noise_std: float = 0.0,
    as_dict: bool = False,
    return_dict: bool = False,
) -> Union[TrajectoryResult, Dict[str, np.ndarray]]:
    """
    Generates a procedural alien branching tree swaying in the wind.
    Guarantees strict constant bone segment lengths (zero distance variance) when noise_std == 0.0.
    """
    bone_pairs = get_morphology_bone_pairs("branching_tree", levels=levels, branch_factor=branch_factor)

    # Collect joint names hierarchically
    joint_names = ["Root"]
    current_level = ["Root"]
    for lvl in range(1, levels):
        next_level = []
        for parent_name in current_level:
            for b in range(branch_factor):
                node_name = f"Node_L{lvl}_{parent_name}_B{b}"
                joint_names.append(node_name)
                next_level.append(node_name)
        current_level = next_level

    T = max(2, frames)
    times = np.linspace(0.0, (T - 1) / max(1.0, fps), T, dtype=np.float32)
    omega = 2.0 * math.pi * 0.5  # 0.5 Hz wind sway oscillation

    # Compute rest positions matching create_branching_tree_skeleton
    rest_positions: Dict[str, np.ndarray] = {"Root": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    current_level_nodes = [("Root", np.array([0.0, 0.0, 0.0], dtype=np.float32))]
    for lvl in range(1, levels):
        next_level_nodes = []
        seg_length = 0.5 / lvl
        spread = 0.3 / lvl
        for parent_name, parent_pos in current_level_nodes:
            for b in range(branch_factor):
                sign = -1.0 if b == 0 else 1.0
                node_name = f"Node_L{lvl}_{parent_name}_B{b}"
                pos = parent_pos + np.array([sign * spread, seg_length, 0.0], dtype=np.float32)
                rest_positions[node_name] = pos
                next_level_nodes.append((node_name, pos))
        current_level_nodes = next_level_nodes

    # Compute rest relative vectors and lengths
    bone_defs: Dict[str, Tuple[str, np.ndarray, float, int]] = {}
    for parent, child in bone_pairs:
        v_rest = rest_positions[child] - rest_positions[parent]
        L = float(np.linalg.norm(v_rest))
        u_rest = v_rest / max(1e-6, L)
        lvl = int(child.split("_L")[1].split("_")[0]) if "_L" in child else 1
        bone_defs[child] = (parent, u_rest.astype(np.float32), L, lvl)

    j_idx = {name: i for i, name in enumerate(joint_names)}
    coords = np.zeros((T, len(joint_names), 3), dtype=np.float32)

    for t_step in range(T):
        tau = float(times[t_step])
        # Root gentle sway
        coords[t_step, j_idx["Root"]] = np.array([
            0.02 * math.sin(omega * tau),
            0.0,
            0.01 * math.cos(omega * tau),
        ], dtype=np.float32)

        # Forward kinematics for each child
        for parent, child in bone_pairs:
            p_pos = coords[t_step, j_idx[parent]]
            _, u_rest, L, lvl = bone_defs[child]

            # Wind deflection angle around Z (side sway) and X (gust sway)
            theta_z = (0.12 * lvl) * math.sin(omega * tau - 0.35 * lvl)
            theta_x = (0.06 * lvl) * math.cos(1.25 * omega * tau - 0.25 * lvl)

            cz, sz = math.cos(theta_z), math.sin(theta_z)
            cx, sx = math.cos(theta_x), math.sin(theta_x)
            R = np.array([
                [cz, -sz * cx, sz * sx],
                [sz, cz * cx, -cz * sx],
                [0.0, sx, cx],
            ], dtype=np.float32)

            u_swayed = R @ u_rest
            u_swayed /= np.linalg.norm(u_swayed)

            c_pos = p_pos + L * u_swayed
            coords[t_step, j_idx[child]] = c_pos

    coords = _apply_noise(coords, noise_std)
    result = TrajectoryResult(
        coords,
        joint_names=joint_names,
        fps=fps,
        morphology_name="branching_tree",
        bone_pairs=bone_pairs,
    )
    if as_dict or return_dict:
        return result.as_dict()
    return result


# =============================================================================
# Rigidity and Invariant Verification Utilities
# =============================================================================

def compute_rigidity_invariants(
    trajectory: Union[np.ndarray, Dict[str, np.ndarray]],
    bone_pairs: List[Tuple[str, str]],
    joint_names: Optional[List[str]] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Computes temporal distance variance Var_t(||P_child(t) - P_parent(t)||)
    for each connected bone pair. For rigid bones, variance should approach 0.0.
    """
    coords, names = _normalize_trajectory_input(trajectory, joint_names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    variances: Dict[Tuple[str, str], float] = {}
    for parent, child in bone_pairs:
        if parent in name_to_idx and child in name_to_idx:
            p_pos = coords[:, name_to_idx[parent], :]
            c_pos = coords[:, name_to_idx[child], :]
            dist = np.linalg.norm(c_pos - p_pos, axis=1)
            variances[(parent, child)] = float(np.var(dist))
    return variances


def verify_bone_rigidity(
    trajectory: Union[np.ndarray, Dict[str, np.ndarray]],
    bone_pairs: List[Tuple[str, str]],
    max_variance: float = 1e-4,
    joint_names: Optional[List[str]] = None,
) -> bool:
    """Returns True if every registered bone pair has temporal distance variance <= max_variance."""
    variances = compute_rigidity_invariants(trajectory, bone_pairs, joint_names)
    if not variances:
        return True
    return all(v <= max_variance for v in variances.values())


def trajectory_to_dict(coords: np.ndarray, joint_names: List[str]) -> Dict[str, np.ndarray]:
    """Helper converting (T, N, 3) array into dictionary of (T, 3) arrays."""
    return {name: np.array(coords[:, i, :], dtype=np.float32) for i, name in enumerate(joint_names)}


def trajectory_to_array(traj_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Helper converting dictionary of (T, 3) arrays into (T, N, 3) array and names list."""
    names = list(traj_dict.keys())
    arrays = [np.asarray(traj_dict[k], dtype=np.float32) for k in names]
    min_T = min(a.shape[0] for a in arrays)
    coords = np.stack([a[:min_T] for a in arrays], axis=1)
    return coords, names


def _normalize_trajectory_input(
    trajectory: Union[np.ndarray, Dict[str, np.ndarray]],
    joint_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Internal helper to convert input into standard (T, N, 3) and names list."""
    if isinstance(trajectory, dict):
        return trajectory_to_array(trajectory)
    coords = np.asarray(trajectory, dtype=np.float32)
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]
    N = coords.shape[1]
    names = list(joint_names) if joint_names is not None else [f"Joint_{i}" for i in range(N)]
    return coords, names
