"""
Universal Morphology Understanding & Classification Module for Leocherd Iris Energy.
Classifies biological and procedural morphologies from kinematic poses and 3D landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from iris.core.exceptions import MorphologyError
from iris.core.logging import get_logger
from iris.morphology.base import BaseMorphology
from iris.morphology.biped import BipedMorphology
from iris.morphology.custom import CustomMorphology
from iris.morphology.human import HumanMorphology
from iris.morphology.quadruped import QuadrupedMorphology
from iris.morphology.serpentine import SerpentineMorphology
from iris.morphology.tentacular import TentacularMorphology
from iris.pose.base import Joint3D, PoseFrame

logger = get_logger("iris.morphology.classifier")

# Type alias for Avian morphology
AvianMorphology = BipedMorphology


class MorphologyType(str, Enum):
    """Categorical classification of anatomical and procedural morphologies."""

    HUMAN = "human"
    QUADRUPED = "quadruped"
    AVIAN = "avian"
    SERPENTINE = "serpentine"
    TENTACULAR = "tentacular"
    CUSTOM = "custom"


@dataclass
class MorphologyPrediction:
    """Prediction result containing inferred morphology type, confidence score, and metrics."""

    predicted_type: MorphologyType
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)
    morphology_name: str = ""

    def __post_init__(self) -> None:
        if not self.morphology_name and self.predicted_type:
            self.morphology_name = (
                self.predicted_type.value
                if isinstance(self.predicted_type, MorphologyType)
                else str(self.predicted_type)
            )
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


class MorphologyClassifier:
    """
    Universal morphology classifier analyzing geometric and kinematic feature vectors across time.

    Extracts:
    1. Spine Orientation: Angle between principal spine axis and vertical gravity vector (+Y).
       Upright (~0°-25°) indicates biped/human; horizontal (~65°-90°) indicates quadruped.
    2. Bounding Box Proportions: Height / Length aspect ratio.
    3. Ground Contact Extremities: Detects how many distinct trajectory clusters make persistent
       contact with the ground plane (Y ≈ 0). 2 contacts = biped/avian; 4 contacts = quadruped;
       0 distinct limb contacts = serpentine.
    4. Wave Propagation Metric: Measures whether motion consists of undulating traveling waves
       along a single axis (serpentine).
    5. Limb Count & Radial Symmetry (tentacular vs biped).
    """

    def __init__(self, confidence_threshold: float = 0.35) -> None:
        self.confidence_threshold = confidence_threshold

    def classify_from_poses(self, poses: List[PoseFrame]) -> MorphologyPrediction:
        """
        Extracts geometric and kinematic feature vectors across a sequence of pose frames
        and predicts the morphology type with confidence metrics.
        """
        if not poses:
            return MorphologyPrediction(
                predicted_type=MorphologyType.CUSTOM,
                confidence=0.0,
                metrics={
                    "aspect_ratio": 0.0,
                    "spine_tilt_degrees": 0.0,
                    "ground_contact_count": 0.0,
                    "limb_chain_count": 0.0,
                    "curvature_variance": 0.0,
                    "wave_propagation": 0.0,
                    "radial_symmetry": 0.0,
                },
                morphology_name=MorphologyType.CUSTOM.value,
            )

        # 1. Spine Orientation
        spine_tilt_degrees = self._calculate_spine_tilt(poses)

        # 2. Bounding Box Proportions (Height / Length)
        aspect_ratio = self._calculate_bounding_box_aspect_ratio(poses)

        # 3. Ground Contact Extremities
        ground_contact_count, fraction_on_ground = self._calculate_ground_contacts(poses)

        # 4. Wave Propagation Metric & Curvature Variance
        wave_propagation, curvature_variance = self._calculate_wave_propagation(poses)

        # 5. Limb Count & Radial Symmetry
        limb_chain_count, radial_symmetry = self._calculate_limb_and_radial_symmetry(poses)

        # If majority of joints are flat on ground, enforce 0 limb contacts
        if fraction_on_ground > 0.50:
            ground_contact_count = 0
            limb_chain_count = 0

        metrics = {
            "aspect_ratio": round(aspect_ratio, 4),
            "spine_tilt_degrees": round(spine_tilt_degrees, 2),
            "ground_contact_count": float(ground_contact_count),
            "limb_chain_count": float(limb_chain_count),
            "curvature_variance": round(curvature_variance, 6),
            "wave_propagation": round(wave_propagation, 4),
            "radial_symmetry": round(radial_symmetry, 4),
        }

        # Classification decision scoring
        scores = self._score_morphologies(metrics, fraction_on_ground)

        best_type = max(scores, key=scores.get)
        confidence = float(np.clip(scores[best_type], 0.0, 1.0))

        if confidence < self.confidence_threshold:
            best_type = MorphologyType.CUSTOM
            confidence = 0.5

        return MorphologyPrediction(
            predicted_type=best_type,
            confidence=confidence,
            metrics=metrics,
            morphology_name=best_type.value,
        )

    def classify_from_landmarks(self, raw_joints_dict: Dict[str, Joint3D]) -> MorphologyPrediction:
        """
        Single-frame static posture fallback classifier.
        """
        single_frame = PoseFrame(
            frame_index=0,
            timestamp_sec=0.0,
            joints=raw_joints_dict,
        )
        return self.classify_from_poses([single_frame])

    def get_morphology_instance(
        self, prediction: Union[MorphologyPrediction, MorphologyType, str]
    ) -> BaseMorphology:
        """
        Instantiates the corresponding morphology plugin from iris.morphology
        (HumanMorphology, QuadrupedMorphology, SerpentineMorphology, TentacularMorphology, etc.).
        """
        if isinstance(prediction, MorphologyPrediction):
            pred_type = prediction.predicted_type
            name = prediction.morphology_name
        elif isinstance(prediction, MorphologyType):
            pred_type = prediction
            name = prediction.value
        elif isinstance(prediction, str):
            name = prediction.lower()
            try:
                pred_type = MorphologyType(name)
            except ValueError:
                pred_type = MorphologyType.CUSTOM
        else:
            raise MorphologyError(f"Unsupported prediction argument type: {type(prediction)}")

        if pred_type == MorphologyType.HUMAN:
            return HumanMorphology()
        elif pred_type == MorphologyType.QUADRUPED:
            return QuadrupedMorphology()
        elif pred_type == MorphologyType.SERPENTINE:
            return SerpentineMorphology()
        elif pred_type == MorphologyType.TENTACULAR:
            return TentacularMorphology()
        elif pred_type == MorphologyType.AVIAN:
            return BipedMorphology(with_tail=True)
        elif pred_type == MorphologyType.CUSTOM:
            return CustomMorphology({"root_joint": "Root", "joints": {"Root": [0.0, 0.0, 0.0]}})
        else:
            # Fallback to dynamic lookup in registry
            from iris.morphology import MorphologyRegistry

            return MorphologyRegistry.get(name)

    # -------------------------------------------------------------------------
    # Feature Extraction Methods
    # -------------------------------------------------------------------------

    def _calculate_spine_tilt(self, poses: List[PoseFrame]) -> float:
        """
        Calculates angle between principal spine axis and vertical gravity vector (+Y).
        Upright (~0°-25°) indicates biped/human; horizontal (~65°-90°) indicates quadruped.
        """
        tilts: List[float] = []

        # Common anatomical landmark identifiers
        head_names = ["head", "nose", "neck", "spine_thoracic", "vertebra_11", "vertebra_7"]
        root_names = ["hips", "pelvis", "vertebra_0", "centralbody", "mid_hip"]

        for pose in poses:
            joints = pose.joints
            if not joints:
                continue

            # 1. Semantic Spine Vector
            head_joint: Optional[Joint3D] = None
            root_joint: Optional[Joint3D] = None

            for h_name in head_names:
                for k, v in joints.items():
                    if h_name == k.lower():
                        head_joint = v
                        break
                if head_joint:
                    break

            for r_name in root_names:
                for k, v in joints.items():
                    if r_name == k.lower():
                        root_joint = v
                        break
                if root_joint:
                    break

            if head_joint and root_joint and head_joint != root_joint:
                spine_vec = head_joint.to_numpy() - root_joint.to_numpy()
                norm = float(np.linalg.norm(spine_vec))
                if norm > 1e-4:
                    spine_unit = spine_vec / norm
                    # Angle with vertical gravity vector (+Y = [0, 1, 0])
                    # We compute acute angle between spine axis and vertical
                    cos_tilt = abs(spine_unit[1])
                    tilt_deg = math.degrees(math.acos(float(np.clip(cos_tilt, 0.0, 1.0))))
                    tilts.append(tilt_deg)
                    continue

            # 2. Geometric PCA Fallback
            coords = pose.get_coordinates_matrix()
            if len(coords) >= 3:
                coords_centered = coords - np.mean(coords, axis=0)
                cov = np.cov(coords_centered, rowvar=False)
                if cov.shape == (3, 3):
                    eig_vals, eig_vecs = np.linalg.eigh(cov)
                    principal_axis = eig_vecs[:, np.argmax(eig_vals)]
                    cos_tilt = abs(principal_axis[1])
                    tilt_deg = math.degrees(math.acos(float(np.clip(cos_tilt, 0.0, 1.0))))
                    tilts.append(tilt_deg)

        if not tilts:
            return 0.0
        return float(np.mean(tilts))

    def _calculate_bounding_box_aspect_ratio(self, poses: List[PoseFrame]) -> float:
        """
        Calculates bounding box Height / Length aspect ratio across poses.
        """
        ratios: List[float] = []
        for pose in poses:
            coords = pose.get_coordinates_matrix()
            if len(coords) == 0:
                continue
            min_bounds = np.min(coords, axis=0)
            max_bounds = np.max(coords, axis=0)

            height = max_bounds[1] - min_bounds[1]  # Y axis
            length_x = max_bounds[0] - min_bounds[0]  # X axis
            length_z = max_bounds[2] - min_bounds[2]  # Z axis
            horizontal_len = max(length_x, length_z, 1e-4)

            aspect = height / horizontal_len
            ratios.append(aspect)

        if not ratios:
            return 1.0
        return float(np.mean(ratios))

    def _calculate_ground_contacts(self, poses: List[PoseFrame]) -> Tuple[int, float]:
        """
        Detects how many distinct trajectory clusters make persistent contact with the ground plane (Y ≈ 0).
        2 contacts = biped/avian; 4 contacts = quadruped; 0 distinct limb contacts = serpentine.
        """
        all_pts: List[np.ndarray] = []
        for pose in poses:
            for j in pose.joints.values():
                if j.confidence >= 0.2:
                    all_pts.append(j.to_numpy())

        if not all_pts:
            return 0, 0.0

        pts_arr = np.array(all_pts)
        min_y = float(np.min(pts_arr[:, 1]))
        max_y = float(np.max(pts_arr[:, 1]))
        height = max(max_y - min_y, 0.1)

        # Ground plane threshold: within lowest 15% of body height or 0.15m from minimum
        y_thresh = min_y + min(0.18 * height, 0.20)

        # Fraction of total observations on ground
        on_ground_mask = pts_arr[:, 1] <= y_thresh
        fraction_on_ground = float(np.mean(on_ground_mask))

        # If majority of joints are on the ground (e.g. continuous snake body)
        if fraction_on_ground > 0.50:
            return 0, fraction_on_ground

        # Semantic Paw / Foot Check
        semantic_contacts = self._detect_semantic_ground_contacts(poses, y_thresh)
        if semantic_contacts is not None:
            return semantic_contacts, fraction_on_ground

        # Spatial clustering in horizontal XZ plane
        frame_counts: List[int] = []
        for pose in poses:
            ground_pts = [
                j.to_numpy()
                for j in pose.joints.values()
                if j.confidence >= 0.2 and j.y <= y_thresh
            ]
            if not ground_pts:
                continue

            xz_pts = np.array([[p[0], p[2]] for p in ground_pts])
            clusters: List[List[np.ndarray]] = []
            cluster_dist_thresh = 0.18

            for pt in xz_pts:
                assigned = False
                for c in clusters:
                    centroid = np.mean(c, axis=0)
                    if np.linalg.norm(pt - centroid) < cluster_dist_thresh:
                        c.append(pt)
                        assigned = True
                        break
                if not assigned:
                    clusters.append([pt])
            frame_counts.append(len(clusters))

        if not frame_counts:
            return 0, fraction_on_ground

        median_count = int(np.round(np.median(frame_counts)))
        return median_count, fraction_on_ground

    def _detect_semantic_ground_contacts(
        self, poses: List[PoseFrame], y_thresh: float
    ) -> Optional[int]:
        """Checks for standard anatomical foot / paw joint naming conventions."""
        foot_keywords_human = ["foot", "ankle", "toe", "heel"]
        quad_paw_keywords = [
            ("frontleft", "fl_paw"),
            ("frontright", "fr_paw"),
            ("backleft", "bl_paw"),
            ("backright", "br_paw"),
        ]

        # Sample first few frames
        for pose in poses[:5]:
            joints = pose.joints
            lower_names = [k.lower() for k in joints.keys()]

            # Quadruped paw checks
            paws_found = 0
            for kw_group in quad_paw_keywords:
                if any(any(kw in name for kw in kw_group) for name in lower_names):
                    paws_found += 1
            if paws_found >= 3:
                return 4

            # Human / Biped foot checks
            has_left_foot = any(
                ("left" in name or "l_" in name or name.startswith("l"))
                and any(fk in name for fk in foot_keywords_human)
                for name in lower_names
            )
            has_right_foot = any(
                ("right" in name or "r_" in name or name.startswith("r"))
                and any(fk in name for fk in foot_keywords_human)
                for name in lower_names
            )
            if has_left_foot and has_right_foot:
                return 2

        return None

    def _calculate_wave_propagation(self, poses: List[PoseFrame]) -> Tuple[float, float]:
        """
        Measures whether motion consists of undulating traveling waves along a single axis (serpentine).
        Returns: (wave_propagation_score, curvature_variance)
        """
        if not poses:
            return 0.0, 0.0

        num_frames = len(poses)
        first_pose = poses[0]
        joint_names = list(first_pose.joints.keys())
        num_joints = len(joint_names)

        if num_joints < 3:
            return 0.0, 0.0

        # Sort joints along principal axis of first frame
        coords_0 = first_pose.get_coordinates_matrix()
        c_mean = np.mean(coords_0, axis=0)
        c_centered = coords_0 - c_mean
        cov = np.cov(c_centered, rowvar=False)

        if cov.shape != (3, 3):
            return 0.0, 0.0

        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal_idx = int(np.argmax(eig_vals))
        principal_axis = eig_vecs[:, principal_idx]

        # Lateral axis (perpendicular in horizontal plane)
        lateral_axis = np.array([-principal_axis[2], 0.0, principal_axis[0]])
        if np.linalg.norm(lateral_axis) < 1e-4:
            lateral_axis = np.array([1.0, 0.0, 0.0])
        else:
            lateral_axis /= np.linalg.norm(lateral_axis)

        # Projections along principal axis
        projections = [float(np.dot(coords_0[i], principal_axis)) for i in range(num_joints)]
        sorted_indices = np.argsort(projections)

        # Compute curvature variance across ordered chain
        curvatures_per_frame: List[float] = []
        for pose in poses:
            ordered_coords = np.array([pose.joints[joint_names[idx]].to_numpy() for idx in sorted_indices])
            if len(ordered_coords) < 3:
                continue
            p_prev = ordered_coords[:-2]
            p_curr = ordered_coords[1:-1]
            p_next = ordered_coords[2:]

            d1 = np.linalg.norm(p_curr - p_prev, axis=1)
            d2 = np.linalg.norm(p_next - p_curr, axis=1)
            ds_sq = np.maximum(d1 * d2, 1e-4)

            curv_vecs = p_next - 2.0 * p_curr + p_prev
            curv_mags = np.linalg.norm(curv_vecs, axis=1) / ds_sq
            if len(curv_mags) > 0:
                curvatures_per_frame.append(float(np.var(curv_mags)))

        avg_curv_variance = float(np.mean(curvatures_per_frame)) if curvatures_per_frame else 0.0

        # Traveling wave correlation across time
        if num_frames < 3 or num_joints < 4:
            # Fallback for static single-frame posture
            wave_score = min(1.0, avg_curv_variance * 50.0) if avg_curv_variance > 0.005 else 0.0
            return float(wave_score), avg_curv_variance

        # Extract 2D matrix of lateral displacements: [joint, frame]
        lat_matrix = np.zeros((num_joints, num_frames), dtype=np.float32)
        for t, pose in enumerate(poses):
            for i, idx in enumerate(sorted_indices):
                pos = pose.joints[joint_names[idx]].to_numpy()
                lat_matrix[i, t] = np.dot(pos, lateral_axis)

        # Compute spatial gradient d/ds and temporal gradient d/dt
        # For an undulating wave f(s - ct), d/dt is proportional to -c * d/ds
        d_ds = (lat_matrix[2:, 1:-1] - lat_matrix[:-2, 1:-1]) / 2.0
        d_dt = (lat_matrix[1:-1, 2:] - lat_matrix[1:-1, :-2]) / 2.0

        flat_ds = d_ds.flatten()
        flat_dt = d_dt.flatten()

        std_ds = float(np.std(flat_ds))
        std_dt = float(np.std(flat_dt))

        if std_ds > 1e-4 and std_dt > 1e-4:
            corr = float(np.corrcoef(flat_ds, flat_dt)[0, 1])
            # Wave propagation metric is the absolute correlation between space and time derivatives
            wave_prop = abs(corr) if not np.isnan(corr) else 0.0
        else:
            wave_prop = 0.0

        return float(np.clip(wave_prop, 0.0, 1.0)), avg_curv_variance

    def _calculate_limb_and_radial_symmetry(self, poses: List[PoseFrame]) -> Tuple[int, float]:
        """
        Determines limb count and radial symmetry around the vertical central axis.
        """
        first_pose = poses[0]
        joint_names = [k.lower() for k in first_pose.joints.keys()]

        # 1. Semantic Check
        # Serpentine check: only vertebrae
        if all("vertebra" in k or "vert_" in k or "spine" in k for k in joint_names):
            return 0, 0.0

        # Tentacular check: tentacles
        tentacle_indices = set()
        for k in joint_names:
            if "tentacle" in k:
                # e.g., "Tentacle_0_Seg_1" -> index 0
                parts = k.split("_")
                for p in parts:
                    if p.isdigit():
                        tentacle_indices.add(int(p))
                        break
        if len(tentacle_indices) >= 3:
            return len(tentacle_indices), 0.85

        # 2. Geometric Radial Symmetry Analysis
        coords = first_pose.get_coordinates_matrix()
        if len(coords) < 4:
            return 2, 0.0

        center = np.mean(coords, axis=0)
        # Extremity detection: distance from center in XZ plane
        xz_dists = np.linalg.norm(coords[:, [0, 2]] - center[[0, 2]], axis=1)
        max_dist = float(np.max(xz_dists))

        if max_dist < 1e-4:
            return 0, 0.0

        # Find distal extremity joints (outer 40% of radial span)
        extremity_mask = xz_dists >= 0.60 * max_dist
        extremity_pts = coords[extremity_mask]

        if len(extremity_pts) < 3:
            return 4, 0.1

        # Angular distribution in XZ plane: [-pi, pi]
        angles = np.arctan2(extremity_pts[:, 2] - center[2], extremity_pts[:, 0] - center[0])

        # Bin into 8 sectors of 45 degrees
        num_bins = 8
        bin_counts, _ = np.histogram(angles, bins=num_bins, range=(-math.pi, math.pi))
        occupied_sectors = int(np.count_nonzero(bin_counts))

        # Circular uniformity / resultant vector length R_bar
        r_bar = float(np.abs(np.mean(np.exp(1j * angles))))
        # High radial symmetry: evenly spaced across many sectors, low resultant R_bar
        radial_symmetry = (occupied_sectors / num_bins) * max(0.0, 1.0 - r_bar)

        if radial_symmetry >= 0.55 and occupied_sectors >= 5:
            limb_chain_count = occupied_sectors
        else:
            limb_chain_count = 4

        return limb_chain_count, float(np.clip(radial_symmetry, 0.0, 1.0))

    # -------------------------------------------------------------------------
    # Classification Scoring
    # -------------------------------------------------------------------------

    def _score_morphologies(
        self, metrics: Dict[str, float], fraction_on_ground: float
    ) -> Dict[MorphologyType, float]:
        """
        Evaluates weighted evidence scores for each morphology family.
        """
        tilt = metrics["spine_tilt_degrees"]
        ar = metrics["aspect_ratio"]
        contacts = int(metrics["ground_contact_count"])
        limbs = int(metrics["limb_chain_count"])
        wave = metrics["wave_propagation"]
        curv = metrics["curvature_variance"]
        rad = metrics["radial_symmetry"]

        scores: Dict[MorphologyType, float] = {
            MorphologyType.HUMAN: 0.0,
            MorphologyType.QUADRUPED: 0.0,
            MorphologyType.SERPENTINE: 0.0,
            MorphologyType.TENTACULAR: 0.0,
            MorphologyType.AVIAN: 0.0,
            MorphologyType.CUSTOM: 0.05,
        }

        # ---------------------------------------------------------------------
        # 1. SERPENTINE Evaluation
        # ---------------------------------------------------------------------
        serp_score = 0.0
        if contacts == 0 or fraction_on_ground > 0.50:
            serp_score += 0.35
        if limbs == 0:
            serp_score += 0.25
        if ar < 0.35:
            serp_score += 0.20
        if wave > 0.40 or curv > 0.01:
            serp_score += 0.15
        if tilt >= 55.0:
            serp_score += 0.10
        scores[MorphologyType.SERPENTINE] = serp_score

        # ---------------------------------------------------------------------
        # 2. HUMAN Evaluation
        # ---------------------------------------------------------------------
        human_score = 0.0
        # Upright spine tilt: 0° - 25°
        if tilt <= 25.0:
            human_score += 0.35
        elif tilt <= 35.0:
            human_score += 0.20 * (35.0 - tilt) / 10.0 + 0.15

        # Bounding box aspect ratio: tall and upright (Height >= Length)
        if ar >= 1.0:
            human_score += 0.25
        elif ar >= 0.8:
            human_score += 0.15

        # Ground contact: 2 distinct feet
        if contacts == 2:
            human_score += 0.25

        # 4 limbs (2 arms, 2 legs)
        if limbs == 4:
            human_score += 0.15
        elif limbs == 2:
            human_score += 0.10

        # Low wave propagation and bilateral symmetry
        if wave < 0.35 and rad < 0.45:
            human_score += 0.05

        scores[MorphologyType.HUMAN] = human_score

        # ---------------------------------------------------------------------
        # 3. QUADRUPED Evaluation
        # ---------------------------------------------------------------------
        quad_score = 0.0
        # Horizontal spine tilt: 65° - 90°
        if tilt >= 65.0:
            quad_score += 0.35
        elif tilt >= 50.0:
            quad_score += 0.20 * (tilt - 50.0) / 15.0 + 0.15

        # Ground contacts: 4 paws
        if contacts == 4:
            quad_score += 0.30

        # Aspect ratio: Length > Height (typically < 0.90)
        if ar < 0.90:
            quad_score += 0.20
        elif ar < 1.10:
            quad_score += 0.10

        # 4 limbs
        if limbs == 4:
            quad_score += 0.15

        # Low wave propagation
        if wave < 0.35:
            quad_score += 0.05

        scores[MorphologyType.QUADRUPED] = quad_score

        # ---------------------------------------------------------------------
        # 4. TENTACULAR Evaluation
        # ---------------------------------------------------------------------
        tent_score = 0.0
        if rad >= 0.55:
            tent_score += 0.45
        elif rad >= 0.40:
            tent_score += 0.25

        if limbs >= 5:
            tent_score += 0.40
        elif limbs >= 3:
            tent_score += 0.20

        if contacts == 0 or contacts >= 5:
            tent_score += 0.15

        scores[MorphologyType.TENTACULAR] = tent_score

        # ---------------------------------------------------------------------
        # 5. AVIAN Evaluation
        # ---------------------------------------------------------------------
        avian_score = 0.0
        if contacts == 2:
            avian_score += 0.35
        # Forward leaning or bipedal tilt
        if 25.0 <= tilt <= 60.0:
            avian_score += 0.30
        elif tilt < 25.0:
            avian_score += 0.15

        if 0.60 <= ar <= 1.40:
            avian_score += 0.20

        if limbs in (2, 4):
            avian_score += 0.15

        scores[MorphologyType.AVIAN] = avian_score

        return scores


class MorphologyClassificationStage:
    """
    Pipeline stage integrating Universal Morphology Classification into the Iris execution pipeline.
    """

    def __init__(self, name: str = "MorphologyClassificationStage") -> None:
        self.name = name
        self.classifier = MorphologyClassifier()

    def run(self, context: Any, event_bus: Optional[Any] = None) -> Any:
        if not getattr(context, "pose_sequence", None):
            raise MorphologyError("No pose sequence available in ExecutionContext.")

        prediction = self.classifier.classify_from_poses(context.pose_sequence)
        context.morphology_prediction = prediction
        context.morphology_name = prediction.morphology_name
        context.morphology = self.classifier.get_morphology_instance(prediction)
        return context
