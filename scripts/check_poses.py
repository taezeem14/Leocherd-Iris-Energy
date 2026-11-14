import pickle
import numpy as np
from iris.rigging.skeleton import Skeleton
from iris.morphology.human import HumanMorphology

morph = HumanMorphology()
skel = Skeleton.from_morphology(morph)

# Let's inspect the video detections
from iris.video.ingestion import VideoIngestionStage
from iris.pose.estimator import PoseEstimationStage
from iris.core.pipeline import ExecutionContext

ctx = ExecutionContext(
    input_path="data/samples/online_walking_sample.mp4",
    output_dir="outputs/test_inspect",
    morphology_name="human",
    target_fps=12,
    config_overrides={"max_frames": 120}
)

v_stage = VideoIngestionStage()
p_stage = PoseEstimationStage()

v_stage.run(ctx)
p_stage.run(ctx)

print(f"Total pose frames: {len(ctx.pose_sequence)}")
for f in [0, 20, 39, 45, 60, 75, 90, 110]:
    p = ctx.pose_sequence[f]
    confs = [j.confidence for j in p.joints.values()]
    avg_conf = np.mean(confs)
    print(f"\n--- Frame {f} (avg conf: {avg_conf:.2f}) ---")
    for side in ['left', 'right']:
        h = p.joints.get(f'{side}_hip')
        k = p.joints.get(f'{side}_knee')
        a = p.joints.get(f'{side}_ankle')
        if h and k and a:
            print(f"  {side.upper()} LEG: Hip=({h.x:.2f}, {h.y:.2f}, {h.z:.2f}), Knee=({k.x:.2f}, {k.y:.2f}, {k.z:.2f}), Ankle=({a.x:.2f}, {a.y:.2f}, {a.z:.2f})")


