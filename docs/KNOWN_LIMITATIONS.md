# Known Limitations & Scientific Boundaries

## 1. Monocular Ambiguity
- Monocular RGB video does not measure true depth directly.
- Depth values along the camera axis ($Z$) are inferred from apparent foreshortening and neural landmark regression.
- Extreme foreshortening (e.g. limbs pointing straight down the optical axis) has higher uncertainty than lateral motion in the image plane.

## 2. Occlusion Handling
- When a limb is fully occluded behind the torso or an environmental object, neural confidence drops.
- Iris Energy marks occluded joints with low confidence ($c < 0.3$) and uses temporal kinematic extrapolation to bridge short gaps (< 10 frames). Extended occlusions maintain the last known plausible angle or blend towards rest pose.

## 3. Hardware Constraints on 2-Thread / Integrated Graphics Systems
- On legacy 2-core processors without AVX (e.g. Intel Pentium B950), frame processing rates are constrained to ~5-15 FPS.
- Video sampling is automatically downscaled to 15 FPS and resolution limited to 720p to maintain responsive execution and prevent memory exhaustion.
