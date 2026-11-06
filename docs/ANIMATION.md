# Animation Engine & Retargeting

## 1. Animation Representation

Animation in Iris Energy is stored as discrete keyframe curves or sampled timeseries:
- **Root Translation Curve:** $\vec{T}_{root}(t) \in \mathbb{R}^3$
- **Bone Rotation Curves:** $q_b(t) \in \mathbb{H}$ (unit quaternion for each bone $b$)
- **Bone Scale Curves:** $\vec{S}_b(t) \in \mathbb{R}^3$ (typically $\vec{1}$ unless morphology incorporates scaling)

## 2. Quaternion Slerp & Temporal Resampling

When resampling keyframes or filling occluded intervals:
$$\text{Slerp}(q_1, q_2, u) = \frac{\sin((1-u)\theta)}{\sin \theta} q_1 + \frac{\sin(u\theta)}{\sin \theta} q_2$$
where $\cos \theta = q_1 \cdot q_2$.
To prevent antipodal rotation flips (taking the long path around $S^3$), Iris Energy ensures $q_1 \cdot q_2 \ge 0$, inverting $q_2 \leftarrow -q_2$ if the inner product is negative.

## 3. Retargeting Subsystem

Retargeting allows motion derived from one subject morphology to drive a different target rig:
1. **Joint Name Mapping Table:** Maps source joints to target bone names.
2. **Rest Pose Normalization:** Computes orientation offset $\Delta q = q_{target, rest}^{-1} \otimes q_{source, rest}$.
3. **Scale Factor Estimation:** Normalizes root displacement by limb length ratios:
   $$s = \frac{L_{target, leg}}{L_{source, leg}}$$
4. **Rotation Transfer:**
   $$q_{target}(t) = \Delta q \otimes q_{source}(t)$$
