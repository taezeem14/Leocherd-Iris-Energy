import numpy as np
from scipy.interpolate import interp1d

def impute_with_rest(
    positions: np.ndarray,
    confidences: np.ndarray,
    timestamps: np.ndarray,
    min_conf: float = 0.35,
    default_rest_pos = None,
    blend_frames: int = 8,
) -> np.ndarray:
    T, D = positions.shape
    if T < 2:
        return positions.copy()

    valid_mask = confidences >= min_conf
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        if default_rest_pos is not None:
            return np.tile(default_rest_pos, (T, 1)).astype(np.float32)
        return positions.copy()

    times_list = []
    vals_list = []

    first_idx = valid_indices[0]
    if first_idx > 0:
        if default_rest_pos is not None:
            times_list.append(timestamps[0])
            vals_list.append(default_rest_pos)
            blend_start_idx = max(0, first_idx - blend_frames)
            if blend_start_idx > 0:
                times_list.append(timestamps[blend_start_idx])
                vals_list.append(default_rest_pos)
        else:
            times_list.append(timestamps[0])
            vals_list.append(positions[first_idx])

    for idx in valid_indices:
        times_list.append(timestamps[idx])
        vals_list.append(positions[idx])

    last_idx = valid_indices[-1]
    if last_idx < T - 1:
        if default_rest_pos is not None:
            blend_end_idx = min(T - 1, last_idx + blend_frames)
            if blend_end_idx < T - 1:
                times_list.append(timestamps[blend_end_idx])
                vals_list.append(default_rest_pos)
            times_list.append(timestamps[-1])
            vals_list.append(default_rest_pos)
        else:
            times_list.append(timestamps[-1])
            vals_list.append(positions[last_idx])

    t_arr = np.array(times_list, dtype=np.float64)
    v_arr = np.array(vals_list, dtype=np.float32)

    unique_t, unique_indices = np.unique(t_arr, return_index=True)
    unique_v = v_arr[unique_indices]

    if len(unique_t) < 2:
        return positions.copy()

    out = np.zeros_like(positions)
    for d in range(D):
        interp_func = interp1d(
            unique_t,
            unique_v[:, d],
            kind="linear",
            bounds_error=False,
            fill_value=(unique_v[0, d], unique_v[-1, d]),
        )
        out[:, d] = interp_func(timestamps)

    return out

# Test with 120 frames where only frames 45-75 are valid
T = 120
t = np.linspace(0, 10, T)
pos = np.zeros((T, 3))
pos[45:76] = [0.1, 0.2, 0.3]
conf = np.zeros(T)
conf[45:76] = 0.9

rest = np.array([0.0, 0.9, 0.0], dtype=np.float32)
res = impute_with_rest(pos, conf, t, min_conf=0.35, default_rest_pos=rest)

print("Frame 0:", res[0])    # should be rest: [0, 0.9, 0]
print("Frame 30:", res[30])  # should be rest: [0, 0.9, 0]
print("Frame 40:", np.round(res[40], 2))  # blending towards [0.1, 0.2, 0.3]
print("Frame 50:", np.round(res[50], 2))  # valid: [0.1, 0.2, 0.3]
print("Frame 80:", np.round(res[80], 2))  # blending back towards rest
print("Frame 90:", res[90])  # should be rest: [0, 0.9, 0]
print("Frame 119:", res[119]) # should be rest: [0, 0.9, 0]
