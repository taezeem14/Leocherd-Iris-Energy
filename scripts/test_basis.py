"""Test quaternion orientation math."""
import numpy as np
import math

def normalize(v):
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(max(1e-8, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1e-8, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1e-8, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)

def build_basis(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Build orthonormal 3x3 basis [X, Y, Z] where Y is primary and Z is roughly secondary."""
    y = normalize(primary)
    # Project secondary orthogonal to y
    z = secondary - np.dot(secondary, y) * y
    z = normalize(z)
    x = normalize(np.cross(y, z))
    z = normalize(np.cross(x, y))  # re-orthogonalize
    return np.column_stack([x, y, z])

# Test identity
R_id = np.eye(3, dtype=np.float32)
q_id = matrix_to_quat(R_id)
print("Identity quat:", q_id, "norm:", np.linalg.norm(q_id))
assert np.allclose(q_id, [1, 0, 0, 0], atol=1e-4)

# Test basis
B1 = build_basis(np.array([0, 1, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
print("B1:\n", B1)
assert np.allclose(B1, np.eye(3), atol=1e-4)

# Test rotation from rest to cur
# If arm is pointing downwards [0, -1, 0] instead of [0, 1, 0], with pole [0, 0, 1]
B2 = build_basis(np.array([0, -1, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
R_rot = B2 @ B1.T
q_rot = matrix_to_quat(R_rot)
print("Rotation quat:", q_rot, "norm:", np.linalg.norm(q_rot))
print("All orientation tests passed!")
