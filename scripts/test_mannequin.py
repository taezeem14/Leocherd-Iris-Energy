"""Test mannequin generation and export."""
import numpy as np
from iris.rigging.proxy_mesh import ProxyMeshGenerator
from iris.rigging.skeleton import Skeleton
from iris.morphology.human import HumanMorphology

m = HumanMorphology()
s = Skeleton.from_morphology(m)
v, n, idx, ji, jw = ProxyMeshGenerator.create_skinned_mannequin(s)

print(f"Vertices: {len(v)}")
print(f"Normals: {len(n)}")
print(f"Triangles: {len(idx)}")
print(f"Joint indices shape: {ji.shape}, dtype: {ji.dtype}")
print(f"Joint weights shape: {jw.shape}, dtype: {jw.dtype}")

# Verify weights
weight_sums = np.sum(jw, axis=1)
assert np.allclose(weight_sums, 1.0, atol=1e-4), f"Weight sums deviate from 1.0! Min: {np.min(weight_sums)}, Max: {np.max(weight_sums)}"

# Verify indices in bounds
assert np.min(idx) == 0
assert np.max(idx) == len(v) - 1

# Verify bounds
print(f"Min pos: {np.min(v, axis=0)}")
print(f"Max pos: {np.max(v, axis=0)}")
print("All mesh sanity checks passed successfully!")
