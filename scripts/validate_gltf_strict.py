import json
import base64
import numpy as np

with open("outputs/online_walking_run/reconstruction_skinned_walking.gltf") as f:
    doc = json.load(f)

uri = doc["buffers"][0]["uri"]
buf = base64.b64decode(uri.split(",", 1)[1])

# Check every accessor with min/max
errors = []
warnings = []

type_map = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16
}

component_map = {
    5123: np.uint16,
    5126: np.float32,
}

for i, acc in enumerate(doc["accessors"]):
    bv = doc["bufferViews"][acc["bufferView"]]
    offset = bv["byteOffset"] + acc.get("byteOffset", 0)
    dtype = component_map[acc["componentType"]]
    num_comps = type_map[acc["type"]]
    count = acc["count"]
    total_elements = count * num_comps
    
    data = np.frombuffer(buf[offset:offset + total_elements * dtype().itemsize], dtype=dtype)
    if num_comps > 1:
        data = data.reshape((count, num_comps))
    
    if "min" in acc and "max" in acc:
        if num_comps == 1:
            actual_min = float(np.min(data))
            actual_max = float(np.max(data))
            declared_min = float(acc["min"][0])
            declared_max = float(acc["max"][0])
            if actual_min < declared_min:
                errors.append(f"Acc {i}: actual_min {actual_min} < declared_min {declared_min}")
            if actual_max > declared_max:
                errors.append(f"Acc {i}: actual_max {actual_max} > declared_max {declared_max}")
        else:
            actual_min = np.min(data, axis=0)
            actual_max = np.max(data, axis=0)
            for c in range(num_comps):
                if actual_min[c] < acc["min"][c]:
                    errors.append(f"Acc {i} comp {c}: actual_min {actual_min[c]} < declared_min {acc['min'][c]}")
                if actual_max[c] > acc["max"][c]:
                    errors.append(f"Acc {i} comp {c}: actual_max {actual_max[c]} > declared_max {acc['max'][c]}")

# Check joints and weights
for mesh in doc["meshes"]:
    for prim in mesh["primitives"]:
        j_acc_idx = prim["attributes"]["JOINTS_0"]
        w_acc_idx = prim["attributes"]["WEIGHTS_0"]
        j_acc = doc["accessors"][j_acc_idx]
        w_acc = doc["accessors"][w_acc_idx]
        
        j_bv = doc["bufferViews"][j_acc["bufferView"]]
        w_bv = doc["bufferViews"][w_acc["bufferView"]]
        
        j_data = np.frombuffer(buf[j_bv["byteOffset"]:j_bv["byteOffset"] + j_acc["count"]*8], dtype=np.uint16).reshape((-1, 4))
        w_data = np.frombuffer(buf[w_bv["byteOffset"]:w_bv["byteOffset"] + w_acc["count"]*16], dtype=np.float32).reshape((-1, 4))
        
        # Check: if weight is 0, joint MUST be 0
        zero_w = w_data == 0.0
        non_zero_j_with_zero_w = (j_data != 0) & zero_w
        violations = np.sum(non_zero_j_with_zero_w)
        if violations > 0:
            warnings.append(f"Found {violations} joints used with zero weight and non-zero index!")
            
        # Check sum of weights
        sums = np.sum(w_data, axis=1)
        if not np.allclose(sums, 1.0, atol=1e-4):
            errors.append(f"Weights do not sum to 1.0! Max deviation: {np.max(np.abs(sums - 1.0))}")

print(f"Validation finished.")
print(f"Errors ({len(errors)}):", errors)
print(f"Warnings ({len(warnings)}):", warnings)
