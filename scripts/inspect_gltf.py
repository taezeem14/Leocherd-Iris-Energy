"""Quick glTF inspection script."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/online_walking_run/reconstruction_skinned_walking.gltf"
d = json.load(open(path))

print(f"Nodes: {len(d.get('nodes', []))}")
print(f"Meshes: {len(d.get('meshes', []))}")
print(f"Skins: {len(d.get('skins', []))}")
print(f"Materials: {len(d.get('materials', []))}")
print(f"Animations: {len(d.get('animations', []))}")
print(f"Accessors: {len(d.get('accessors', []))}")
print(f"BufferViews: {len(d.get('bufferViews', []))}")

# Check scene nodes
for s in d.get('scenes', []):
    print(f"Scene nodes: {s.get('nodes', [])}")

# Check mesh details
for i, m in enumerate(d.get('meshes', [])):
    for j, p in enumerate(m.get('primitives', [])):
        attrs = p.get('attributes', {})
        print(f"  Mesh[{i}].prim[{j}]: attrs={list(attrs.keys())}, indices_acc={p.get('indices')}")
        # Check vertex count from POSITION accessor
        pos_acc = attrs.get('POSITION')
        if pos_acc is not None:
            acc = d['accessors'][pos_acc]
            print(f"    POSITION accessor: count={acc['count']}, type={acc['type']}")

# Check nodes for mesh/skin assignments
for i, node in enumerate(d.get('nodes', [])):
    if 'mesh' in node or 'skin' in node:
        print(f"  Node[{i}] '{node.get('name','')}': mesh={node.get('mesh')}, skin={node.get('skin')}")

# Check if the skin's joints refer to valid nodes
for i, skin in enumerate(d.get('skins', [])):
    print(f"  Skin[{i}]: joints={skin.get('joints', [])[:5]}..., skeleton={skin.get('skeleton')}")
