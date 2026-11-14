import json
import base64
import struct
import numpy as np

with open('outputs/online_walking_run/reconstruction_skinned_walking.gltf') as f:
    gltf = json.load(f)

print("Nodes count:", len(gltf['nodes']))
for i, n in enumerate(gltf['nodes']):
    print(f"Node {i}: {n.get('name')}, children: {n.get('children')}")

# Decode buffer
uri = gltf['buffers'][0]['uri']
header, b64data = uri.split(',', 1)
buf = base64.b64decode(b64data)

channels = gltf['animations'][0]['channels']
samplers = gltf['animations'][0]['samplers']
accessors = gltf['accessors']
buffer_views = gltf['bufferViews']

print(f"\nAnimation channels count: {len(channels)}")

# Let's inspect some rotation values for key joints: Hips, LeftUpLeg, RightUpLeg, LeftArm, RightArm
for ch in channels:
    node_idx = ch['target']['node']
    node_name = gltf['nodes'][node_idx]['name']
    path = ch['target']['path']
    samp = samplers[ch['sampler']]
    acc = accessors[samp['output']]
    bv = buffer_views[acc['bufferView']]
    offset = bv['byteOffset'] + acc.get('byteOffset', 0)
    count = acc['count']
    if path == 'rotation':
        # VEC4 float32
        data = np.frombuffer(buf[offset:offset + count*16], dtype=np.float32).reshape((count, 4))
        # Print frame 0, 40, 60, 80
        print(f"\n{node_name} {path} (shape {data.shape}):")
        for f in [0, 40, 60, 80]:
            if f < count:
                print(f"  Frame {f:3d}: x={data[f,0]:.3f}, y={data[f,1]:.3f}, z={data[f,2]:.3f}, w={data[f,3]:.3f}")
