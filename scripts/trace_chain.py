import json
import base64
import numpy as np
from iris.rigging.orientation import QuaternionMath

with open('outputs/online_walking_run/reconstruction_skinned_walking.gltf') as f:
    gltf = json.load(f)

uri = gltf['buffers'][0]['uri']
header, b64data = uri.split(',', 1)
buf = base64.b64decode(b64data)

nodes = gltf['nodes']
channels = gltf['animations'][0]['channels']
samplers = gltf['animations'][0]['samplers']
accessors = gltf['accessors']
buffer_views = gltf['bufferViews']

T = accessors[samplers[0]['input']]['count']

translations = {}
rotations = {}

for ch in channels:
    node_idx = ch['target']['node']
    node_name = nodes[node_idx]['name']
    path = ch['target']['path']
    samp = samplers[ch['sampler']]
    acc = accessors[samp['output']]
    bv = buffer_views[acc['bufferView']]
    offset = bv['byteOffset'] + acc.get('byteOffset', 0)
    count = acc['count']
    if path == 'translation':
        data = np.frombuffer(buf[offset:offset + count*12], dtype=np.float32).reshape((count, 3))
        translations[node_name] = data
    elif path == 'rotation':
        data = np.frombuffer(buf[offset:offset + count*16], dtype=np.float32).reshape((count, 4))
        q_internal = np.zeros((count, 4), dtype=np.float32)
        q_internal[:, 0] = data[:, 3]
        q_internal[:, 1] = data[:, 0]
        q_internal[:, 2] = data[:, 1]
        q_internal[:, 3] = data[:, 2]
        rotations[node_name] = q_internal

node_dict = {n['name']: n for n in nodes}

t = 0
world_pos = {}
world_rot = {}

world_pos['Hips'] = translations.get('Hips', np.zeros((T, 3)))[t]
world_rot['Hips'] = rotations.get('Hips', np.tile([1,0,0,0], (T, 1)))[t]

print(f"Hips: pos={world_pos['Hips']}, rot={world_rot['Hips']}")

def step(p_name, c_name):
    p_pos = world_pos[p_name]
    p_rot = world_rot[p_name]
    c_local_t = np.array(node_dict[c_name]['translation'], dtype=np.float32)
    c_local_q = rotations.get(c_name, np.tile([1,0,0,0], (T, 1)))[t]
    
    rotated_offset = QuaternionMath.rotate_vector(p_rot, c_local_t)
    c_world_pos = p_pos + rotated_offset
    c_world_rot = QuaternionMath.multiply(p_rot, c_local_q)
    
    world_pos[c_name] = c_world_pos
    world_rot[c_name] = c_world_rot
    print(f"{c_name:<14}: local_t={c_local_t}, rot_offset={np.round(rotated_offset, 3)}, world_pos={np.round(c_world_pos, 3)}, local_q={np.round(c_local_q, 3)}")

step('Hips', 'Spine')
step('Spine', 'Chest')
step('Chest', 'Neck')
step('Neck', 'Head')
step('Chest', 'LeftShoulder')
step('LeftShoulder', 'LeftArm')
step('LeftArm', 'LeftForeArm')
step('LeftForeArm', 'LeftHand')
