import json
import base64
import numpy as np
from iris.rigging.orientation import QuaternionMath

# Load glTF
with open('outputs/online_walking_run/reconstruction_skinned_walking.gltf') as f:
    gltf = json.load(f)

# Decode buffer
uri = gltf['buffers'][0]['uri']
header, b64data = uri.split(',', 1)
buf = base64.b64decode(b64data)

nodes = gltf['nodes']
channels = gltf['animations'][0]['channels']
samplers = gltf['animations'][0]['samplers']
accessors = gltf['accessors']
buffer_views = gltf['bufferViews']

# Extract animation tracks
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
        # glTF has [x, y, z, w], convert to [w, x, y, z] for QuaternionMath
        q_internal = np.zeros((count, 4), dtype=np.float32)
        q_internal[:, 0] = data[:, 3]
        q_internal[:, 1] = data[:, 0]
        q_internal[:, 2] = data[:, 1]
        q_internal[:, 3] = data[:, 2]
        rotations[node_name] = q_internal

print(f"Total animation frames: {T}")

# Let's compute world positions of joints via forward kinematics
# Hierarchy:
# Hips -> Spine -> Chest -> Neck -> Head
# Chest -> LeftShoulder -> LeftArm -> LeftForeArm -> LeftHand
# Chest -> RightShoulder -> RightArm -> RightForeArm -> RightHand
# Hips -> LeftUpLeg -> LeftLeg -> LeftFoot
# Hips -> RightUpLeg -> RightLeg -> RightFoot

node_dict = {n['name']: n for n in nodes}

def get_rest_local(n_name):
    n = node_dict[n_name]
    return np.array(n['translation'], dtype=np.float32)

def compute_fk_frame(t):
    world_pos = {}
    world_rot = {}
    
    # Hips root
    root_t = translations.get('Hips', np.zeros((T, 3)))[t]
    root_q = rotations.get('Hips', np.tile([1,0,0,0], (T, 1)))[t]
    world_pos['Hips'] = root_t
    world_rot['Hips'] = root_q
    
    def traverse(p_name, c_name):
        p_pos = world_pos[p_name]
        p_rot = world_rot[p_name]
        c_local_t = get_rest_local(c_name)
        c_local_q = rotations.get(c_name, np.tile([1,0,0,0], (T, 1)))[t]
        
        # In glTF, parent rotation rotates child's local translation
        rotated_offset = QuaternionMath.rotate_vector(p_rot, c_local_t)
        c_world_pos = p_pos + rotated_offset
        c_world_rot = QuaternionMath.multiply(p_rot, c_local_q)
        
        world_pos[c_name] = c_world_pos
        world_rot[c_name] = c_world_rot

    # Spine chain
    traverse('Hips', 'Spine')
    traverse('Spine', 'Chest')
    traverse('Chest', 'Neck')
    traverse('Neck', 'Head')
    
    # Left Arm
    traverse('Chest', 'LeftShoulder')
    traverse('LeftShoulder', 'LeftArm')
    traverse('LeftArm', 'LeftForeArm')
    traverse('LeftForeArm', 'LeftHand')

    # Right Arm
    traverse('Chest', 'RightShoulder')
    traverse('RightShoulder', 'RightArm')
    traverse('RightArm', 'RightForeArm')
    traverse('RightForeArm', 'RightHand')

    # Left Leg
    traverse('Hips', 'LeftUpLeg')
    traverse('LeftUpLeg', 'LeftLeg')
    traverse('LeftLeg', 'LeftFoot')

    # Right Leg
    traverse('Hips', 'RightUpLeg')
    traverse('RightUpLeg', 'RightLeg')
    traverse('RightLeg', 'RightFoot')
    
    return world_pos

for frame in [0, 20, 39, 45, 60, 75, 90, 100, 105, 110, 115, 119]:
    wp = compute_fk_frame(frame)
    print(f"\n================ FRAME {frame} ================")
    print(f"Hips:      {wp['Hips'].round(2)}")
    print(f"L_Elbow:   {wp['LeftForeArm'].round(2)}")
    print(f"L_Hand:    {wp['LeftHand'].round(2)}")
    print(f"R_Elbow:   {wp['RightForeArm'].round(2)}")
    print(f"R_Hand:    {wp['RightHand'].round(2)}")
    print(f"L_Knee:    {wp['LeftLeg'].round(2)}")
    print(f"L_Foot:    {wp['LeftFoot'].round(2)}")
    print(f"R_Knee:    {wp['RightLeg'].round(2)}")
    print(f"R_Foot:    {wp['RightFoot'].round(2)}")
