import numpy as np
import torch
import trimesh
import pyrender
import time
import pickle
from smpl import Smpl
import os
import numpy as np
from data import AmassDataset

def create_ground(size=2.0, color=[0.6, 0.6, 0.6, 1.0]):
    """
    创建一个大小为 size 的地面平面，位于 y=0（即 z=0 in SMPL）
    """
    # 创建一个 2x2 的方格平面，在 xz 平面上
    plane = trimesh.creation.box(extents=(size, size, -0.1))
    plane.apply_translation([0,  0, -1.1])  # 把它的中心下沉，使表面在 y=0

    # 创建材质和渲染用 mesh
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=1.0,
        baseColorFactor=color
    )
    mesh = pyrender.Mesh.from_trimesh(plane, material=material)
    return mesh


def load_smpl(file_name):
    f_type = os.path.splitext(file_name)[-1]
    if f_type=='.pkl':
        f = open(file_name, 'rb')
        data = pickle.load(f, encoding='latin1')
    elif f_type=='.npz':
        data = np.load(file_name, allow_pickle=True)
    else:
        print("The file format should be pkl or npz")
        return
    return data


# verts_seq: [T, 6890, 3]
# faces: [13776, 3]


R = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
])
data = load_smpl('/home/lanhai/restore/dataset/mocap/3dpw/sequenceFiles/sequenceFiles/train/courtyard_box_00.pkl')

# dataset = AmassDataset('/home/lanhai/restore/dataset/mocap/amass')
# data = load_smpl('/home/lanhai/restore/dataset/mocap/amass/BMLrub/BioMotionLab_NTroje/rub103/0000_treadmill_norm_poses.npz')
# data = dataset[0]

betas = torch.from_numpy(data['betas'][0]).float()
# gender = torch.from_numpy(data['genders'][0])
poses = torch.from_numpy(data['poses'][0]).float()
trans = torch.from_numpy(data['trans'][0]).float()

model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device='cpu')
faces = model.faces
T = poses.shape[0]
verts_seq = []
for i in range(T):
    verts_seq.append(model(betas=betas[None,],
                          body_pose=poses[i, 3:][None,],
                          global_orient=poses[i, 0:3][None,],
                          transl=trans[i][None,])['vertices'].squeeze())
# 初始化窗口
scene = pyrender.Scene()
mesh = trimesh.Trimesh(vertices=verts_seq[0].detach().numpy()@R.T, faces=faces, process=False)
mesh_node = pyrender.Mesh.from_trimesh(mesh)
node = scene.add(mesh_node)

viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
scene.add(create_ground(size=5.0))
# 动画循环
for t in range(1, T):
    # 更新顶点
    mesh = trimesh.Trimesh(vertices=verts_seq[t].detach().numpy()@R.T, faces=faces, process=False)
    new_mesh = pyrender.Mesh.from_trimesh(mesh)

    # ⭐ 使用线程锁，避免渲染线程与我们冲突
    with viewer.render_lock:
        if node in scene.nodes:
            scene.remove_node(node)
        node = scene.add(new_mesh)
    print(t)
    time.sleep(1 / 30)  # 控制帧率，30FPS
