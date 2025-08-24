import os.path as osp
import argparse

import numpy as np
import torch

from smpl import Smplx, Smpl
from utils import visualize,visualize_trimesh
from pytorch3d.ops.knn import  knn_points

smplx_marker_id = {
    'head':8994,
    'chest':5533,
    'left_arm':3952,
    'left_forearm':4580,
    'left_hand':4612,
    'left_leg':3610,
    'left_shin':3746,
    'left_foot':5894,
    'right_arm':8133,
    'right_forearm':7316,
    'right_hand':7348,
    'right_leg':6371,
    'right_shin':6504,
    'right_foot':8588
}

smpl_marker_id = {
    'head':335,
    'chest':3073,
    'left_arm':2821,
    'left_forearm':1591,
    'left_hand':2000,
    'left_leg':981,
    'left_shin':1115,
    'left_foot':3341,
    'right_arm':4794,
    'right_forearm':5059,
    'right_hand':5459,
    'right_leg':4465,
    'right_shin':4599,
    'right_foot':6742
}


def smplx2smpl():
    all_vid = [value for value in smplx_marker_id.values()]
    smplx_model = Smplx(model_path='/home/lanhai/Projects/HPE/support_data/smplx/neutral/model.npz')
    smpl_model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz')

    smplx_faces = smplx_model.faces
    smpl_faces = smpl_model.faces

    smpl_betas = torch.zeros([1, smpl_model.num_betas], dtype=torch.float32)
    smplx_betas = torch.zeros([1, smplx_model.num_betas], dtype=torch.float32)
    expression = torch.zeros(
        [1, smplx_model.num_expression_coeffs], dtype=torch.float32)
    smpl_output = smpl_model(betas=smpl_betas,
                   body_pose=None,
                   global_orient=torch.tensor([[np.pi / 2, 0, 0]]),
                   transl=None,
                   return_verts=True)

    smplx_output = smplx_model(betas=smplx_betas,
                   body_pose=None,
                   global_orient=torch.tensor([[np.pi / 2, 0, 0]]),
                   transl=None,
                   expression = expression,
                   return_verts=True)

    smpl_joints = smpl_output['joints'].squeeze()
    smplx_joints = smplx_output['joints'].squeeze()
    smpl_vertices = smpl_output['vertices'].squeeze()
    smplx_vertices = smplx_output['vertices'].squeeze()

    # offset = torch.mean(smplx_vertices, dim = 0) - torch.mean(smpl_vertices,dim=0)
    offset = smplx_joints[0] - smpl_joints[0]

    smpl_vertices = smpl_vertices + offset[None,:]
    markers = smplx_vertices[all_vid]
    dist, idx, _ = knn_points(markers[None,...], smpl_vertices[None,...], K=1, return_nn=False)
    corr_id = [item for item in zip(all_vid, idx.squeeze().tolist())]

    vertices = torch.concatenate([smpl_vertices, smplx_vertices])
    faces = np.concatenate([smpl_faces, smplx_faces+6890])

    # vertices = smpl_vertices
    # faces = smpl_model.faces
    # markers = vertices[idx.squeeze().tolist()]

    visualize(vertices.detach().cpu().numpy(), faces, [markers.detach().cpu().numpy()])

    print(corr_id)

def plot_smpl():
    model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz')
    all_vid = [value for value in smpl_marker_id.values()]

    print(model)
    betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    # bodypose = 0.5*torch.pi*torch.randn([1, 23*3], dtype=torch.float32)
    bodypose = None
    transl = torch.randn([1, 3], dtype=torch.float32)

    output = model(betas=betas,
                   body_pose=bodypose,
                   global_orient=torch.tensor([[0, 0, 0]]),
                   transl=None,
                   return_verts=True)
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    joints = output['joints'].detach().cpu().numpy().squeeze()
    faces = model.faces
    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    # smpl_joints = [joints[0:24],  # 身体关节，红色 [0:22]
    #                joints[24:]]  # 脸部关节，蓝色 [22:25]

    smpl_joints = [joints[0:24],  # 身体关节，红色 [0:22]
                   joints[24:],
                   vertices[all_vid]]

    visualize(vertices, faces, smpl_joints)


def plot_smplx():
    all_vid = [value for value in smplx_marker_id.values()]
    model = Smplx(model_path='/home/lanhai/Projects/HPE/support_data/smplx/neutral/model.npz')
    print(model)
    betas, expression = None, None
    betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)
    # bodypose = 0.5*torch.pi*torch.randn([1, 23*3], dtype=torch.float32)
    bodypose = None
    transl = torch.randn([1, 3], dtype=torch.float32)

    output = model(betas=betas,
                   body_pose=bodypose,
                   global_orient=torch.tensor([[np.pi / 2, 0, 0]]),
                   transl=None,
                   expression=expression,
                   return_verts=True)
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    joints = output['joints'].detach().cpu().numpy().squeeze()
    faces = model.faces
    markers = vertices[all_vid]
    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    # smpl_joints = [joints[0:22],  # 身体关节，红色 [0:22]
    #                joints[22:25],  # 脸部关节，蓝色 [22:25]
    #                joints[25:55],  # 手部关节，绿色 [25:55]
    #                joints[55:]]  # 额外的关键点，黄色 [55:127]
    smpl_joints = [joints[0:22],  # 身体关节，红色 [0:22]
                   markers]  # 额外的关键点，黄色 [55:127]

    visualize(vertices, faces, smpl_joints)



def main():
    plot_smpl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')


    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=False,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    main()
