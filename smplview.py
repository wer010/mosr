import os.path as osp
import argparse

import numpy as np
import torch

from smpl import Smplx, Smpl

def main(model_type='smpl',
         ext='npz',
         gender='neutral',
         plot_joints=True,
         num_betas=10,
         sample_shape=True,
         sample_expression=False,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):


    # model = Smplx(model_path='/home/lanhai/restore/dataset/mocap/models/smplx/SMPLX_NEUTRAL.npz')
    # model = Smplx(model_path='/home/lanhai/Projects/HPE/support_data/smplx/neutral/model.npz')
    model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz')


    print(model)
    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)
    bodypose = 0.5*torch.pi*torch.randn([1, 23*3], dtype=torch.float32)
    transl = torch.randn([1,  3], dtype=torch.float32)




    output = model( betas=betas,
                    body_pose=bodypose,
                    global_orient = torch.tensor([[np.pi/2,0,0]]),
                    transl = None,
                    expression=expression,
                    return_verts=True)
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    joints = output['joints'].detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    if plotting_module == 'pyrender':
        import pyrender
        import trimesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            # 定义不同颜色区间和对应的颜色 (RGBA格式)
            color_ranges = [
                (0, 22, [0.9, 0.1, 0.1, 1.0]),  # 红色 [0:22]
                (22, 25, [0.1, 0.1, 0.9, 1.0]),  # 蓝色 [22:25]
                (25, 55, [0.1, 0.9, 0.1, 1.0]),  # 绿色 [25:55]
                (55, 127, [0.9, 0.9, 0.1, 1.0])  # 黄色 [55:127]
            ]

            # 为每个颜色区间创建球体并添加到场景
            for start_idx, end_idx, color in color_ranges:
                if start_idx >= len(joints):  # 避免索引越界
                    continue

                # 截取当前区间的关节
                current_joints = joints[start_idx:end_idx]
                if len(current_joints) == 0:
                    continue

                # 创建对应颜色的球体
                sm = trimesh.creation.uv_sphere(radius=0.005)
                sm.visual.vertex_colors = color

                # 设置球体的位置变换矩阵
                tfs = np.tile(np.eye(4), (len(current_joints), 1, 1))
                tfs[:, :3, 3] = current_joints

                # 创建Mesh并添加到场景
                joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)
    elif plotting_module == 'matplotlib':
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
        plt.show()
    elif plotting_module == 'open3d':
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices)
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry = [mesh]
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            joints_pcl.points = o3d.utility.Vector3dVector(joints)
            joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
            geometry.append(joints_pcl)

        o3d.visualization.draw_geometries(geometry)
    else:
        raise ValueError('Unknown plotting_module: {}'.format(plotting_module))


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

    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression

    main(model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour)
