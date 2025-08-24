import torch
import pickle
import numpy as np
from smplview import Smpl
from utils import vis_diff, visualize, visualize_aitviewer, vis_diff_aitviewer
from mocap_fitter import moshpp_marker_id
from data import BabelDataset



def plot_diff():
    smpl_model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz')
    vid = [value for value in moshpp_marker_id.values()]
    marker_pose_gt = smpl_model.v_template[vid] + 0.0095 * smpl_model.vertex_normals[vid]

    with open('/home/lanhai/PycharmProjects/mosr/results/20250809-2245/results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    for item in data:
        output = item['output']
        gt = item['gt']
        metrics = item['metrics']
        eucl_dists = metrics['MPJPE raw']
        eucl_dists_max_id = torch.argmax(torch.sum(eucl_dists, dim=-1), dim=-1)

        for i,max_ind in enumerate(eucl_dists_max_id.detach().tolist()):
            print(f'The biggest MPJPE frame is {eucl_dists_max_id[i]}')
            # visualize_aitviewer('smpl', full_poses=gt['poses'][i], betas=gt['betas'][i], trans=gt['trans'][i])
            vis_diff_aitviewer('smpl', gt_full_poses=gt['poses'][i], gt_betas=gt['betas'][i], gt_trans=gt['trans'][i],
                               pred_full_poses=output['poses'][i], pred_betas=output['betas'][i], pred_trans=output['trans'][i])

            print('Visualize the markers offset.')
            marker_pos_pred = output['marker_pos'][i].squeeze()


            visualize(smpl_model.v_template, smpl_model.faces, [marker_pose_gt, marker_pos_pred.detach().cpu().numpy()])
            #
            # vis_diff(output_vertices[max_ind].detach().cpu().numpy(), gt_vertices[max_ind].detach().cpu().numpy(), smpl_model.faces)
    
def plot_data():
    dataset = BabelDataset('/home/lanhai/restore/dataset/mocap/mosr/meta_train_data.pkl')
    tasks = dataset.task_id
    num_tasks = len(tasks)

    tasks = dataset.task_id
    t = 27
    print(f'Display the task {t}: {tasks[int(t)]}')
    data = dataset[t]
    i =25
    visualize_aitviewer('smpl',full_poses=data['poses'][i],betas=data['betas'][i],trans=data['trans'][i])
    print('Done')



if __name__ == '__main__':
    plot_diff()
    # plot_data()