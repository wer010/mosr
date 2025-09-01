import os
import pickle
import torch
import numpy as np
import argparse
from glob import glob
import os.path as osp
from data import BabelDataset, virtual_marker, MetaCollate
from torch.utils.data import DataLoader, random_split
from models import Moshpp, SimpleRNN, ResNet
from metric import MetricsEngine
from smpl import Smpl
from tqdm import tqdm
from utils import visualize, vis_diff
from datetime import datetime
from utils import visualize_aitviewer, vis_diff_aitviewer
import torch.optim as optim
from tensorboardX import SummaryWriter
import higher

mosr_marker_id = {
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
moshpp_marker_id = {'ARIEL': 411, 'C7': 3470, 'CLAV': 3171, 'LANK': 3327, 'LBHD': 182, 'LBSH': 2940, 'LBWT': 3122,
                    'LELB': 1666, 'LELBIN': 1725, 'LFHD': 0, 'LFIN': 2174, 'LFRM': 1568, 'LFSH': 1317, 'LFWT': 857,
                    'LHEE': 3387, 'LIWR': 2112, 'LKNE': 1053, 'LKNI': 1058, 'LMT1': 3336, 'LMT5': 3346, 'LOWR': 2108,
                    'LSHN': 1082, 'LTHI': 1454, 'LTHMB': 2224, 'LTOE': 3233, 'LUPA': 1443, 'MBWT': 3022, 'MFWT': 3503,
                    'RANK': 6728, 'RBHD': 3694, 'RBSH': 6399, 'RBWT': 6544, 'RELB': 5135, 'RELBIN': 5194, 'RFHD': 3512,
                    'RFIN': 5635, 'RFRM': 5037, 'RFSH': 4798, 'RFWT': 4343, 'RHEE': 6786, 'RIWR': 5573, 'RKNE': 4538,
                    'RKNI': 4544, 'RMT1': 6736, 'RMT5': 6747, 'ROWR': 5568, 'RSHN': 4568, 'RTHI': 4927, 'RTHMB': 5686,
                    'RTOE': 6633, 'RUPA': 4918, 'STRN': 3506, 'T10': 3016}

vid = [value for value in moshpp_marker_id.values()]

def loss_fn(output, gt, do_fk = True):
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    pose_loss = mse_loss(output['poses'], gt['poses'])
    shape_loss = l1_loss(output['betas'], gt['betas'])
    tran_loss = mse_loss(output['trans'], gt['trans'])

    if do_fk:
        joints_hat = smpl_model(
            betas=output['betas'].expand(-1, L, -1).reshape(batch_size * L, -1),
            body_pose=output['poses'].reshape(batch_size * L, -1)[:, 3:],
            global_orient=output['poses'].reshape(batch_size * L, -1)[:, 0:3],
            transl=output['trans'].reshape(batch_size * L, -1)
        )['joints'].reshape(batch_size, L, -1, 3)
        fk_loss = mse_loss(joints_hat, support_joints[batch_ind])
    else:
        fk_loss = torch.zeros(1, device=device)
    total_loss = pose_loss + shape_loss + tran_loss + 0.1 * fk_loss

    losses = {'pose': pose_loss,
                 'shape': shape_loss,
                 'tran': tran_loss,
                 'fk': fk_loss,
                 'total_loss': total_loss}
    return losses

def train(model, tasks, writer, metrics_engine, batch_size = 5, device = 'cuda', do_fk = True):
    smpl_model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device=device)
    model.train()


    for t in tasks:
        print(f'Train on task {t}: {tasks[int(t)]}')
        train(model, dataset[t], writer, metrics_engine, device = 'cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    support_set = {key:value[:25] for key, value in tasks.items()}
    query_set = {key:value[25:] for key, value in tasks.items()}

    support_betas = support_set['betas']
    support_pose = support_set['poses']
    support_trans = support_set['trans']
    support_joints = support_set['joints']
    support_marker_pos = support_set['marker_pos']
    support_marker_ori = support_set['marker_ori']
    query_betas = query_set['betas']
    query_pose = query_set['poses']
    query_trans = query_set['trans']
    query_marker_pos = query_set['marker_pos']
    query_marker_ori = query_set['marker_ori']

    data_len = len(support_betas)
    global_step = 0
    print('Begin training.')
    for epoch in tqdm(range(400)):
        shuffle_ind = torch.randperm(data_len, device=device)  # 随机打乱索引

        for i in range(data_len//batch_size):
            optimizer.zero_grad()
            global_step+=1
            batch_ind = shuffle_ind[i * batch_size: (i + 1) * batch_size]
            B = len(batch_ind)
            L = support_pose.shape[1]
            x = support_marker_pos[batch_ind].contiguous().view(B, L, -1)

            output = model(x)

            losses = loss_fn(output, support_set)
            if writer is not None:
                mode_prefix = 'train'
                for k in losses:
                    prefix = '{}/{}'.format(k, mode_prefix)
                    writer.add_scalar(prefix, losses[k].cpu().item(), global_step)

            losses['total_loss'].backward()
            optimizer.step()


    # evaluate the query set (support set)
    output_list = []
    for i in range(len(support_betas)):
        input = support_marker_pos[i].contiguous().view(1, L, -1)

        output = model(input)
        output['joints'] = smpl_model(
            betas=output['betas'].reshape(-1),
            body_pose=output['poses'].reshape(L, -1)[:, 3:],
            global_orient=output['poses'].reshape(L, -1)[:, 0:3],
            transl=output['trans'].reshape(L, -1)
        )['joints'].reshape(L, -1, 3)
        # vis_diff_aitviewer('smpl', gt_full_poses=support_set['poses'][i], gt_betas=support_set['betas'][i], gt_trans=support_set['trans'][i],
        #                    pred_full_poses=output['poses'].squeeze(), pred_betas=output['betas'].squeeze(),
        #                    pred_trans=output['trans'].squeeze())
        output_list.append(output)
    output = {}
    for key in output_list[0].keys():
        output[key] = torch.stack([item[key] for item in output_list])

    metrics = metrics_engine.compute(output, support_set)
    print(metrics_engine.to_pretty_string(metrics, f"{model.model_name()}-SupportSet"))

    # evaluate the query set (test set)
    output_list = []
    for i in range(len(query_betas)):
        input = query_marker_pos[i].contiguous().view(1, L, -1)

        output = model(input)
        output['joints'] = smpl_model(
            betas=output['betas'].reshape(-1),
            body_pose=output['poses'].reshape(L, -1)[:, 3:],
            global_orient=output['poses'].reshape(L, -1)[:, 0:3],
            transl=output['trans'].reshape(L, -1)
        )['joints'].reshape(L, -1, 3)

        output_list.append(output)
    output = {}
    for key in output_list[0].keys():
        output[key] = torch.stack([item[key] for item in output_list])

    metrics = metrics_engine.compute(output, query_set)
    print(metrics_engine.to_pretty_string(metrics, f"{model.model_name()}-QuerySet"))

def eval(model, tasks, metrics_engine, device = 'cuda'):
    smpl_model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device=device)
    n_marker = len(vid)
    model.eval()

    support_set = {key:value[:25] for key, value in tasks.items()}
    query_set = {key:value[25:] for key, value in tasks.items()}

    support_betas = support_set['betas']
    support_pose = support_set['poses']
    support_trans = support_set['trans']
    support_joints = support_set['joints']
    support_marker_pos = support_set['marker_pos']
    support_marker_ori = support_set['marker_ori']
    query_betas = query_set['betas']
    query_pose = query_set['poses']
    query_trans = query_set['trans']
    query_marker_pos = query_set['marker_pos']
    query_marker_ori = query_set['marker_ori']

    # evaluate the query set (test set)
    output_list = []
    L = support_pose.shape[1]
    for i in range(len(query_betas)):
        input = query_marker_pos[i].contiguous().view(1, L, -1)
        # visualize_aitviewer('smpl', full_poses=query_pose[i], betas=query_betas[i], trans=query_trans[i], extra_points=[query_marker_pos.detach().cpu().numpy()])
        output = model(input)
        output['joints'] = smpl_model(
            betas=output['betas'].reshape(-1),
            body_pose=output['poses'].reshape(L, -1)[:, 3:],
            global_orient=output['poses'].reshape(L, -1)[:, 0:3],
            transl=output['trans'].reshape(L, -1)
        )['joints'].reshape(L, -1, 3)
        output_list.append(output)
    output = {}
    for key in output_list[0].keys():
        output[key] = torch.stack([item[key] for item in output_list])
    metrics = metrics_engine.compute(output, query_set)
    print(metrics_engine.to_pretty_string(metrics, f"{model.model_name()}-QuerySet"))
    return output, metrics, query_set

def metatrain(model, dataset, writer, device):
    collate_fn = MetaCollate()
    trainloader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
    model.train()
    meta_opt = optim.Adam(model.parameters(), lr=1e-3)

    for data in trainloader:
        # Sample a batch of support and query images and labels.
        supp_set, qry_set = data

        task_num, supp_num, seq_num, marker_num, _ = supp_set['marker_pos'].shape()
        qry_num = qry_set['marker_pos'].shape(1)

        input = supp_set['marker_pos'].contiguous()
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(model.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(
                    model, inner_opt, copy_initial_weights=True,track_higher_grads=True
            ) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(input[i])
                    spt_loss = loss_fn(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry[i])
                qry_loss = loss_fn(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(
                    dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()

        meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
        })

def metatest(net, db, writer, metrics_engine, device):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The query loss and acc induced by these parameters.
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i], reduction='none')
                qry_losses.append(qry_loss.detach())
                qry_accs.append(
                    (qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(
        f'[Epoch {epoch + 1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
    )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })

def main(config):
    # load data
    device = 'cuda'
    n_marker = len(vid)

    dataset = BabelDataset('/home/lanhai/restore/dataset/mocap/mosr/meta_train_data_with_marker.pkl', device = device)
    tasks = dataset.task_id
    num_tasks = len(dataset)

    train_tasks = torch.arange(0, num_tasks*0.9, dtype = int)
    test_tasks = torch.arange(num_tasks*0.9, num_tasks, dtype = int)
    # model = Moshpp(iter_stage1 = 1000, iter_stage2 = 2000)
    model = ResNet(input_size=3*n_marker, betas_size=10, poses_size=24*3, trans_size=3,num_layers=2, hidden_size=256).to(device)
    
    metrics_engine = MetricsEngine()


    # train
    save_dir = osp.join('./results', datetime.now().strftime('%Y%m%d-%H%M'))
    os.mkdir(save_dir)
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    train(model, dataset, writer, metrics_engine, device)

    metatrain(model, dataset, writer, device = device)


    torch.save(model.state_dict(), osp.join(save_dir,"model.pth"))
    results = []

    # evaluate  
    for t in test_tasks:
        print(f'Evaluate on task {t}: {tasks[int(t)]}')
        output, metrics, label = eval(model, dataset[t], metrics_engine)
        res = {
            'output':output,
            'metrics':metrics,
            'gt':label
        }
        results.append(res)
        # vis_results(output, metrics)

    with open(osp.join(save_dir, 'results.pkl'),'wb') as f:
        pickle.dump(results,f)
    return





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General.
    parser.add_argument('--data_path', default=None, help='Use this experiment ID or create a new one.')
    parser.add_argument('--experiment_id', default=None, help='Use this experiment ID or create a new one.')
    parser.add_argument('--seed', type=int, default=42, help='Random generator seed.')
    parser.add_argument('--data_workers', type=int, default=4, help='Number of parallel threads for data loading.')
    parser.add_argument('--print_every', type=int, default=25, help='Print stats to console every so many iters.')
    parser.add_argument('--eval_every', type=int, default=700, help='Evaluate validation set every so many iters.')
    parser.add_argument('--tag', default='', help='A custom tag for this experiment.')
    parser.add_argument('--test', action='store_true', help='Will tag this run as a test run.')
    parser.add_argument('--device', type = str,  default='cuda', help='Use cpu or cuda')


    # Model configurations.
    parser.add_argument('--m_type', default='rnn', choices=['rnn', 'resnet', 'ief', 'lgd'], help='The type of model.')
    parser.add_argument('--m_estimate_shape', action='store_true', help='The model estimates the body shape.')
    parser.add_argument('--m_shape_hidden_size', default=256,
                        help='Size of the network estimating the shape.')  # Only used in RNN/ResNet.
    parser.add_argument('--m_fk_loss', type=float, default=0.0, help='Add an FK loss, requires shape estimate.')
    parser.add_argument('--m_dropout', type=float, default=0.0, help='Dropout applied on inputs.')
    parser.add_argument('--m_hidden_size', type=int, default=1024, help='Number of hidden units.')
    parser.add_argument('--m_num_layers', type=int, default=2, help='Number of layers.')
    parser.add_argument('--m_learn_init_state', action='store_true', help='Learn initial hidden state.')
    parser.add_argument('--m_bidirectional', action='store_true', help='Bidirectional RNN.')

    # IEF model specific.
    parser.add_argument('--m_num_iterations', type=int, default=4, help='Number of iterations for IEF.')
    parser.add_argument('--m_dropout_hidden', type=float, default=0.0, help='Dropout applied inside layers.')
    parser.add_argument('--m_step_size', type=float, default=0.1, help='Step size for IEF update.')
    parser.add_argument('--m_reprojection_loss_weight', type=float, default=0.01, help='Reprojection loss weight.')
    parser.add_argument('--m_shape_loss_weight', type=float, default=1.0, help='Loss for the shape weight.')
    parser.add_argument('--m_pose_loss_weight', type=float, default=1.0, help='Loss for the shape weight.')
    parser.add_argument('--m_average_shape', action='store_true', help='Average the shape per sequence.')
    parser.add_argument('--m_use_gradient', action='store_true', help='Feed dL/dtheta to the network.')
    parser.add_argument('--m_skip_connections', action='store_true', help='Skip connections in the MLP.')
    parser.add_argument('--m_no_batch_norm', action='store_true', help="Don't use batch norm.")
    parser.add_argument('--m_rnn_init', action='store_true', help="Initial estimate is provided by an RNN.")
    parser.add_argument('--m_rnn_denoiser', action='store_true', help="Use an RNN to de-noise the markers.")
    parser.add_argument('--m_rnn_bidirectional', action='store_true', help="BiRNN or not.")
    parser.add_argument('--m_rnn_hidden_size', type=int, default=512, help="Hidden size for the init RNN.")
    parser.add_argument('--m_rnn_num_layers', type=int, default=2, help="Number of layers for the init RNN.")

    # Input data.
    parser.add_argument('--use_marker_pos', action='store_true', help='Feed marker positions.')
    parser.add_argument('--use_marker_ori', action='store_true', help='Feed marker orientations.')
    parser.add_argument('--use_marker_nor', action='store_true', help='Feed marker normal instead of orientation.')
    parser.add_argument('--use_real_offsets', action='store_true',
                        help='Sampling is informed by real offset distribution.')
    parser.add_argument('--offset_noise_level', type=int, default=0, help='How much noise to add to real offsets.')
    parser.add_argument('--n_markers', type=int, default=12, help='Subselect a number of markers for the input.')

    # Data augmentation.
    parser.add_argument('--noise_num_markers', type=int, default=1, help='How many markers are affected by the noise.')
    parser.add_argument('--spherical_noise_strength', type=float, default=0.0, help='Magnitude of noise in %.')
    parser.add_argument('--spherical_noise_length', type=float, default=0.0, help='Temporal length of noise in %.')
    parser.add_argument('--suppression_noise_length', type=float, default=0.0, help='Marker suppression length.')
    parser.add_argument('--suppression_noise_value', type=float, default=0.0, help='Marker suppression value.')

    # Learning configurations.
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--bs_train', type=int, default=16, help='Batch size for the training set.')
    parser.add_argument('--bs_eval', type=int, default=16, help='Batch size for valid/test set.')
    parser.add_argument('--eval_window_size', type=int, default=None, help='Window size for evaluation on test set.')
    parser.add_argument('--window_size', type=int, default=120, help='Number of frames to extract per sequence.')
    parser.add_argument('--load', action='store_true', help='Whether to load the model with the given ID.')
    config = parser.parse_args()
    main(config)