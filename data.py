import os.path as osp
# import ezc3d
from glob import glob
from torch.utils.data import Dataset, DataLoader
from marker_vids import all_marker_vids, general_labels_map
import re
import numpy as np
import torch
from torch.utils.data import Sampler
import random
import pandas as pd
from smpl import Smpl
import pickle
from tqdm import tqdm
from geo_utils import estimate_lcs_with_faces
from utils import visualize, visualize_aitviewer
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

class MocapSequenceDataset(Dataset):
    def __init__(self, sequence_dir):
        self.sequence_paths =  glob(osp.join(sequence_dir, '*/*.c3d'))  # or .pkl/.npz
        self.sequence_labels = ['_'.join(re.split(r'[/.]',a)[-4:-1]) for a in self.sequence_paths]
        self.subject_names = [a.split('/')[-2] for a in self.sequence_paths]
        self.smpl_id_tab  = all_marker_vids['smpl']
    
    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx, stagei= True):
        sequence_path = self.sequence_paths[idx]
        marker_data = ezc3d.c3d(sequence_path)
        markers = marker_data['data']['points'][:3].transpose(2, 1, 0)
        nan_mask = np.isnan(markers).any(axis = (1,2))
        markers = markers[~nan_mask]
        frame_rate = marker_data['parameters']['POINT']['RATE']['value'][0]
        labels = marker_data['parameters']['POINT']['LABELS']['value']
        labels = [item.split(':')[-1] for item in labels]
        general_labels = [key for key in general_labels_map]
        mapped_labels = [general_labels_map[key] if key in general_labels else key for key in labels]
        smplid = [self.smpl_id_tab[key] for key in mapped_labels]

        return {
            "sequence_labels": self.sequence_labels[idx],
            'subject_name': self.subject_names[idx],
            'markers_pos': markers, 
            'labels': labels,
            'id': smplid,
            'frame_rate': frame_rate, 
            'marker_data': marker_data
        }


class BabelDataset(Dataset):
    def __init__(self, path, device = 'cuda'):

        self.device = device
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.task_id = {}
        self.samples_per_task = []
        for i, key in enumerate(self.data.keys()):
            self.task_id[i] = key
            self.samples_per_task.append(self.data[key]['poses'].shape[0])

    def __len__(self):
        return sum(self.samples_per_task)

    def __getitem__(self, t):
        task_idx, sample_idx = t
        data = {key:value[sample_idx].to(self.device) for key, value in self.data[self.task_id[int(task_idx)]].items()}
        return data

class MocapTaskBatchSampler(Sampler):
    def __init__(self, task_dataset, task_batch_size=4, support_ratio=0.5, query_ratio=0.5, shuffle=True):
        """
        对babeldataset的采样，因为babeldataset中的getitem(ind)函数中，我把ind设置为一个tuple(task_id, sample_id),
        从而能够从某个任务中提取几个样本。
        那这个sampler的要实现的功能就是：
        1、支持meta learning的训练模式数据分发，返回数据为[num_tasks, num_seq, dim_features]。
        2、按比例随机划分supp/qry set。
        Args:
            task_dataset: list of tasks, each task is a list of sequence indices
            task_batch_size: number of tasks per outer loop
            support_size: number of sequences for inner loop (support set)
            query_size: number of sequences for outer loop (query set)
            shuffle: whether to shuffle tasks and sequences each iteration
        """
        self.task_dataset = task_dataset
        self.num_tasks = len(task_dataset)
        self.task_batch_size = task_batch_size
        self.support_ratio = support_ratio
        self.query_ratio = query_ratio
        self.shuffle = shuffle

    def __iter__(self):
        # Shuffle tasks order
        task_indices = list(range(self.num_tasks))
        if self.shuffle:
            random.shuffle(task_indices)

        batch_tasks = []
        for t_idx in task_indices:
            task_sequences = list(range(len(self.task_dataset[t_idx])))
            if self.shuffle:
                random.shuffle(task_sequences)

            # Randomly split support/query
            support_idx = task_sequences[:self.support_size]
            query_idx = task_sequences[self.support_size:self.support_size + self.query_size]

            batch_tasks.append((t_idx, support_idx, query_idx))

            # Yield meta-batch
            if len(batch_tasks) == self.task_batch_size:
                yield batch_tasks
                batch_tasks = []

        # Yield remaining tasks if any
        if len(batch_tasks) > 0:
            yield batch_tasks

    def __len__(self):
        # Number of meta-batches per epoch
        return (self.num_tasks + self.task_batch_size - 1) // self.task_batch_size

    @staticmethod
    def collate_fn(item_list):
        # item_list: batch_tasks
        batch = []
        for t_idx, support_idx, query_idx in item_list:
            # Retrieve sequences from original dataset
            sequences = self.task_dataset[t_idx]
            support_seqs = [sequences[i] for i in support_idx]
            query_seqs = [sequences[i] for i in query_idx]
            batch.append({'task_idx': t_idx,
                          'support': torch.stack(support_seqs, dim=0),
                          'query': torch.stack(query_seqs, dim=0)})
        return batch


class AmassDataset(Dataset):
    def __init__(self, sequence_dir):
        cmu_path = osp.join(sequence_dir,'CMU')
        bml_path = osp.join(sequence_dir, 'BMLrub')
        cmu_files = glob(osp.join(cmu_path, '*/*.npz'))
        bml_files = glob(osp.join(bml_path, '*/*/*.npz'))
        cmu_files = [s for s in cmu_files if "stageii.npz" in s]

        self.sequence_paths = bml_files + cmu_files

        self.sequence_labels = ['_'.join(re.split(r'[/.]', a)[-4:-1]) for a in self.sequence_paths]
        self.subject_names = [a.split('/')[-2] for a in self.sequence_paths]
        self.smpl_id_tab = all_marker_vids['smpl']

    def __len__(self):
        return len(self.sequence_labels)

        sequence_path = self.sequence_paths[idx]
        data = np.load(sequence_path, allow_pickle=True)
        betas = torch.from_numpy(data['betas'][0:10]).float()
        # gender = torch.from_numpy(data['genders'][0])
        poses = torch.from_numpy(data['poses'][:, 0:66]).float()
        trans = torch.from_numpy(data['trans']).float()

        return {
            "sequence_labels": self.sequence_labels[idx],
            'subject_name': self.subject_names[idx],
            'markers': markers,
            'labels': labels,
            'id': smplid,
            'frame_rate': frame_rate,
            'marker_data': marker_data
        }

class CMUDataset(Dataset):
    def __init__(self,
                 data_path,
                 num_frame = 120):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.num_frame = num_frame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class MyCollator:
    def __init__(self, num_frame):
        self.num_frame = num_frame

    def __call__(self, data_list):
        ret = {}
        keys = data_list[0].keys()

        for key in keys:
            if data_list[0][key].dim()==1:
                values = [item[key].expand([self.num_frame, -1]) for item in data_list]
            else:
                len_list = [item[key].shape[0] for item in data_list]
                si_list = [torch.randint(0, l-self.num_frame, (1,)) for l in len_list]
                values = [item[key][si:si+self.num_frame] for si, item in zip(si_list, data_list)]
            values = torch.concatenate(values)
            ret[key] = values
        return ret

def generate_marker_data(fp, vid):
    n_marker = len(vid)
    device = 'cuda'
    pos_offset = torch.tensor([0.0095, 0, 0, 1]).expand([n_marker, -1]).to(device)
    ori_offset = torch.eye(3).expand([n_marker, -1, -1]).to(device)
    vid_tensor = torch.tensor(vid).to(device)
    dataset = BabelDataset(fp, device=device)
    tasks = dataset.task_id
    save_data = {}
    for t in range(len(dataset)):
        print(f'{t}: Generate the marker for {dataset.task_id[int(t)]}')
        data = dataset[t]
        marker_pos_list = []
        marker_ori_list = []
        joints_list = []
        for i in range(len(data['betas'])):
            frame_num = data['poses'][i].shape[0]
            vid_tensor = vid_tensor.expand([frame_num, -1]).to(device)
            marker_pos, marker_ori, v_posed, joints = virtual_marker(data['betas'][i],
                                                                     data['poses'][i],
                                                                     data['trans'][i],
                                                                     vid_tensor,
                                                                     pos_offset,
                                                                     ori_offset,
                                                                     visualize_flag=False)
            marker_pos_list.append(marker_pos)
            marker_ori_list.append(marker_ori)
            joints_list.append(joints)

        marker_pos = torch.stack(marker_pos_list)
        marker_ori = torch.stack(marker_ori_list)
        joints = torch.stack(joints_list)

        ret = {'betas': data['betas'],
               'poses': data['poses'],
               'trans': data['trans'],
               'marker_pos': marker_pos,
               'marker_ori': marker_ori,
               'joints': joints}
        save_data[dataset.task_id[int(t)]] = ret

    with open(fp.replace('.pkl', '_with_marker.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

def convert_dataset():
    df = pd.read_excel('/home/lanhai/PycharmProjects/mosr/cmu_motion.xlsx', header = None)

    motion_cates = df.iloc[:,0].dropna().unique().tolist()
    sub_motion_cats = df.iloc[:,1].to_list()
    subject_ids = df.columns[2:]

    all_paths = []
    motions = {}
    for row_idx, row in df.iterrows():

        if not pd.isna(row[0]):
            sub_motions = {}
            motion = row[0]
        sub_motion_key = row[1]
        sub_motion = {}
        for sub in row[2:].dropna().to_list():
            match = re.search(r'Subject (\d+)', sub)
            sub_ind = match.group(1).zfill(2)
            trials_ind_list = re.findall(r'\n(\d+)', sub)
            sub_motion[sub_ind] = [s.zfill(2) for s in trials_ind_list]
        sub_motions[sub_motion_key] = sub_motion
        motions[motion] = sub_motions
    one_motions = {}
    all_motions = []
    for motion, value in motions.items():
        file_paths = []
        for sub_motion, sub_value in value.items():
            for subject, trials in sub_value.items():
                for t in trials:
                    file_path = osp.join('/home/lanhai/restore/dataset/mocap/amass/CMU-smplx-n/', subject, f'{subject}_{t}_stageii.npz')
                    file_paths.append(file_path)
                    trial = {
                        'file_path':file_path,
                        'trial_id': t,
                        'subject_id': subject,
                        'sub_motion':sub_motion,
                        'motion': motion
                    }
                    all_motions.append(trial)
        one_motions[motion] =  file_paths

    model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device='cpu')

    for key, value in one_motions.items():
        output_file = f'/home/lanhai/restore/dataset/mocap/amass/CMU_motion/{key}.pkl'
        if osp.exists(output_file):
            continue
        res = {}
        motions = []
        num_trials = len(value)
        valid_num = 0
        for file_path in value:
            if osp.exists(file_path):
                # convert to lmdb
                valid_num+=1

                data = np.load(file_path, allow_pickle=True)
                print(f'{file_path} is {valid_num}th available amass data in {num_trials} trials with {data["poses"].shape[0]} frames')

                betas = torch.from_numpy(data['betas'][0:10]).float()
                # gender = torch.from_numpy(data['genders'][0])
                pose_body = data['poses'][:, 0:66]
                pose_hands = np.zeros([pose_body.shape[0], 6])
                poses = np.concatenate([pose_body,pose_hands],axis=-1)
                poses = torch.from_numpy(poses).float()
                trans = torch.from_numpy(data['trans']).float()
                if data["poses"].shape[0]>3000:
                    n_slices = data["poses"].shape[0]//3000
                    joints_list = []
                    for i in range(n_slices+1):
                        joints_list.append(model(betas=betas,
                                       body_pose=poses[3000*i:3000*(i+1), 3:],
                                       global_orient=poses[3000*i:3000*(i+1), 0:3],
                                       transl=trans[3000*i:3000*(i+1),:])['joints'])
                    joints = torch.concatenate(joints_list)
                else:
                    joints = model(betas=betas,
                              body_pose=poses[:,3:],
                              global_orient=poses[:, 0:3],
                              transl=trans)['joints']
                trial = {'betas': betas.cpu(),
                         'poses': poses.cpu(),
                         'trans': trans.cpu(),
                         'joints': joints.cpu()
                         }
                motions.append(trial)
        print(f'{valid_num} trials is save in {key} motion class')

        with open(output_file,'wb') as f:
            pickle.dump(motions, f)
        del motions
        torch.cuda.empty_cache()
    print('All done')

def virtual_marker(betas, 
                   pose, 
                   trans, 
                   vid, 
                   pos_offset=None, 
                   ori_offset=None,
                   visualize_flag = False):
    # get the 6 dof info of markers under the given pose
    '''
    :param betas: shape (10)
    :param pose: shape (n,24*3)
    :param trans: shape (n,3)
    :param vid: shape (n, m) m is the num of markers
    :param pos_offset: (m, 4)
    :param ori_offset: (m, 3, 3)
    :return:
    '''
    model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device=betas.device)
    output = model(betas=betas,
                     body_pose=pose[:,3:],
                     global_orient=pose[:,0:3],
                     transl=trans)
    v_posed = output['vertices']
    joints = output['joints']

    lcs = estimate_lcs_with_faces(vid=vid,
                                  fid=model.vertex_faces[vid],
                                  vertices=v_posed,
                                  faces=model.faces_tensor)

    marker_pos = torch.matmul(lcs, pos_offset[None, ..., None])[:, :, 0:3, 0]
    marker_ori = torch.matmul(lcs[:, :, 0:3, 0:3], ori_offset)
    if visualize_flag:
        # visualize a random frame
        i = torch.randint(0, v_posed.shape[0], [])
        visualize(v_posed[i].cpu().detach().numpy(), model.faces,
                [joints[i].cpu().detach().numpy()], lcs[i].cpu().detach().numpy())
        visualize_aitviewer('smpl',
                            full_poses=pose,
                            betas=betas,
                            trans=trans,
                            extra_points=[marker_pos.detach().cpu().numpy()])

    return marker_pos, marker_ori, v_posed, joints


if __name__ == '__main__':
    pass
    # convert_dataset()
    generate_marker_data('/home/lanhai/restore/dataset/mocap/mosr/meta_val_data.pkl', vid)