import json
import numpy as np
import os.path as osp

import numpy.random
import torch
from smpl import Smpl
import pickle
from tqdm import tqdm
from pathlib import Path
from glob import glob
meta_classes = [
    'arm movements',
    'backwards movement',
    'bend',
    'circular movement',
    'clean something',
    'dance',
    'foot movements',
    'forward movement',
    'gesture',
    'grasp object',
    'hand movements',
    'head movements',
    'interact with/use object',
    'jump',
    'knee movement',
    'leg movements',
    'look',
    'move something',
    'move up/down incline',
    'raising body part',
    'run',
    'sit',
    'stand',
    'step',
    'stretch',
    'swing body part',
    'touch object',
    'turn',
    'walk',
    'wave'
]

def get_valid_path(root, path):
    if osp.exists(osp.join(root, path)):
        return osp.join(root, path)
    else:
        p = Path(path)
        parts = p.parts
        new_path = p.relative_to(parts[0])
        if osp.exists(Path(root)/new_path):
            return Path(root)/new_path
        else:
            return None

def subsample_to_30fps(orig_ft, orig_fps):
        '''Get features at 30fps frame-rate
        Args:
            orig_ft <array> (T, 25*3): Feats. @ `orig_fps` frame-rate
            orig_fps <float>: Frame-rate in original (ft) seq.
        Return:
            ft <array> (T', 25*3): Feats. @ 30fps
        '''
        T  = orig_ft['poses'].shape[0]
        out_fps = 30.0
        # Matching the sub-sampling used for rendering
        if int(orig_fps)%int(out_fps):
            sel_fr = np.floor(orig_fps / out_fps * np.arange(int(out_fps))).astype(int)
            n_duration = int(T/int(orig_fps))
            t_idxs = []
            for i in range(n_duration):
                t_idxs += list(i * int(orig_fps) + sel_fr)
            if int(T % int(orig_fps)):
                last_sec_frame_idx = n_duration*int(orig_fps)
                t_idxs += [x+ last_sec_frame_idx for x in sel_fr if x + last_sec_frame_idx < T ]
        else:
            t_idxs = np.arange(0, T, orig_fps/out_fps, dtype=int)

        ft = {}
        for key, values in orig_ft.items():
            if key in ['file_path', 'betas']:
                continue
            ft[key] = values[t_idxs]
        return ft

def extract_babel_dataset():
    amass_file_path = '/home/lanhai/restore/dataset/mocap/AMASS'
    babel_file_path = '/home/lanhai/restore/dataset/mocap/babel_v1.0_release'
    train_file_path = osp.join(babel_file_path, 'train.json')
    val_file_path = osp.join(babel_file_path, 'val.json')
    test_file_path = osp.join(babel_file_path, 'test.json')
    model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device='cpu')

    with open(train_file_path, 'r') as f:
        train_data = json.load(f)

    with open(val_file_path, 'r') as f:
        val_data = json.load(f)

    with open(test_file_path, 'r') as f:
        test_data = json.load(f)

    with open('/home/lanhai/PycharmProjects/mosr/action_recognition/data/featp_2_fps.json', 'r') as f:
        fps_dict = json.load(f)
    with open('/home/lanhai/PycharmProjects/mosr/action_recognition/data/action_label_2_idx.json', 'rb') as f:
        act2idx_150 = json.load(f)
    act2idx = {k: act2idx_150[k] for k in act2idx_150 if act2idx_150[k] < 60}

    res = {}
    n = 0
    data_list = list(train_data.values())
    for i, item in enumerate(data_list):

        fp = get_valid_path(amass_file_path, item['feat_p'])
        fps = fps_dict[item['feat_p']]
        if fp is None:
            print(item['feat_p'])
        dur = item['dur']

        if item['frame_ann'] is not None:
            n += 1

            smpl_data = np.load(fp)
            labels = item['frame_ann']['labels']
            num_frames = smpl_data['poses'].shape[0]
            print(f'Extract {len(labels)} motion spans for {n}/{len(data_list)} in {i}th iter.')
            for label in labels:

                st_frame = int(num_frames * (label['start_t'] / dur))
                end_frame = int(num_frames * (label['end_t'] / dur))
                if (end_frame - st_frame) < 3:
                    continue
                elif (end_frame - st_frame) > 3000:
                    print(f'Very large span with {end_frame - st_frame} frames.')

                betas = torch.from_numpy(smpl_data['betas'][0:10]).float()
                # gender = torch.from_numpy(data['genders'][0])
                pose_body = smpl_data['poses'][:, 0:66]
                pose_hands = np.zeros([pose_body.shape[0], 6])
                poses = np.concatenate([pose_body, pose_hands], axis=-1)
                poses = torch.from_numpy(poses).float()[st_frame:end_frame]
                trans = torch.from_numpy(smpl_data['trans']).float()[st_frame:end_frame]


                trial = {'file_path': item['feat_p'],
                         'betas': betas.cpu(),
                         'poses': poses.cpu(),
                         'trans': trans.cpu()
                         }
                subsample_to_30fps(trial, fps)
                for act_cat in label['act_cat']:
                    if act_cat not in act2idx:
                        continue
                    if act_cat not in res:
                        res[act_cat] = []
                    res[act_cat].append(trial)

        if n % 500 == 0 or i == (len(data_list) - 1):
            print(f'Extract {n} trials in {i}th iteration.')
            output_file = osp.join(amass_file_path, f'amass_babel_train{n}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(res, f)
            del res
            res = {}

    return

def babel_sample_300frames():
    fp = glob('/home/lanhai/restore/dataset/mocap/AMASS/*.pkl')

    train_fp = [item for item in fp if 'train' in item]
    val_fp = [item for item in fp if 'val' in item]
    all_tasks = {}
    for key in meta_classes:
        all_tasks[key] = []

    for fp in train_fp:
        print(fp)
        with open(fp, 'rb') as f:
            data = pickle.load(f)
        for key in data.keys():
            if key not in meta_classes:
                continue
            task  =  data[key]
            for item in task:
                if item['poses'].shape[0]>300:
                    st_id = int(item['poses'].shape[0]/2)-150
                    end_id = st_id + 300
                    ft = {}
                    for k, v in item.items():
                        if k in ['file_path', 'betas']:
                            ft[k] = v
                        else:
                            ft[k] = v[st_id:end_id]
                    all_tasks[key].append(ft)
    with open('./meta_train_data_30classes.pkl', 'wb') as f:
        pickle.dump(all_tasks, f)
    print('Train data saved')

    del all_tasks
    all_tasks = {}
    for key in meta_classes:
        all_tasks[key] = []

    for fp in val_fp:
        print(fp)
        with open(fp, 'rb') as f:
            data = pickle.load(f)
        for key in data.keys():
            if key not in meta_classes:
                continue
            task = data[key]
            for item in task:
                if item['poses'].shape[0] > 300:
                    st_id = int(item['poses'].shape[0] / 2) - 150
                    end_id = st_id + 300
                    ft = {}
                    for k, v in item.items():
                        if k in ['file_path', 'betas']:
                            ft[k] = v
                        else:
                            ft[k] = v[st_id:end_id]
                    all_tasks[key].append(ft)
    with open('./meta_val_data_30classes.pkl', 'wb') as f:
        pickle.dump(all_tasks, f)
    print('Val data saved')

def babel_sample_30spans():
    # train data
    with open('/home/lanhai/PycharmProjects/mosr/tools/meta_train_data_30classes.pkl', 'rb') as f:
        train_data = pickle.load(f)

    save_data = {}
    for key, value in train_data.items():

        ind_list = torch.randperm(len(value))[:30]
        betas_list = []
        poses_list = []
        trans_list = []
        joints_list = []

        for ind in ind_list:
            betas_list.append(value[ind]['betas'])
            poses_list.append(value[ind]['poses'])
            trans_list.append(value[ind]['trans'])
            joints_list.append(value[ind]['joints'])

        save_data[key] = {'betas': torch.stack(betas_list),
                          'poses':torch.stack(poses_list),
                          'trans':torch.stack(trans_list),
                          'joints':torch.stack(joints_list),
                          'ind': ind_list
                          }
        print(f'{key} class data extracted')

    with open('./meta_train_data.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    print('Data saved')

    del train_data
    # val data
    with open('/home/lanhai/PycharmProjects/mosr/tools/meta_val_data_30classes.pkl', 'rb') as f:
        train_data = pickle.load(f)

    save_data = {}
    for key, value in train_data.items():
        ind_list = torch.randperm(len(value))[:30]
        betas_list = []
        poses_list = []
        trans_list = []
        joints_list = []

        for ind in ind_list:
            betas_list.append(value[ind]['betas'])
            poses_list.append(value[ind]['poses'])
            trans_list.append(value[ind]['trans'])
            joints_list.append(value[ind]['joints'])

        save_data[key] = {'betas': torch.stack(betas_list),
                          'poses':torch.stack(poses_list),
                          'trans':torch.stack(trans_list),
                          'joints':torch.stack(joints_list),
                          'ind': ind_list
                          }
        print(f'{key} class data extracted')


    with open('./meta_val_data.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    print('Data saved')




if __name__ == '__main__':
    # extract_babel_dataset()
    # babel_sample_300frames()
    babel_sample_30spans()
