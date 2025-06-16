import os.path as osp
import ezc3d
from glob import glob
from torch.utils.data import Dataset, DataLoader
from marker_vids import all_marker_vids, general_labels_map
import re

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
        frame_rate = marker_data['parameters']['POINT']['RATE']['value'][0]
        labels = marker_data['parameters']['POINT']['LABELS']['value']
        labels = [item.split(':')[-1] for item in labels]
        general_labels = [key for key in general_labels_map]
        mapped_labels = [general_labels_map[key] if key in general_labels else key for key in labels]
        smplid = [self.smpl_id_tab[key] for key in mapped_labels]

        return {
            "sequence_labels": self.sequence_labels[idx],
            'subject_name': self.subject_names[idx],
            'markers': markers, 
            'labels': labels,
            'id': smplid,
            'frame_rate': frame_rate, 
            'marker_data': marker_data
        }

class MocapDataLoader(DataLoader):
    def __init__(self, dataset, stage='i', sample_size=12, **kwargs):
        self.stage = stage
        self.sample_size = sample_size
        self.dataset = dataset
        super().__init__(dataset, **kwargs)

    def __iter__(self):
        if self.stage == 'i':
            # Stage I: 从所有帧中采样
            indices = self.dataset.sample_indices(self.sample_size)
        elif self.stage == 'ii':
            # Stage II: 每次以一个 c3d 文件为单位
            pass
        else:
            raise ValueError(f"Unknown stage {self.stage}")

        # 用 PyTorch 的 Sampler 和 batch 机制处理这些 index
        sampler = torch.utils.data.SequentialSampler(indices)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)

        # 复用 PyTorch 的 DataLoader 工作线程、collate_fn 等机制
        return super().__iter__().__class__(self.dataset, batch_sampler=batch_sampler,
                                            collate_fn=self.collate_fn, num_workers=self.num_workers,
                                            pin_memory=self.pin_memory)
