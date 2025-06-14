import os.path as osp
import ezc3d
from glob import glob
from torch.utils.data import Dataset

class MocapSequenceDataset(Dataset):
    def __init__(self, sequence_dir):
        self.sequence_paths =  glob(osp.join(sequence_dir, '*/*.c3d'))  # or .pkl/.npz
        self.sequence_labels = ['_'.join(a.split('/')[-3:-1]) for a in self.sequence_paths]
        self.subject_names = [a.split('/')[-2] for a in self.sequence_paths]   
    
    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx, stagei= True):
        sequence_path = self.sequence_paths[idx]
        marker_data = ezc3d.c3d(sequence_path)
        markers = marker_data['data']['points'][:3].transpose(2, 1, 0)
        frame_rate = marker_data['parameters']['POINT']['RATE']['value'][0]
        labels = marker_data['parameters']['POINT']['LABELS']['value']

        return {
            "sequence_labels": self.sequence_labels[idx],
            'subject_name': self.subject_names[idx],
            'markers': markers, 
            'labels': labels, 
            'frame_rate': frame_rate, 
            'marker_data': marker_data
        }