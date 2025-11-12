import torch
import numpy as np
import logging
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

class ECGDataset(Dataset):

    def __init__(self, data, labels, dynamic_features, dynamic_feature_dim, stage1_labels=None, stage2_labels=None):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        if isinstance(dynamic_features, np.ndarray):
            self.dynamic_features = torch.from_numpy(dynamic_features).float()
        elif dynamic_features is None or len(dynamic_features) == 0:
            n_samples = data.shape[0] if isinstance(data, np.ndarray) else 0
            seq_len = data.shape[1] if isinstance(data, np.ndarray) and data.ndim > 1 else 0
            self.dynamic_features = torch.zeros(n_samples, seq_len, dynamic_feature_dim).float()
        else:
            raise TypeError(f'Unsupported type for dynamic_features: {type(dynamic_features)}')
        self.stage1_labels = torch.from_numpy(stage1_labels).long() if stage1_labels is not None else None
        self.stage2_labels = torch.from_numpy(stage2_labels).long() if stage2_labels is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item_dict = {'data': self.data[idx], 'dynamic_features': self.dynamic_features[idx], 'fine_grained_label': self.labels[idx]}
        if self.stage1_labels is not None:
            item_dict['stage1_label'] = self.stage1_labels[idx]
        if self.stage2_labels is not None:
            item_dict['stage2_label'] = self.stage2_labels[idx]
        return item_dict
