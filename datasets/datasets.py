import numpy as np
import torch
from torch.utils.data import Dataset


class My_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        raise NotImplementedError

    def __getitem__(self, idx):
        data, target = None, None
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        raise NotImplementedError
        return data, target