from copy import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size):
        """
        :param data_dir: Data directory
        :param batch_size: Batch size used in __next__()

        Note:
            Modify __init__() to fit your data
        """
        super(DataLoader, self).__init__(batch_size)
        self.data_dir = data_dir
        self.data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=256, shuffle=False)
        self.x = []
        self.y = []
        for data, target in self.data_loader:
            self.x += [i for i in data.numpy()]
            self.y += [i for i in target.numpy()]
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.n_batch = len(self.x) // self.batch_size
        self.batch_idx = 0

    def __iter__(self):
        self.n_batch = len(self.x) // self.batch_size
        self.batch_idx = 0
        assert self.n_batch > 0
        return self

    def __next__(self):
        if self.batch_idx < self.n_batch:
            x_batch = self.x[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            y_batch = self.y[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            self.batch_idx = self.batch_idx + 1
            return x_batch, y_batch
        else:
            raise StopIteration

    def __len__(self):
        """
        :return: Total batch number
        """
        self.n_batch = len(self.x) // self.batch_size
        return self.n_batch

