import torch
import numpy as np
from torchvision import datasets, transforms
from base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size):
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
        self.n_batch = len(self.data_loader) * 256 / batch_size
        self.batch_idx = 0

    def next_batch(self):
        x_batch = self.x[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
        y_batch = self.y[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
        self.batch_idx = self.batch_idx + 1 if self.batch_idx != self.n_batch - 1 else 0
        return x_batch, y_batch

    def __len__(self):
        return len(self.data_loader)
