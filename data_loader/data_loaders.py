import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, valid_batch_size=1000, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=4):
        """
        :param data_dir: Data directory
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        trsfm = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST('../data', train=True, download=True, transform=trsfm)
        
        super(MnistDataLoader, self).__init__(self.dataset, self.batch_size, self.valid_batch_size, shuffle, validation_split, validation_fold, num_workers)

