import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader


def get_data_loader(config):
    """Returns data loader as specified in configuration."""
    loader_type = config['data_loader']['type']

    if loader_type == 'MnistDataLoader':
        return MnistDataLoader(config)
    else:
        raise NotImplementedError(f"Loader {loader_type} not implemented.")


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, config):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = config['data_loader']['data_dir']
        self.dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, config)
        