import os
from copy import copy
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_validation(data_loader, validation_split, randomized=True):
    if validation_split == 0.0:
        return data_loader, None
    valid_data_loader = copy(data_loader)
    if randomized:
        rand_idx = np.random.permutation(len(data_loader.x))
        data_loader.x = np.array([data_loader.x[i] for i in rand_idx])
        data_loader.y = np.array([data_loader.y[i] for i in rand_idx])
    split = int(len(data_loader.x) * validation_split)
    data_loader.x = data_loader.x[split:]
    data_loader.y = data_loader.y[split:]
    valid_data_loader.x = data_loader.x[:split]
    valid_data_loader.y = data_loader.y[:split]
    return data_loader, valid_data_loader
