import numpy as np
from base import BaseDataLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_data_loaders(loader_config, dataset, config_full=None):
    """Returns training and validation data loaders, as specified in configuration.

    :param loader_config: data loader configuration
    :param dataset: instance of dataset class
    :param config_full: only required for legacy 'base_data_loader' class (not for standard PyTorch loader)
    :return: two data loaders: loader_train, loader_val
    """
    loader_type = loader_config['type']

    # Handling validation split:
    len_dataset = int(len(dataset))
    len_val = int(loader_config['validation']['split']*len_dataset)

    indices = list(range(len_dataset))

    # TODO: Consider enabling saving train/validation split to ensure train/validation split
    # TODO cont. is always the same (even if training continues on different machine).
    # Ensure same validation set over multiple runs:
    np.random.seed(0)

    val_idx = np.random.choice(indices,
                               size=len_val,
                               replace=False)
    train_idx = list(set(indices) - set(val_idx))

    if loader_config['shuffle_data']:
        sampler_val = SubsetRandomSampler(val_idx)
        sampler_train = SubsetRandomSampler(train_idx)

    if loader_type == 'PyTorch':
        loader_train = DataLoader(dataset=dataset,
                                  sampler=sampler_train,
                                  **loader_config['train']['kwargs'])
        loader_val = DataLoader(dataset=dataset,
                                sampler=sampler_val,
                                **loader_config['validation']['kwargs'])

    else:
        print("Standard PyTorch dataloader is NOT selected, will look for custom loader.")

        if config_full is None:
            # TODO: Evaluate removing dependency on 'config_full' from BaseDataLoader
            raise ValueError("'get_data_loader' needs 'config' for custom data loader.")
        try:
            loader_ = eval(loader_type)

            loader_train = loader_(dataset=dataset, config=config_full)
            loader_val = loader_train.split_validation()

        except NameError as e:
            raise NameError(f"Data loader ({loader_type}) not found.")

    return loader_train, loader_val


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, config):
        self.dataset = dataset
        super(MnistDataLoader, self).__init__(self.dataset, config)
        