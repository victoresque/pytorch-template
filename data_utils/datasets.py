# Standard library imports:
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
# Local imports:
from data_utils.transforms import get_composed_transforms


def get_dataset(ds_config):
    """Returns dataset as specified in configuration.

    :param ds_config: 'dataset' section of configuration file
    :return: instance of dataset object
    """
    dataset_type = ds_config['type']

    try:
        dataset = eval(dataset_type)
    except NameError as e:
        raise NameError(f"Data loader ({dataset_type}) not found.")

    composed_transforms = get_composed_transforms(ds_config['transforms'])

    return dataset(transform=composed_transforms,
                   **ds_config['kwargs'])


def MnistDataset(*args, **kwargs):
    """Function returning MNIST class (as it's already implemented)."""
    return MNIST(*args, **kwargs)


class CustomDataset(Dataset):
    """Template for implementing your own custom dataset."""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        # Implement the rest of your initialization code.

        if self.train:
            self.train_data = None # Implement your data loading
            self.train_labels = None # Implement your data loading
        else:
            self.test_data = None # Implement your data loading
            self.test_labels = None # Implement your data loading

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            sample, target = self.train_data[index], self.train_labels[index]
        else:
            sample, target = self.test_data[index], self.test_labels[index]

        return sample, target
