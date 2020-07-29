from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
    trsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.FashionMNIST(data_dir, train=training, download=True, transform=trsfm)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    if training:
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(validation_split, int):
            assert validation_split > 0
            assert validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = validation_split
        else:
            num_valid = int(num_total * validation_split)
        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])
        return DataLoader(train_dataset, **loader_args), \
               DataLoader(valid_dataset, **loader_args)
    else:
        return DataLoader(dataset, **loader_args)

