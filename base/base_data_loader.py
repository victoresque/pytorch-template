import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, valid_batch_size, shuffle, validation_split, validation_fold, num_workers, collate_fn=default_collate):
        """
        :param batch_size: Mini-batch size
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.shuffle = shuffle
        self._n_samples = len(self.dataset)
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        if validation_split is 0.0:
            self.sampler, self.valid_sampler = None, None
        else:
            self.sampler, self.valid_sampler = self._split_sampler(validation_split, validation_fold)
            self.shuffle = False # ignore shuffle option which is mutually exclusive with sampler

        self.init_kwargs = {
            'dataset': self.dataset,
            'num_workers': self.num_workers,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn
            }
        super(BaseDataLoader, self).__init__(**self.init_kwargs, sampler=self.sampler, batch_size=self.batch_size)

    def __len__(self):
        """
        :return: Total number of batches
        """
        return self._n_samples // self.batch_size

    def _split_sampler(self, split, fold):
        assert(split > 0.0)
        assert((fold + 1) * split < 1.0)
        idx_full = np.arange(self._n_samples)

        # TODO: make sure that this seed does not influence other sampling
        np.random.seed(0) 
        np.random.shuffle(idx_full)

        len_valid = int(self._n_samples * split)

        start = fold * len_valid
        stop = (fold + 1) * len_valid

        valid_idx = idx_full[start:stop]
        train_idx = np.delete(idx_full, np.arange(start, stop))
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self._n_samples = len(train_idx)
        return train_sampler, valid_sampler

    def get_valid_loader(self):
        if self.valid_sampler is None:
            return None
        else:
            valid_loader = DataLoader(**self.init_kwargs, sampler=self.valid_sampler, batch_size=self.valid_batch_size)
            return valid_loader