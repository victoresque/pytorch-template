from copy import copy
import numpy as np


class BaseDataLoader:
    """
    Base class for all data loaders
    """
    def __init__(self, batch_size, shuffle):
        """
        :param batch_size: Mini-batch size
        :param shuffle: If shuffle is True, samples are shuffled upon calling __iter__()
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        """
        :return: Iterator
        """
        self.n_batch = self._n_samples() // self.batch_size
        self.batch_idx = 0
        assert self.n_batch > 0
        if self.shuffle:
            self._shuffle_data()
        return self

    def __next__(self):
        """
        :return: Next batch
        """
        packed = self._pack_data()
        if self.batch_idx < self.n_batch:
            batch = packed[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            self.batch_idx = self.batch_idx + 1
            return self._unpack_data(batch)
        else:
            raise StopIteration

    def __len__(self):
        """
        :return: Total number of batches
        """
        return NotImplementedError

    def _n_samples(self):
        """
        :return: Total number of samples
        """
        return NotImplementedError

    def _pack_data(self):
        """
        Pack all data into a list/tuple/ndarray/...

        :return: Packed data in the data loader
        """
        return NotImplementedError

    def _unpack_data(self, packed):
        """
        Unpack packed data (from _pack_data())

        :param packed: Packed data
        :return: Unpacked data
        """
        return NotImplementedError

    def _update_data(self, unpacked):
        """
        Update data member in the data loader

        :param unpacked: Unpacked data (from _update_data())
        """
        return NotImplementedError

    def _shuffle_data(self):
        """
        Shuffle data members in the data loader
        """
        packed = self._pack_data()
        rand_idx = np.random.permutation(len(packed))
        packed = [packed[i] for i in rand_idx]
        self._update_data(self._unpack_data(packed))

    def split_validation(self, validation_split, shuffle=False):
        """
        Validation data splitting

        :param validation_split: Ratio of validation data, 0.0 means no validation data
        :param shuffle: Shuffles all training samples before splitting
        :return: Validation data loader, which is the same class as original data loader

        Note:
            After calling data_loader.split_validation(), data_loader will be changed
        """
        if validation_split == 0.0:
            return None
        valid_data_loader = copy(self)
        if shuffle:
            self._shuffle_data()
        split = int(self._n_samples() * validation_split)
        packed = self._pack_data()
        train_data = self._unpack_data(packed[split:])
        val_data = self._unpack_data(packed[:split])
        valid_data_loader._update_data(val_data)
        self._update_data(train_data)
        return valid_data_loader

