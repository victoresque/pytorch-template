from copy import copy
import numpy as np


class BaseDataLoader:
    """
    Base class for all data loaders
    """
    def __init__(self, config):
        self.config = config
        self.batch_size = config['data_loader']['batch_size']
        self.shuffle = config['data_loader']['shuffle']
        self.batch_idx = 0

    def __iter__(self):
        """
        :return: Iterator
        """
        assert self.__len__() > 0
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_data()
        return self

    def __next__(self):
        """
        :return: Next batch
        """
        packed = self._pack_data()
        if self.batch_idx < self.__len__():
            batch = packed[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            self.batch_idx = self.batch_idx + 1
            return self._unpack_data(batch)
        else:
            raise StopIteration

    def __len__(self):
        """
        :return: Total number of batches
        """
        return self._n_samples() // self.batch_size

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

    def split_validation(self):
        """
        Split validation data from data loader based on self.config['validation']
        """
        validation_split = self.config['validation']['validation_split']
        shuffle = self.config['validation']['shuffle']
        if validation_split == 0.0:
            return None
        if shuffle:
            self._shuffle_data()
        valid_data_loader = copy(self)
        split = int(self._n_samples() * validation_split)
        packed = self._pack_data()
        train_data = self._unpack_data(packed[split:])
        val_data = self._unpack_data(packed[:split])
        valid_data_loader._update_data(val_data)
        self._update_data(train_data)
        return valid_data_loader
