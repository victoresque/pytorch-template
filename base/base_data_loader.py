
class BaseDataLoader:
    """ Base class for all data loaders.

    Note:
        No need to modify this in most cases.
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        return NotImplementedError

    def __next__(self):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError

