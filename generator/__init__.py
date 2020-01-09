from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader

from .patch import *

def make_generator(set, batch_size, shuffle, num_workers=4):
    """Makes a generator from a given set that returns elements with the specified batch size.

    :param torch.utils.data.Dataset set: the data _set_.
    :param batch_size: batch size of generated items.
    :param shuffle: whether to shuffle the generator elements each epoch (SHOULD be True for training generators).
    :param num_workers: (default: 4) number of parallel workers getting items.

    :Example:

    >>> myset = ListSet(list(range(10)))
    >>> mygen = make_generator(myset, batch_size=2, shuffle=False)
    >>> print([x for x in mygen])
    [tensor([0, 1]), tensor([2, 3]), tensor([4, 5]), tensor([6, 7]), tensor([8, 9])]
    """

    return DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class ZipSet(TorchDataset):
    """Zips a list of sets and returns a tuple with the ith element of each given set.

    :param sets: list or tuple of sets.

    :Example:

    >>> zipped_set = ZipSet([ListSet([1,2,3]), ListSet([1,2,3])])
    >>> print([x for x in zipped_set])
    [(1, 1), (2, 2), (3, 3)]
    """

    def __init__(self, sets):
        assert all([isinstance(set, TorchDataset) for set in sets])
        self.sets = sets

    def __len__(self):
        return min(len(s) for s in self.sets)

    def __getitem__(self, index):
        return tuple([dataset[index] for dataset in self.sets])


class ChainSet(TorchDataset):
    """Chains a list of sets one after the other.

    :param sets: list or tuple of sets.

    :Example:

    >>> chained_set = ChainSet([ListSet([1,2,3]), ListSet([4,5,6])])
    >>> print([x for x in chained_set])
    [1, 2, 3, 4, 5, 6]
    """

    def __init__(self, sets):
        assert all([isinstance(s, TorchDataset) for s in sets])
        self.sets = sets
        self.index_to_set = [] # List with set idx, and subset idx
        for set_idx, s in enumerate(self.sets):
            self.index_to_set += [(set_idx, set_subidx) for set_subidx in range(len(s))]

    def __len__(self):
        return sum([len(s) for s in self.sets])

    def __getitem__(self, index):
        set_idx, set_subidx = self.index_to_set[index]
        return self.sets[set_idx][set_subidx]


class ListSet(TorchDataset):
    """Creates a data set from a given list.

    :param items: list of elements to return.

    :Example:

    >>> myset = ListSet([1, 2, 3])
    >>> print(myset[0])
    1
    """

    def __init__(self, items):
        assert isinstance(items, list)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class FunctionSet(TorchDataset):
    """Creates a data set that returns the result of a function call.

    :param callable function: function that returns a set element when called. This function will be called with no arguments.
    :param num_samples: desired length of the set.

    :Example:

    >>> get_random_int = lambda : np.random.randint(0, 1000)
    >>> myset = FunctionSet(function=get_random_int, num_samples=10)
    >>> print([myset[i] for i in range(10)])
    [895, 688, 442, 537, 317, 26, 268, 319, 786, 153]
    """

    def __init__(self, function, num_samples):
        assert callable(function)
        self.function = function
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index >= self.num_samples:
            return None
        return self.function()