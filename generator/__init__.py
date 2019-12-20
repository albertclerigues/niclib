from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader

from .patch import *

def make_generator(set, batch_size, shuffle, num_workers=4):
    return DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class ZipSet(TorchDataset):
    def __init__(self, sets):
        assert all([isinstance(set, TorchDataset) for set in sets])
        self.sets = sets

    def __len__(self):
        return min(len(s) for s in self.sets)

    def __getitem__(self, index):
        return tuple([dataset[index] for dataset in self.sets])


class ChainSet(TorchDataset):
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
    def __init__(self, items):
        assert isinstance(items, list)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class FunctionSet(TorchDataset):
    def __init__(self, function, num_samples):
        assert callable(function)
        self.function = function
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.function()