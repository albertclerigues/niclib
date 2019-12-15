from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader

def make_generator(set, batch_size, shuffle, num_workers=4):
    return DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class ZipSets(TorchDataset):
    def __init__(self, sets):
        assert all([isinstance(set, DataLoader) for set in sets])
        self.sets = sets

    def __len__(self):
        return len(self.sets[0])

    def __getitem__(self, index):
        return tuple([dataset[index] for dataset in self.sets])


class ChainSets(TorchDataset):
    def __init__(self, sets):
        assert all([isinstance(s, DataLoader) for s in sets])
        self.sets = sets
        self.index_to_set = [] # List with set idx, and subset idx
        for set_idx, s in enumerate(self.sets):
            self.index_to_set += [(set_idx, subset_idx) for subset_idx in range(len(s))]

    def __len__(self):
        return sum([len(s) for s in self.sets])

    def __getitem__(self, index):
        set_idx, subset_idx = self.index_to_set[index]
        return self.sets[set_idx][subset_idx]


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