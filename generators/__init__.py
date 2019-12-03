from abc import ABC, abstractclassmethod
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as TorchDataset


def DataGenerator(set, batch_size, shuffle, num_workers=4):
    return DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class ZipSets(TorchDataset):
    def __init__(self, sets):
        assert all([isinstance(set, TorchDataset) for set in sets])
        self.sets = sets

    def __len__(self):
        return len(self.sets[0])

    def __getitem__(self, index):
        return tuple([dataset[index] for dataset in self.sets])


class ListSet(TorchDataset):
    def __init__(self, items):
        assert isinstance(items, list)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
