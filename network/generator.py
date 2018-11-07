import torch
import numpy as np

from math import ceil, floor
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchGenerator

from niclib.patch.extraction import build_set_extraction_instructions
from niclib.patch.instructions import extract_patch_with_instruction

def get_patch_generator(images, batch_size, in_shape, out_shape, sampling, sampling_options, shuffle=True, num_workers=4):
    assert sampling in {'uniform', 'hybrid'}

    # Get instructions according to desired sampling
    instructions = build_set_extraction_instructions(images, in_shape, out_shape, sampling, sampling_options)

    patch_set = PatchSet(images, instructions)
    return torchGenerator(patch_set, batch_size, shuffle=shuffle, num_workers=num_workers)

class PatchSet(torchDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, images, instructions):
        self.images = images
        self.instructions = instructions

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.instructions)

    def __getitem__(self, index):
        'Generates one sample of data'
        x, y = extract_patch_with_instruction(self.images, self.instructions[index])
        x, y = torch.from_numpy(x).float(), torch.from_numpy(np.squeeze(y, axis=0)).long()
        return x, y

