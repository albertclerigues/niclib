import torch
import numpy as np

from math import ceil, floor
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchGenerator

from niclib.patch.extraction import *
from niclib.patch.instructions import extract_patch_with_instruction

from niclib.volume import zeropad_set

class InstructionGenerator:
    def __init__(self, batch_size, in_shape, out_shape, sampler, augment_to=None, autoencoder=False, zeropad_images=False, shuffle=False, num_workers=4):
        self.bs = batch_size
        self.sampler = sampler
        self.in_shape, self.out_shape = in_shape, out_shape
        self.shuffle, self.num_workers = shuffle, num_workers
        self.augment_num = augment_to
        self.zeropad = zeropad_images
        self.autoencoder = autoencoder

    def build_patch_generator(self, images, return_instructions=False):
        if self.zeropad:
            images = zeropad_set(images, self.in_shape)

        if isinstance(images, list):
            instructions = build_set_extraction_instructions(images, self.in_shape, self.out_shape, self.sampler, self.augment_num)
        else:
            instructions = build_sample_extraction_instructions(images, self.in_shape, self.out_shape, self.sampler, self.augment_num)

        patch_set = PatchSet(images, instructions, self.autoencoder)
        patch_gen = torchGenerator(patch_set, self.bs, shuffle=self.shuffle, num_workers=self.num_workers)

        if not return_instructions:
            return patch_gen
        else:
            return patch_gen, instructions

class PatchSet(torchDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, images, instructions, autoencoder=False):
        self.images = images
        self.instructions = instructions
        self.autoencoder = autoencoder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.instructions)

    def __getitem__(self, index):
        'Generates one sample of data'
        x, y = extract_patch_with_instruction(self.images, self.instructions[index])
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

        if self.autoencoder:
            return x, x
        else:
            return x, y

