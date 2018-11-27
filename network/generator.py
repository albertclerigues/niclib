import torch
import numpy as np

from math import ceil, floor
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchGenerator

from niclib.patch.extraction import *
from niclib.patch.instructions import extract_patch_with_instruction

from niclib.volume import zeropad_set

class InstructionGenerator:
    def __init__(self, batch_size, in_shape, out_shape, sampler, augment_to=None, autoencoder=False, autoencoder_noise=False, zeropad_images=False, shuffle=False, num_workers=4):
        self.bs = batch_size
        self.sampler = sampler
        self.in_shape, self.out_shape = in_shape, out_shape
        self.shuffle, self.num_workers = shuffle, num_workers
        self.augment_num = augment_to
        self.zeropad = zeropad_images

        self.autoencoder = autoencoder
        self.autoencoder_noise = autoencoder_noise

    def build_patch_generator(self, images, return_instructions=False):
        if self.zeropad:
            images = zeropad_set(images, self.in_shape)

        if isinstance(images, list):
            instructions = build_set_extraction_instructions(images, self.in_shape, self.out_shape, self.sampler, self.augment_num)
        else:
            instructions = build_sample_extraction_instructions(images, self.in_shape, self.out_shape, self.sampler, self.augment_num)

        if self.autoencoder:
            patch_set = AutoencoderPatchSet(images, instructions, add_noise=self.autoencoder_noise)
        else:
            patch_set = SegmentationPatchSet(images, instructions)

        patch_gen = torchGenerator(patch_set, self.bs, shuffle=self.shuffle, num_workers=self.num_workers)

        if return_instructions:
            return patch_gen, instructions

        return patch_gen



class AutoencoderPatchSet(torchDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, images, instructions, add_noise=False):
        self.images = images
        self.instructions = instructions
        self.add_noise = add_noise

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.instructions)

    def __getitem__(self, index):
        'Generates one sample of data'
        x, _ = extract_patch_with_instruction(self.images, self.instructions[index], normalise=False)
        x = torch.from_numpy(x).float()

        if self.add_noise:
            noise_scale = np.random.uniform(low=2.0, high=10.0)

            poisson = torch.distributions.Poisson(rate=noise_scale)
            gaussian = torch.distributions.Normal(loc=0.0, scale=noise_scale)

            x_in = x + poisson.sample(x.shape) + gaussian.sample(x.shape)
        else:
            x_in = x

        return x_in, x

class SegmentationPatchSet(torchDataset):
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
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        return x, y

