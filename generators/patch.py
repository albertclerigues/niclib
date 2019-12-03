import copy
import itertools
import math
import warnings
from abc import ABC, abstractmethod, abstractproperty
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset as TorchDataset

import torch
import numpy as np

from niclib2.data import *

class PatchInstruction:
    def __init__(self, idx, center, shape, norm_stats=None, augment_fn=None):
        self.idx = idx
        self.center = center
        self.shape = shape
        self.norm_stats = norm_stats
        self.augment_fn = augment_fn


class PatchSet(TorchDataset):
    def __init__(self, images, sampling):
        """
        Creates a torch dataset that returns patches extracted from images according to the chosen sampling

        :param list images: list of images
        :param sampling: Sampling object (inherited from PatchSampling)
        """
        assert isinstance(sampling, PatchSampling), 'Invalid sampling'
        assert all([img.ndim == 4 for img in images]), 'Images must be numpy ndarrays with dimensions (C, X, Y, Z)'

        self.images = images
        self.instructions = sampling.build_instructions(images)
        print("Making PatchSet with {} patches".format(0))

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        instr = self.instructions[index]
        x_patch = copy.deepcopy(self.images[instr.idx][get_patch_slice(instr.center, instr.shape)])
        if instr.norm_stats is not None:
            x_patch = normalize_by_statistics(x_patch, *instr.norm_stats)
        if instr.augment_fn is not None:
            x_patch = instr.augment_fn(x_patch)
        x = torch.Tensor(np.ascontiguousarray(x_patch, dtype=np.float32))
        return x


class PatchSampling(ABC):
    """Abstract Base Class for arr sampling"""
    @abstractmethod
    def build_instructions(self, images):
        pass

class UniformSampling(PatchSampling):
    def __init__(self, shape, step, num_samples=None, masks=None, normalization=None, augmentation=None):
        self.shape = shape
        self.step = step if num_samples is None else (1, 1, 1)
        self.maximum = num_samples
        self.normalization = normalization
        self.augmentation = augmentation

        assert all([mask.ndim == 3 for mask in masks])
        self.masks = masks if masks is None else [mask.astype('float16') for mask in masks]

    def build_instructions(self, images):
        assert all([img.ndim == 4 for img in images])
        if self.masks is not None:
            assert len(self.masks) == len(images)

        span = [(math.ceil(dim / 2.0), math.floor(dim / 2.0)) for dim in self.shape]

        instructions = []
        for idx, img in enumerate(images):

            dim_centers = [list(range(span[i][0], img[0].shape[i] - span[i][1], self.step[i])) for i in range(3)]
            for center in itertools.product(*dim_centers):
                if self.masks is not None and self.masks[idx][center] == 0.0:
                    continue

                # TODO normalization

                # TODO augmentation

                instructions.append(PatchInstruction(
                    idx=idx, center=center, shape=self.shape, norm_stats=None, augment_fn=None))
        return instructions


class LesionSampling(PatchSampling):
    def __init__(self, shape, num_samples, masks=None, normalization=None, augmentation=None):
        self.shape = shape
        self.maximum = num_samples
        self.normalization = normalization
        self.augmentation = augmentation

        assert all([mask.ndim == 3 for mask in masks])
        self.masks = masks if masks is None else [mask.astype('float16') for mask in masks]

    def build_instructions(self, images):
        assert all([img.ndim == 4 for img in images])
        if self.masks is not None:
            assert len(self.masks) == len(images)

        span = [(math.ceil(dim / 2.0), math.floor(dim / 2.0)) for dim in self.shape]

        instructions = []

        for idx, img in enumerate(images):
            dim_centers = [list(range(span[i][0], img[0].shape[i] - span[i][1])) for i in range(3)]
            for center in itertools.product(*dim_centers):
                if self.masks is not None and self.masks[idx][center] == 0.0:
                    continue

                # TODO normalization

                # TODO augmentation

                instructions.append(PatchInstruction(
                    idx=idx, center=center, shape=self.shape, norm_stats=None, augment_fn=None))
        return instructions




def get_patch_slice(center, patch_shape):
    """
    :param tuple center: a tuple (x,y,z) or a list of tuples (x,y,z)
    :param tuple patch_shape: (x,y,z) tuple with arr dimensions
    :return: a tuple (channel_slice, x_slice, y_slice, z_slice) or a list of them
    """

    if not isinstance(center, list):
        center = [center]

    # Pre-compute arr sides for slicing
    half_sizes = [[dim // 2, dim // 2] for dim in patch_shape]
    for i in range(len(half_sizes)):  # If even dimension, subtract 1 to account for assymetry
        if patch_shape[i] % 2 == 0: half_sizes[i][1] -= 1

    patch_slices = []
    for c in center:  # Actually create slices
        patch_slice = (slice(None),  # slice(None) selects all channels
                       slice(c[0] - half_sizes[0][0], c[0] + half_sizes[0][1] + 1),
                       slice(c[1] - half_sizes[1][0], c[1] + half_sizes[1][1] + 1),
                       slice(c[2] - half_sizes[2][0], c[2] + half_sizes[2][1] + 1))
        patch_slices.append(patch_slice)

    return patch_slices if len(patch_slices) > 1 else patch_slices[0]
