import torch
import numpy as np

from math import ceil, floor
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchGenerator

from niclib.patch.instructions import *

from niclib.volume import zeropad_set

class PatchGeneratorBuilder:
    def __init__(self, instruction_generator, batch_size, zeropad_shape=False, shuffle=False, num_workers=4):
        assert isinstance(instruction_generator, NIC_InstructionGenerator)
        self.bs = batch_size
        self.instr_gen = instruction_generator
        self.zeropad_shape = zeropad_shape

        self.shuffle = shuffle
        self.num_workers = num_workers

    def build_patch_generator(self, images, return_instructions=False):
        if not isinstance(images, list):
            images = [images]

        if self.zeropad_shape is not None:
            images = zeropad_set(images, self.zeropad_shape)

        instructions = self.instr_gen.generate_instructions(images)
        patch_set = InstructionPatchSet(images, instructions)
        patch_gen = torchGenerator(patch_set, self.bs, shuffle=self.shuffle, num_workers=self.num_workers)

        if return_instructions:
            return patch_gen, instructions
        return patch_gen

class InstructionPatchSet(torchDataset):
    def __init__(self, images, instructions):
        self.images = images
        self.instructions = instructions

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        instruction = self.instructions[index]
        sample = instruction.extract_from(self.images)
        return sample