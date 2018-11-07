import itertools as iter

from niclib.patch.centers import *
from niclib.patch.instructions import *
from niclib.patch.slices import *
from niclib.patch.sampling import *
from niclib.io.terminal import printProgressBar

def build_set_extraction_instructions(images, in_shape, out_shape, sampler, augment_to=None):
    assert isinstance(sampler, NICSampler)

    set_instructions = []
    for idx, image in enumerate(images):
        printProgressBar(idx, len(images), suffix='samples processed')

        centers = sampler.get_centers(image)

        if isinstance(centers, tuple): # Sampling that have two sets of centers
            pos_centers, unif_centers = centers
            pos_instructions = get_instructions_from_centers(idx, pos_centers, in_shape, out_shape, augment_to=augment_to)
            unif_instructions = get_instructions_from_centers(idx, unif_centers, in_shape, out_shape, augment_to=None)
            image_instructions = pos_instructions + unif_instructions
        else:
            image_instructions = get_instructions_from_centers(idx, centers, in_shape, out_shape, augment_to=augment_to)

        set_instructions += image_instructions

    printProgressBar(len(images), len(images), suffix='samples processed')
    return set_instructions

def get_instructions_from_centers(sample_idx, centers, patch_shape, output_shape, augment_to=None):
    data_slices = get_patch_slices(centers, patch_shape)
    label_slices = get_patch_slices(centers, output_shape)

    sample_instructions = list()
    for data_slice, label_slice in zip(data_slices, label_slices):
        instruction = PatchExtractInstruction(
            sample_idx=sample_idx, data_patch_slice=data_slice, label_patch_slice=label_slice)
        sample_instructions.append(instruction)

    if augment_to is not None:
        sample_instructions = augment_instructions(sample_instructions, goal_num_instructions=augment_to)

    return sample_instructions



