from abc import ABC, abstractmethod

import torch
import numpy as np

from niclib.dataset.NICdataset import NICimage
from niclib.network.generator import InstructionGenerator
from niclib.patch.instructions import PatchExtractInstruction

from niclib.volume import zeropad_sample, remove_zeropad_volume

from niclib.io.terminal import printProgressBar

class Predictor(ABC):
    @abstractmethod
    def predict_sample(self, model, sample):
        pass

class PatchPredictor(Predictor):
    """
    Predicts a whole volume using patches with the provided model
    """

    def __init__(self, instruction_generator, num_classes, lesion_class=None, device=torch.device('cuda')):
        assert isinstance(instruction_generator, InstructionGenerator)
        self.instr_gen = instruction_generator

        self.num_classes = num_classes
        self.lesion_class = lesion_class
        self.device = device

    def predict_sample(self, model, sample_in):
        assert isinstance(sample_in, NICimage)
        print("Predicting sample with id:{}".format(sample_in.id))

        sample = zeropad_sample(sample_in, self.instr_gen.in_shape)

        batch_size = self.instr_gen.bs
        sample_generator, instructions = self.instr_gen.build_patch_generator(sample, return_instructions=True)

        voting_img = np.zeros((self.num_classes, ) + sample.data[0].shape, dtype=np.float32)
        counting_img = np.zeros_like(voting_img)

        model.eval()
        model.to(self.device)
        with torch.no_grad():  # Turns off autograd (faster exec)
            for batch_idx, (x, y) in enumerate(sample_generator):
                printProgressBar(batch_idx, len(sample_generator), suffix=' patches predicted')

                x, y = x.to(self.device), y.to(self.device)
                y_pred = model(x).cpu().numpy()

                if len(y_pred.shape) == 4:  # Add third dimension to 2D patches
                    y_pred = np.expand_dims(y_pred, axis=-1)

                batch_slice = slice(batch_idx*batch_size, (batch_idx + 1)*batch_size)
                batch_instructions = instructions[batch_slice]

                assert len(y_pred) == len(batch_instructions)
                for patch_pred, patch_instruction in zip(y_pred, batch_instructions):
                    voting_img[patch_instruction.data_patch_slice] += patch_pred
                    counting_img[patch_instruction.data_patch_slice] += np.ones_like(patch_pred)
            printProgressBar(len(sample_generator), len(sample_generator), suffix=' patches predicted')

        counting_img[counting_img == 0.0] = 1.0 # Avoid division by 0
        volume_probs = np.divide(voting_img, counting_img)

        if self.lesion_class is not None:
            volume_probs = volume_probs[self.lesion_class]
        else:
            volume_probs = np.squeeze(volume_probs, axis=0)

        volume_probs = remove_zeropad_volume(volume_probs, self.instr_gen.in_shape)

        assert np.array_equal(volume_probs.shape, sample_in.foreground.shape), (volume_probs.shape, sample_in.foreground.shape)

        return volume_probs


