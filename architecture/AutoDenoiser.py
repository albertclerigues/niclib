import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from niclib.architecture.SUNet import SUNETx4, SUNETx5


class AutoDenoiser(torch.nn.Module):
    def __init__(self, noiser_module, denoiser_module, segmenter_module):
        super().__init__()
        self.noiser = noiser_module
        self.denoiser = denoiser_module
        self.segmenter = segmenter_module

    def forward(self, x_in):
        xN = self.noiser(x_in)
        xN_D = self.denoiser(xN)

        x_denoised = self.denoiser(x_in)
        x_segmented = self.segmenter(x_denoised)

        if self.training:
            return xN_D, x_segmented





