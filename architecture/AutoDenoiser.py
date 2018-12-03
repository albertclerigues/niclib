import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

import numpy as np

from niclib.architecture.SUNet import SUNETx4, SUNETx5


class AutoDenoiser(torch.nn.Module):
    def __init__(self, noiser_module, denoiser_module, segmenter_module):
        super().__init__()
        warnings.warn("AutoDenoiser requires first modality (C=0) to be CT in an image of dims (B,C,X,Y,Z)")
        self.noiser = noiser_module
        self.denoiser = denoiser_module
        self.segmenter = segmenter_module

    def forward(self, x_in):
        x_ct = x_in[:, :1, ...]
        x_rest = x_in[:, 1:, ...]

        # DS: Denoising and Segmentation (DS) path
        x_ct_denoised = self.denoiser(x_ct)
        x_in_denoised = torch.cat([x_ct_denoised, x_rest], dim=1)
        x_segmented = self.segmenter(x_in_denoised)

        if self.training:
            # ND: Return also Noising and Denoising (ND) path for loss computation
            x_ct_N = self.noiser(x_ct)
            x_ct_ND = self.denoiser(x_ct_N)
            return [x_segmented, x_ct_ND]
        else:
            return x_segmented

    def print_noiser(self):
        for name, param in self.noiser.named_parameters():
            print('{} = {:.3f} (grad={})'.format(name, param.item(), param.grad))









