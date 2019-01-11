import sys

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


import warnings

import numpy as np

from niclib.architecture.SUNet import SUNETx4, SUNETx5
from niclib.network.layers import CTNoiser

import visdom
viz = visdom.Visdom()

class AutoDenoiser(torch.nn.Module):
    def __init__(self, noiser_module, denoiser_module, segmenter_module):
        super().__init__()
        warnings.warn("AutoDenoiser requires first modality (C=0) to be CT in an image of dims (B,C,X,Y,Z)")
        self.noiser = noiser_module
        self.denoiser = denoiser_module
        self.segmenter = segmenter_module

        self.X = torch.tensor([0.0])

        self.vis_interval = 20
        self.vis_counter = 0

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

            if self.vis_counter % self.vis_interval == 0:
                self.vis_counter = 0
                normalize_img = lambda img : (255.0 * ((img - np.min(img)) / (np.max(img) - np.min(img)))).astype('uint8')
                resize_img = lambda img : cv2.resize(img, (320, 320))

                # Visdom printing
                viz.image(resize_img(normalize_img(x_ct[0, 0].detach().cpu().numpy())), win='CT_IN', opts=dict(caption='CT_IN'))
                viz.image(resize_img(normalize_img(x_ct_denoised[0, 0].detach().cpu().numpy())), win='DEN_CT', opts=dict(caption='DEN_CT'))
                viz.image(resize_img(normalize_img(x_segmented[0, 1].detach().cpu().numpy())), win='SEG', opts=dict(caption='SEG'))
                viz.image(resize_img(normalize_img(x_ct_N[0, 0].detach().cpu().numpy())), win='X_N', opts=dict(caption='X_N'))
                viz.image(resize_img(normalize_img(x_ct_ND[0, 0].detach().cpu().numpy())), win='X_ND', opts=dict(caption='X_ND'))
            self.vis_counter += 1

            return [x_segmented, x_ct_ND, x_ct_denoised]
        else:
            return x_segmented

    def print_noiser(self):
        for name, param in self.noiser.named_parameters():
            print('{} = {:.3f} (grad={})'.format(name, param.item(), param.grad))

    def update_visdom(self):
        #viz = visdom.Visdom()
        viz.line(Y=torch.tensor([self.noiser.normal_scale.item()]), X=self.X, update='append', win='scale', opts=dict(title='scale'))
        viz.line(Y=torch.tensor([self.noiser.normal_mean.item()]), X=self.X, update='append', win='mean', opts=dict(title='mean'))
        viz.line(Y=torch.tensor([self.noiser.normal_std.item()]), X=self.X, update='append', win='std', opts=dict(title='std'))
        self.X = self.X + torch.tensor([1.0])








