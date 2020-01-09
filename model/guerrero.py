import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class uResNet_guerrero(nn.Module):
    """uResNet model as defined in [Guerrero2018]_ (https://www.ncbi.nlm.nih.gov/pubmed/29527496)

    .. rubric:: References

    .. [Guerrero2018] Guerrero, R., et al.: "White matter hyperintensity and stroke lesion segmentation and differentiation using convolutional neural networks." NeuroImage: Clinical 17 (2018): 918-934.

    :param int in_ch: number of input channels.
    :param int out_ch: number of output channels.
    :param int ndims: either 2 or 3 dimensions.
    :param torch.nn.Module activation: (default: None) activation function applied before returning output.
        Not using an activation is useful for loss functions that use logits as input for numerical stability
        (i.e. ``torch.nn.CrossEntropyLoss``, ``nn.BCEWithLogitsLoss``, ...).
    :param num_base_filters: number of filters in the first resolution step.
        The number of filters is doubled in each of the four resolution steps.
        The number of channels in the latent space will be equal to ``8 * num_base_filters``.
    :param bool skip_connections: (default: True) wether or not to have skip connections between encoder and
        decoder branches of the uResNet. This is useful for autoencoders where they would be a direct shortcut and
        prevent the latent space from learning useful features.
    """

    def __init__(self, in_ch, out_ch, ndims, activation=None, num_base_filters=32, skip_connections=True):
        super().__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d
        MaxPool = nn.MaxPool2d if ndims is 2 else nn.MaxPool3d

        self.use_skip = skip_connections

        self.enc_res1 = _ResEle(in_ch, num_base_filters, ndims=ndims)
        self.enc_res2 = _ResEle(1 * num_base_filters, 2 * num_base_filters, ndims=ndims)
        self.enc_res3 = _ResEle(2 * num_base_filters, 4 * num_base_filters, ndims=ndims)
        self.enc_res4 = _ResEle(4 * num_base_filters, 8 * num_base_filters, ndims=ndims)

        self.pool1 = MaxPool(2)
        self.pool2 = MaxPool(2)
        self.pool3 = MaxPool(2)

        self.dec_res1 = _ResEle(4 * num_base_filters, 4 * num_base_filters, ndims=ndims)
        self.dec_res2 = _ResEle(4 * num_base_filters, 2 * num_base_filters, ndims=ndims)
        self.dec_res3 = _ResEle(2 * num_base_filters, 2 * num_base_filters, ndims=ndims)
        self.dec_res4 = _ResEle(1 * num_base_filters, 1 * num_base_filters, ndims=ndims)

        self.deconv1 = ConvTranspose(
            8 * num_base_filters, 4 * num_base_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = ConvTranspose(
            2 * num_base_filters, 2 * num_base_filters, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = ConvTranspose(
            2 * num_base_filters, 1 * num_base_filters, 3, stride=2, padding=1, output_padding=1)

        self.out_conv = Conv(num_base_filters, out_ch, 1)
        self.activation = activation

    def forward(self, x_in):
        l1_end = self.enc_res1(x_in)

        l2_start = self.pool1(l1_end)
        l2_end = self.enc_res2(l2_start)

        l3_start = self.pool2(l2_end)
        l3_end = self.enc_res3(l3_start)

        l4_start = self.pool3(l3_end)
        l4_end = self.enc_res4(l4_start)

        r4_start = self.deconv1(l4_end)
        r4_end = self.dec_res1(r4_start)

        r3_start = self.dec_res2(r4_end + l3_end) if self.use_skip else self.dec_res2(r4_end)
        r3_end = self.deconv2(r3_start)

        r2_start = self.dec_res3(r3_end + l2_end) if self.use_skip else self.dec_res3(r3_end)
        r2_end = self.deconv3(r2_start)

        r1_start = self.dec_res4(r2_end + l1_end) if self.use_skip else self.dec_res4(r2_end)
        pred = self.out_conv(r1_start)

        if self.activation is not None:
            pred = self.activation(pred)
        return pred


class _ResEle(torch.nn.Module):
    def __init__(self, nch_in, nch_out, ndims=2):
        super().__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d

        self.selection_path = Conv(nch_in, nch_out, 1)
        self.conv_path = Conv(nch_in, nch_out, 3, padding=1)
        self.output_path = nn.Sequential(
            BatchNorm(nch_out),
            nn.ReLU())

    def forward(self, x_in):
        return self.output_path(self.conv_path(x_in) + self.selection_path(x_in))