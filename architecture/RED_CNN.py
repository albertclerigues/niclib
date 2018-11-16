import torch
import torch.nn as nn
import torch.nn.functional as F


class RED_CNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ndims=2, nfilts=96):
        super(RED_CNN, self).__init__()

        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d

        self.in_conv = nn.Sequential(
            Conv(in_ch, nfilts, 5),
            nn.ReLU())

        self.encoder_block = nn.Sequential(
            Conv(nfilts, nfilts, 5),
            nn.ReLU())

        self.deconv = ConvTranspose(nfilts, nfilts, 5)
        self.relu = nn.ReLU()

        self.out_conv = nn.Sequential(
            ConvTranspose(nfilts, out_ch, 5),
            nn.ReLU())

    def forward(self, x_in):
        e1 = self.in_conv(x_in)
        e2 = self.encoder_block(e1)
        e3 = self.encoder_block(e2)
        e4 = self.encoder_block(e3)
        e5 = self.encoder_block(e4)

        dec1_in = self.deconv(e5)
        dec1_out = self.relu(dec1_in + e4)

        dec2_in = self.deconv(dec1_out)
        dec2_out = self.relu(dec2_in)

        dec3_in = self.deconv(dec2_out)
        dec3_out = self.relu(dec3_in + e2)

        dec4_in = self.deconv(dec3_out)
        dec4_out = self.relu(dec4_in)

        dec5_in = self.out_conv(dec4_out)
        dec5_out = self.relu(dec5_in + x_in)

        return  dec5_out


