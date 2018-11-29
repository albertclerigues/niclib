import torch
import torch.nn as nn

from torch.distributions import Normal, Poisson

class DropoutPrediction(nn.Module):
    def __init__(self, forever_inactive=False, ndims=3):
        super().__init__()
        self.Dropout = nn.Dropout2d if ndims is 2 else nn.Dropout3d
        self.forever_inactive = forever_inactive

        self.active = False
        self.p_out = 0.0
        self.d = self.Dropout(p=0.0)

    def forward(self, x_in):
        if not self.active or self.forever_inactive or self.training:
           return x_in

        assert self.p_out is not 0
        self.d.train()
        x_out = self.d(x_in) * self.p_out
        self.d.eval()
        return x_out

    def activate(self, p_out):
        self.active = True
        self.d = self.Dropout(p=p_out)
        self.p_out = p_out

    def deactivate(self):
        self.active = False
        self.d = self.Dropout(p=0.0)
        self.p_out = 0.0

class CTNoiser(torch.nn.Module):
    """
    Module for additive CT image noise, a mixture of Gaussian and Poisson
    """
    def __init__(self):
        super().__init__()
        self.scale_normal = torch.nn.Parameter(torch.tensor(1.0))
        self.normal_mean = torch.nn.Parameter(torch.tensor(0.0))
        self.normal_std = torch.nn.Parameter(torch.tensor(1.0))

        # Here, this is an "emulated" poisson distribution approximated by a normal distribution
        # The clamp constraints for positivity like in the poisson distribution
        self.scale_poisson = torch.nn.Parameter(torch.tensor(1.0))
        self.poisson_mean = torch.nn.Parameter(torch.tensor(0.0))
        self.poisson_std = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x_in):
        normal_noise =  (Normal(0.0, 1.0).sample(x_in.size()) * self.normal_std) + self.normal_mean
        poisson_emulated = (Normal(0.0, 1.0).sample(x_in.size()) * self.poisson_std) + self.poisson_mean
        poisson_emulated = torch.clamp(poisson_emulated, min=0.0)
        return x_in + self.scale_poisson * poisson_emulated + self.scale_normal * normal_noise


class CTNoiser_old(torch.nn.Module):
    """
    Module for additive CT image noise, a mixture of Gaussian and Poisson
    """

    def __init__(self):
        super().__init__()
        self.scale_normal = torch.nn.Parameter(torch.tensor(1.0))
        self.normal_mean = torch.nn.Parameter(torch.tensor(0.0))
        self.normal_std = torch.nn.Parameter(torch.tensor(1.0))

        self.scale_poisson = torch.nn.Parameter(torch.tensor(1.0))
        self.poisson_rate = torch.nn.Parameter(torch.tensor(1.0).float())

    def forward(self, x_in):
        normal_noise = self.scale_normal * ((Normal(0.0, 1.0).sample(x_in.size()) * self.normal_std) + self.normal_mean)
        poisson_noise = self.scale_poisson * torch.poisson(self.poisson_rate + torch.zeros_like(x_in))
        return x_in + poisson_noise  + normal_noise
