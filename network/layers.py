import torch
import torch.nn as nn

from torch.distributions import Normal, Poisson

class DropoutPrediction(nn.Module):
    def __init__(self, inactive=False, ndims=3):
        """
        :param inactive: if True, layer cannot be activated, this avoids changing network design if dropout is not required
        :param ndims:
        """
        super().__init__()
        #self.Dropout = nn.Dropout2d if ndims is 2 else nn.Dropout3d
        self.Dropout = nn.AlphaDropout
        self.forever_inactive = inactive

        self._running = False
        self.d = self.Dropout(p=0.5)

    def forward(self, x_in):
        if not self._running or self.forever_inactive:
            self.d.eval()
        else:
            self.d.train()

        return self.d(x_in)

    def activate(self, p_out=None):
        self._running = True
        if p_out is not None:
            self.d = self.Dropout(p=p_out)

    def deactivate(self):
        self._running = False

class CTNoiser(torch.nn.Module):
    """
    Module for additive CT image noise, a mixture of Gaussian and Poisson
    """
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device

        self.scale_normal = torch.nn.Parameter(torch.tensor(1.0))
        self.normal_mean = torch.nn.Parameter(torch.tensor(0.0))
        self.normal_std = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x_in):
        normal_noise =  (Normal(0.0, 1.0).sample(x_in.size()).to(self.device) * self.normal_std) + self.normal_mean
        return x_in + self.scale_normal * normal_noise

class CTNoiser_extended(torch.nn.Module):
    """
    Module for additive CT image noise, a mixture of Gaussian and Poisson
    """
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device

        self.scale_normal = torch.nn.Parameter(torch.tensor(0.01))
        self.normal_mean = torch.nn.Parameter(torch.tensor(0.0))
        self.normal_std = torch.nn.Parameter(torch.tensor(1.0))

        # Here, this is an "emulated" poisson distribution approximated by a normal distribution
        # The clamp constraints for positivity like in the poisson distribution
        self.scale_poisson = torch.nn.Parameter(torch.tensor(0.01))
        self.poisson_mean = torch.nn.Parameter(torch.tensor(0.0))
        self.poisson_std = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x_in):
        normal_noise =  (Normal(0.0, 1.0).sample(x_in.size()).to(self.device) * self.normal_std) + self.normal_mean
        poisson_emulated = (Normal(0.0, 1.0).sample(x_in.size()).to(self.device) * self.poisson_std) + self.poisson_mean
        poisson_emulated = torch.clamp(poisson_emulated, min=0.0)
        return x_in + self.scale_poisson * poisson_emulated + self.scale_normal * normal_noise

