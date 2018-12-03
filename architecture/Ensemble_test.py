import torch


class Ensemble_TEST(torch.nn.Module):
    def __init__(self, filepaths, device=torch.device('cuda')):
        super().__init__()
        self.models = torch.nn.ModuleList([torch.load(fp, device) for fp in filepaths])

    def forward(self, x_in):
        return torch.mean(torch.stack([m(x_in) for m in self.models], dim=0), dim=0)


