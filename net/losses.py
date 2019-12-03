import torch
from torch import nn
import torch.nn.functional as F
from snorkel.classification import cross_entropy_with_probs

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=(0.25, 0.75), with_logits=False):
        """
        TODO TEST

        Constructor for the binary focal loss
        :param gamma:
        :param alpha: alpha is weight for class 0 (background)

        Usage:
        :param output:
        :param target:
        """
        super().__init__()
        self.g = gamma
        self.w = weights
        self.with_logits = with_logits

    def forward(self, output, target):
        output = torch.clamp(output, min=1E-7, max=1.0 - 1E-7) # numerical stability
        if self.with_logits:
            output = F.log_softmax(output, dim=1)

        weight_tensor = torch.stack([w * torch.ones_like(output[:, 0, ...]) for w in self.w])
        fl = torch.mul(weight_tensor, torch.pow(target - output, self.g) * torch.log(output))
        return torch.mean(-1.0 * fl)

    # def forward(self, output, target):
    #     target_bin = torch.cat([1.0 - target, target], dim=1).float()
    #     output = torch.clamp(output, min=1E-7, max=1.0 - 1E-7)
    #
    #     mask0, mask1  = target_bin[:, 0, ...], target_bin[:, 1, ...] # background mask (is 1 if bg, 0 if fg)
    #     s0, s1 = output[:, 0, ...], output[:, 1, ...] # s0 = 1 - s1
    #
    #     fl_0 = torch.mul(mask0, self.a * torch.pow(1.0 - s0, self.g) * torch.log(s0))
    #     fl_1 = torch.mul(mask1, (1 - self.a) * torch.pow(1.0 - s1, self.g) * torch.log(s1))
    #
    #     return torch.mean(-1.0 * (fl_0 + fl_1))


def soft_crossentropy_with_logits(output, target):
    num_ch = output.size(1)
    out_flat = output.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
    tgt_flat = target.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
    return cross_entropy_with_probs(out_flat, tgt_flat)