import torch
import torch.nn.functional as F
from snorkel.classification import cross_entropy_with_probs
from torch import nn

# Import losses
from .lovasz_losses import lovasz_softmax_flat


#########################################################################
### Loss Wrappers
class Hard2SoftLoss(nn.Module):
    """Torch loss wrapper that allows the use of probabilistic targets in segmentation losses that use targets with hard integer label_img.

    :param loss_function: the function MUST accept keyword argument ``reduction='none'`` as in ``loss_function(output, target, reduction='none')``. If input tensors are of size (BS, CH, *) the shape returned with ``reduction='none'`` must be (BS, *)
    :param weights: (optional) iterable of length equal to the number of channels (dim=1)
    :param reduction: one of ``mean``, ``sum`` or ``none`` (default: 'mean')
    """

    def __init__(self, loss_function, weights=None, reduction : str ='mean'):
        super().__init__()
        self.loss_fn = loss_function
        self.weights = weights
        self.reduction = reduction

    def forward(self, output, target):
        """
        TODO test
        """
        batch_size, num_classes = output.size()[0], output.size()[1]
        # Note that t.new_zeros, t.new_full just puts tensor on same device as t
        cum_losses = output.new_zeros(output.size()[0:1] + output.size()[2:])
        for y in range(num_classes):
            target_temp = output.new_full(output.size()[0:1] + output.size()[2:], y, dtype=torch.long)
            y_loss = self.loss_fn(output, target_temp, reduction='none')
            if self.weights is not None:
                y_loss = y_loss * self.weights[y]
            cum_losses += target[:, y, ...].float() * y_loss

        if self.reduction == 'none':
            return cum_losses
        elif self.reduction == 'mean':
            return cum_losses.mean()
        elif self.reduction == 'sum':
            return cum_losses.sum()
        else:
            raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


class LossWrapper(nn.Module):
    def __init__(self, loss_function, preprocess_function=None, postprocess_function=None):
        super().__init__()
        self.loss_fn, self.pre_fn, self.post_fn = loss_function, preprocess_function, postprocess_function

    def forward(self, output, target):
        if self.pre_fn is not None:
            output, target = self.pre_fn(output, target)
        loss_output = self.loss_fn(output, target)
        if self.post_fn is not None:
            loss_output = self.post_fn(loss_output)
        return loss_output


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """TODO make version where c classes are already done"""
        num_ch = output.size(1)
        # Flatten the tensors to use flattened version
        out_flat = output.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
        tgt_flat = target.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
        return lovasz_softmax_flat(out_flat, torch.argmax(tgt_flat, dim=1))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=None, with_logits=False):
        """Focal loss

        TODO implement

        :param gamma:
        :param list weights: channel-wise loss weights

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

        log_output = F.log_softmax(pt, dim=1) if self.with_logits else torch.log(pt)
        fl = -1.0 * torch.pow(pt, self.g) * log_output

        if self.w is not None: # Apply channel-wise weighting
            fl = torch.mul(torch.stack([torch.full_like(output[:, 0, ...], w) for w in self.w], dim=1), fl)
        return torch.mean(fl)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, a=0.25):
        """ Binary focal loss
        :param gamma:
        :param alpha: weight for class 0 (background)

        Usage:
        :param output:
        :param target:
        """
        super().__init__()
        self.g = gamma
        self.a = a

    def forward(self, output, target):
        target_bin = torch.cat([1.0 - target, target], dim=1).float()
        output = torch.clamp(output, min=1E-7, max=1.0 - 1E-7)

        mask0, mask1  = target_bin[:, 0, ...], target_bin[:, 1, ...] # background mask (is 1 if bg, 0 if fg)
        s0, s1 = output[:, 0, ...], output[:, 1, ...] # s0 = 1 - s1

        fl_0 = torch.mul(mask0, self.a * torch.pow(1.0 - s0, self.g) * torch.log(s0))
        fl_1 = torch.mul(mask1, (1 - self.a) * torch.pow(1.0 - s1, self.g) * torch.log(s1))

        return torch.mean(-1.0 * (fl_0 + fl_1))


def soft_crossentropy_with_logits(output, target):
    num_ch = output.size(1)
    out_flat = output.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
    tgt_flat = target.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
    return cross_entropy_with_probs(out_flat, tgt_flat)