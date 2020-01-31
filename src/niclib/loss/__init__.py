import torch
import torch.nn.functional as F
from torch import nn

# Import losses
from .lovasz_losses import lovasz_softmax_flat


#########################################################################
### Loss Wrappers
class LossWrapper(nn.Module):
    """LossWrapper creates a nn.Module object that allows to preprocessing of the tensors and/or postprocess the loss output.

    :param callable loss_fn: The loss function with input arguments (output, target)
    :param callable preprocess_fn: Preprocess function with signature ``preprocess_fn(output, target)`` that returns
        a tuple ``(output_preprocessed, target_preprocessed)``.
    :param callable postprocess_fn: Postprocess function with signature ``postprocess_fn(loss_output)`` that returns
        a single tensor ``loss_output_postprocessed``.

    :Example:

    >>> # Crossentropy loss needs a tensor of type long as target, but ours is of type float!
    >>> my_loss = LossWrapper(nn.CrossEntropyLoss(), preprocess_fn=lambda output, target: (output, target.long()))
    """
    def __init__(self, loss_fn, preprocess_fn=None, postprocess_fn=None):
        super().__init__()
        self.loss_fn, self.pre_fn, self.post_fn = loss_fn, preprocess_fn, postprocess_fn

    def forward(self, output, target):
        if self.pre_fn is not None:
            output, target = self.pre_fn(output, target)
        loss_output = self.loss_fn(output, target)
        if self.post_fn is not None:
            loss_output = self.post_fn(loss_output)
        return loss_output


class Hard2SoftLoss(nn.Module):
    """Torch loss wrapper that allows the use of probabilistic targets in segmentation losses that use targets
        with hard integer labels.

    :param loss_function: the function MUST accept keyword argument ``reduction='none'`` as in
        ``loss_function(output, target, reduction='none')``. If input tensors are of size (BS, CH, *)
        the shape returned with ``reduction='none'`` must be (BS, *)
    :param weights: (optional) iterable of length equal to the number of channels (dim=1)
    :param str reduction: one of ``mean``, ``sum`` or ``none`` (default: 'mean')
    """

    def __init__(self, loss_function, weights=None, reduction='mean'):
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


def soft_crossentropy_with_logits(output, target):
    from snorkel.classification import cross_entropy_with_probs

    num_ch = output.size(1)
    out_flat = output.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
    tgt_flat = target.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
    return cross_entropy_with_probs(out_flat, tgt_flat)


# class LovaszSoftmaxLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, output, target):
#         """TODO make version where c classes are already done"""
#         num_ch = output.size(1)
#         # Flatten the tensors to use flattened version
#         out_flat = output.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
#         tgt_flat = target.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
#         return lovasz_softmax_flat(out_flat, torch.argmax(tgt_flat, dim=1))


class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation [Lin2017]_.

    This loss is a difficulty weighted version of the crossentropy loss where more accurate predictions
    have a diminished loss output.

    :param gamma: the *focusing* parameter (where :math:`\\gamma > 0`). A higher gamma will give less weight to confidently
        predicted samples.
    :param list weights: channel-wise weights to multiply before reducing the output.

    .. rubric:: Usage

    :param output: tensor with dimensions :math:`(\\text{BS}, \\text{CH}, \\ast)` containing the output **logits** of the network.
        :math:`\\text{BS}` is the batch size, :math:`\\text{CH}` the number of channels and :math:`\\ast` can be any
        number of additional dimensions with any size.
    :param target: tensor with dimensions :math:`(\\text{BS}, \\ast)` (of type long) containing integer label targets.
    :param reduce: one of ``'none'``, ``'mean'``, ``'sum'`` (default: ``'mean'``)

    .. rubric:: References

    .. [Lin2017] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017. (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self, gamma=2.0, weights=None, reduce='mean'):
        super().__init__()
        assert gamma >= 0.0
        self.g = gamma
        self.w = weights

        self.reduce_fns = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x : x}
        assert reduce in self.reduce_fns.keys()
        self.reduce = self.reduce_fns[reduce]

    def forward(self, output, target, reduce=None):
        pt_mask = torch.stack([target == l for l in torch.arange(output.size(1))], dim=1).float()
        pt_softmax = torch.sum(nn.functional.softmax(output, dim=1) * pt_mask, dim=1)
        pt_log_softmax = torch.sum(torch.nn.functional.log_softmax(output, dim=1) * pt_mask, dim=1)
        fl = -1.0 * torch.pow(torch.sub(1.0, pt_softmax), self.g) * pt_log_softmax

        if self.w is not None:
            fl *= torch.stack(
                [self.w[n] *  pt_mask[:, l, ...] for n, l in enumerate(torch.arange(pt_mask.size(1)))], dim=1)

        if reduce is not None:
            assert reduce in self.reduce_fns.keys()
            self.reduce = self.reduce_fns[reduce]
        return self.reduce(fl)


#
#
# class BinaryFocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, a=0.25):
#         """ Binary focal loss
#         :param gamma:
#         :param alpha: weight for class 0 (background)
#
#         Usage:
#         :param output:
#         :param target:
#         """
#         super().__init__()
#         self.g = gamma
#         self.a = a
#
#     def forward(self, output, target):
#         target_bin = torch.cat([1.0 - target, target], dim=1).float()
#         output = torch.clamp(output, min=1E-7, max=1.0 - 1E-7)
#
#         mask0, mask1  = target_bin[:, 0, ...], target_bin[:, 1, ...] # background mask (is 1 if bg, 0 if fg)
#         s0, s1 = output[:, 0, ...], output[:, 1, ...] # s0 = 1 - s1
#
#         fl_0 = torch.mul(mask0, self.a * torch.pow(1.0 - s0, self.g) * torch.log(s0))
#         fl_1 = torch.mul(mask1, (1 - self.a) * torch.pow(1.0 - s1, self.g) * torch.log(s1))
#
#         return torch.mean(-1.0 * (fl_0 + fl_1))


