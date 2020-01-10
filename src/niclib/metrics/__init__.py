import warnings
import torch
import numpy as np
from .medpy_hausdorff import hd as haussdorf_dist

import copy

def _convert_to_scalar(result, was_numpy):
    if was_numpy:
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy().item()
    return result

def _convert_to_torch(output, target):
    was_numpy = False
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
        was_numpy = True
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
        was_numpy = True
    return output, target, was_numpy


def compute_metrics(outputs, targets, metrics, ids=None):
    """
    Computes the evaluation metrics for the given list of output and target images.

    :param outputs: list of output images
    :param targets: list of gold standard images
    :param metrics: dictionary containing each metric as {metric_name: metric_function}
    :param ids: (Optional) list of image identifiers
    :return: list of dictionaries containing the metrics for each given sample
    """
    assert len(outputs) == len(targets)
    assert isinstance(metrics, dict)

    all_metrics = []
    for n, (output, target) in enumerate(zip(outputs, targets)):
        case_metrics = {'id': str(n) if ids is None else ids[n]}
        for metric_name, metric_func in metrics.items():
            case_metrics.update({metric_name: metric_func(output, target)})
        all_metrics.append(case_metrics)
    return all_metrics

def _true_positives(output, target):
    output, target, was_numpy = _convert_to_torch(output, target)
    result = torch.sum(target.bool() * output.bool())
    return _convert_to_scalar(result, was_numpy)

def _true_negatives(output, target):
    output, target, was_numpy = _convert_to_torch(output, target)
    result = torch.sum((target == 0).bool() * (output == 0).bool())
    return _convert_to_scalar(result, was_numpy)

def _false_positives(output, target):
    output, target, was_numpy = _convert_to_torch(output, target)
    result = torch.sum((target == 0).bool() * output.bool())
    return _convert_to_scalar(result, was_numpy)


def _false_negatives(output, target):
    output, target, was_numpy = _convert_to_torch(output, target)
    result = torch.sum(target.bool() * (output == 0).bool())
    return _convert_to_scalar(result, was_numpy)

def true_positive_fraction(output, target):
    """True positive fraction of binary segmentation."""
    output, target, was_numpy = _convert_to_torch(output, target)
    result = _true_positives(output, target) / (torch.sum(output) + torch.sum(target))
    return _convert_to_scalar(result, was_numpy)

def true_negative_fraction(output, target):
    """True negative fraction of binary segmentation."""
    output, target, was_numpy = _convert_to_torch(output, target)
    result = _true_negatives(output, target) / (torch.sum(output) + torch.sum(target))
    return _convert_to_scalar(result, was_numpy)

def false_negative_fraction(output, target):
    """False negative fraction of binary segmentation."""
    output, target, was_numpy = _convert_to_torch(output, target)
    result = _false_negatives(output, target) / (torch.sum(output) + torch.sum(target))
    return _convert_to_scalar(result, was_numpy)

def false_positive_fraction(output, target):
    """False positive fraction of binary segmentation."""
    output, target, was_numpy = _convert_to_torch(output, target)
    result = _false_positives(output, target) / (torch.sum(output) + torch.sum(target))
    return _convert_to_scalar(result, was_numpy)

def sensitivity(output, target):
    """Sensitivity metric for binary segmentation."""
    output, target, was_numpy = _convert_to_torch(output, target)
    tp, fn = _true_positives(output, target), _false_negatives(output, target)
    result = tp / (tp + fn + 1e-7)
    return _convert_to_scalar(result, was_numpy)

def specificity(output, target):
    """Specificity metric for binary segmentation."""
    output, target, was_numpy = _convert_to_torch(output, target)
    tn, fp = _true_negatives(output, target), _false_positives(output, target)
    result = tn / (tn + fp + 1e-7)
    return _convert_to_scalar(result, was_numpy)


def positive_predictive_value(output, target):
    output, target, was_numpy = _convert_to_torch(output, target)
    """Positive predictive value for binary segmentation."""
    tp, fp = _true_positives(output, target), _false_positives(output, target)
    result = tp / (tp + fp + 1e-7)
    return _convert_to_scalar(result, was_numpy)


def negative_predictive_value(output, target):
    """Negative predictive value for binary segmentation"""
    output, target, was_numpy = _convert_to_torch(output, target)
    tn, fn = _true_negatives(output, target), _false_negatives(output, target)
    result = tn / (tn + fn + 1e-7)
    return _convert_to_scalar(result, was_numpy)


def haussdorf_distance(output, target):
    """Haussdorf distance"""
    original_tensor = copy.copy(output)
    output, target, was_numpy = _convert_to_torch(output, target) # TODO do conversion smarter (to many conversions)

    try:
        result = haussdorf_dist(output.numpy(), target.numpy(), connectivity=3)
    except Exception:
        warnings.warn('Hausdorff distance error, returning NaN', RuntimeWarning)
        result = np.nan

    if not was_numpy:
        result = original_tensor.new(result)
    return result


def dsc(output, target, background_label=0):
    """Dice Similarity Coefficient"""
    output, target, was_numpy = _convert_to_torch(output, target)
    assert output.size() == target.size()

    output_mask = (output != background_label).bool()
    target_mask = (target != background_label).bool()

    intersection = torch.sum((output == target).bool() * (output_mask + target_mask))
    denominator = torch.sum(output_mask) + torch.sum(target_mask)
    result = 2.0 * intersection / denominator if denominator > 0 else 0

    return _convert_to_scalar(result, was_numpy)

def accuracy(output, target, class_dim=1):
    """Accuracy defined as: mean(argmax(output) == argmax(target))"""
    return torch.mean(torch.eq(torch.argmax(output, dim=class_dim), torch.argmax(target, dim=class_dim)).float())


def mse(output, target, ignore_zeros=False):
    """Mean squared error"""
    output, target, was_numpy = _convert_to_torch(output, target)
    assert output.size() == target.size()

    output, target = torch.flatten(output), torch.flatten(target)  # TODO change to method that does not copy images?
    if ignore_zeros:
        nonzero_idxs = torch.nonzero(target)
        output, target = output[nonzero_idxs], target[nonzero_idxs]
    result = torch.mean(torch.pow(output - target, 2))

    return _convert_to_scalar(result, was_numpy)



def mae(output, target, ignore_zeros=False):
    """Mean Absolute Error"""
    output, target, was_numpy = _convert_to_torch(output, target)
    assert output.size() == target.size()

    output, target = torch.flatten(output), torch.flatten(target)  # TODO change to method that does not copy images?
    if ignore_zeros:
        nonzero_idxs = torch.nonzero(target)
        output, target = output[nonzero_idxs], target[nonzero_idxs]
    result = torch.mean(torch.abs(output - target))
    return _convert_to_scalar(result, was_numpy)


def ssim(output, target, background_value=0):
    """Structural Similarity Index."""
    output, target, was_numpy = _convert_to_torch(output, target)

    A = torch.abs(output / output.max()).float()
    B = torch.abs(target / target.max()).float()

    intersect = (A != background_value).bool() * (B != background_value).bool()
    ua = A[intersect].mean()
    ub = B[intersect].mean()
    oa = torch.pow(A[intersect].std(), 2.0)
    ob = torch.pow(B[intersect].std(), 2.0)
    oab = torch.sum(torch.mul(A[intersect] - ua, B[intersect] - ub)) / (torch.sum(intersect) - 1)
    k1 = 0.01
    k2 = 0.03
    L = 1
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    num = (2 * ua * ub + c1) * (2 * oab + c2)
    den = (ua ** 2 + ub ** 2 + c1) * (oa + ob + c2)
    result = num / den
    return _convert_to_scalar(result, was_numpy)


def psnr(output, target, img_range=None):
    """Peak signal to noise ratio"""
    output, target, was_numpy = _convert_to_torch(output, target)
    if img_range is None:
        img_range = torch.max(target.max(), output.max()) - torch.min(target.min(), output.min())
    result = 20.0 * torch.log10(float(img_range) / mse(output, target))
    return _convert_to_scalar(result, was_numpy)
