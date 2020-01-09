import warnings
import numpy as np
from .medpy_hausdorff import hd as haussdorf_dist

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
    return np.sum(np.logical_and(target, output))


def _true_negatives(output, target):
    return np.sum(np.logical_and(target == 0, output == 0))


def _false_positives(output, target):
    return np.sum(np.logical_and(target == 0, output))


def _false_negatives(output, target):
    return np.sum(np.logical_and(target, output == 0))


def true_positive_fraction(output, target):
    """True positive fraction of binary segmentation."""
    return _true_positives(output, target) / (np.sum(output) + np.sum(target))

def true_negative_fraction(output, target):
    """True negative fraction of binary segmentation."""
    return _true_negatives(output, target) / (np.sum(output) + np.sum(target))

def false_negative_fraction(output, target):
    """False negative fraction of binary segmentation."""
    return _false_negatives(output, target) / (np.sum(output) + np.sum(target))

def false_positive_fraction(output, target):
    """False positive fraction of binary segmentation."""
    return _false_positives(output, target) / (np.sum(output) + np.sum(target))

def sensitivity(output, target):
    """Sensitivity metric for binary segmentation."""
    tp, fn = _true_positives(output, target), _false_negatives(output, target)
    return tp / (tp + fn + 1e-7)

def specificity(output, target):
    """Specificity metric for binary segmentation."""
    tn, fp = _true_negatives(output, target), _false_positives(output, target)
    return tn / (tn + fp + 1e-7)


def positive_predictive_value(output, target):
    """Positive predictive value for binary segmentation."""
    tp, fp = _true_positives(output, target), _false_positives(output, target)
    return tp / (tp + fp + 1e-7)


def negative_predictive_value(output, target):
    """Negative predictive value for binary segmentation"""
    tn, fn = _true_negatives(output, target), _false_negatives(output, target)
    return tn / (tn + fn + 1e-7)


def haussdorf_distance(output, target):
    """Haussdorf distance"""
    try:
        return haussdorf_dist(output, target, connectivity=3)
    except Exception:
        warnings.warn('Hausdorff distance error, returning NaN', RuntimeWarning)
        return np.nan


def dsc(output, target, background_label=0):
    """Dice Similarity Coefficient"""
    output, target = np.asanyarray(output), np.asanyarray(target)
    assert np.array_equal(output.shape, target.shape)
    output_mask, target_mask = output != background_label, target != background_label  # Background/Foreground masking
    intersection = np.sum(np.logical_and(np.equal(output, target), np.logical_or(output_mask, target_mask)))
    denominator = np.sum(output_mask) + np.sum(target_mask)
    return 2.0 * intersection / denominator if denominator > 0 else 0


def mse(output, target, ignore_zeros=False):
    """Mean squared error"""
    output, target = np.asanyarray(output), np.asanyarray(target)
    assert np.array_equal(output.shape, target.shape)
    output, target = output.flatten(), target.flatten()  # TODO change to method that does not copy images
    if ignore_zeros:
        nonzero_idxs = np.nonzero(target)
        output, target = output[nonzero_idxs], target[nonzero_idxs]
    return np.mean(np.power(output - target, 2))


def mae(output, target, ignore_zeros=False):
    """Mean Absolute Error"""
    output, target = np.asanyarray(output), np.asanyarray(target)
    assert np.array_equal(output.shape, target.shape)
    output, target = output.flatten(), target.flatten()  # TODO change to method that does not copy images
    if ignore_zeros:
        nonzero_idxs = np.nonzero(target)
        output, target = output[nonzero_idxs], target[nonzero_idxs]
    return np.mean(np.abs(output - target))


def ssim(output, target, background_value=0):
    """Structural Similarity Index."""
    A = np.abs(output / output.max()).astype(float)
    B = np.abs(target / target.max()).astype(float)
    intersect = np.multiply(A != background_value, B != background_value)
    ua = A[intersect].mean()
    ub = B[intersect].mean()
    oa = A[intersect].std() ** 2
    ob = B[intersect].std() ** 2
    oab = np.sum(np.multiply(A[intersect] - ua, B[intersect] - ub)) / (np.sum(intersect) - 1)
    k1 = 0.01
    k2 = 0.03
    L = 1
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    num = (2 * ua * ub + c1) * (2 * oab + c2)
    den = (ua ** 2 + ub ** 2 + c1) * (oa + ob + c2)
    return num / den


def psnr(output, target, img_range=None):
    """Peak signal to noise ratio"""
    if img_range is None:
        img_range = np.max([target, output]) - np.min([target, output])
    return 20.0 * np.log10(float(img_range) / mse(output, target))
