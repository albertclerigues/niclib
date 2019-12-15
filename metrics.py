import numpy as np

from .medpy_hausdorff import hd as haussdorf_dist


# TODO use torch versions to also use GPU if available

def compute_metrics(outputs, targets, metrics, ids=None):
    """
    Computes evaluation metrics for the given images.

    :param outputs: list of output images
    :param targets: list of gold standard images
    :param metrics: dictionary containing the functions for metric computation
    :param ids: (Optional) list of image pair identifiers
    :return: list of dictionaries containing the metrics for each given sample pair
    """
    assert len(outputs) == len(targets)
    assert isinstance(metrics, dict)

    all_metrics = []
    for n, (output, target) in enumerate(zip(outputs, targets)):
        case_metrics = {}
        case_metrics.update({'id': str(n) if ids is None else ids[n]})

        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(output, target)
            case_metrics.update({metric_name: metric_value} if not isinstance(metric_value, dict) else metric_value)
        all_metrics.append(case_metrics)
    return all_metrics

def compute_confusion_matrix(output, target):
    """Returns tuple (true_pos, true_neg, false_pos, false_neg)"""
    assert output.size(1) == target.size(1) == 2
    assert target.size == output.size

    true_pos = np.sum(np.logical_and(target, output))
    true_neg = np.sum(np.logical_and(target == 0, output == 0))

    false_pos = np.sum(np.logical_and(target == 0, output))
    false_neg = np.sum(np.logical_and(target, output == 0))

    return true_pos, true_neg, false_pos, false_neg


def segmentation_metrics(output, target):
    seg_metrics = {}
    eps = np.finfo(np.float32).eps

    # Compute confusion matrix parameters
    tp, tn, fp, fn = compute_confusion_matrix(output, target)
    # Sensitivity and specificity
    seg_metrics['sens'] = tp / (tp + fn + eps) # Correct % of the real lesion
    seg_metrics['spec'] = tn / (tn + fp + eps) # Correct % of the healthy area identified
    # Predictive values
    seg_metrics['ppv'] = tp / (tp + fp + eps) # Of all lesion voxels, % of really lesion
    seg_metrics['npv'] = tn / (tn + fn + eps)  # Of all lesion voxels, % of really healthy

    return seg_metrics

def haussdorf_distance(output: np.ndarray, target: np.ndarray):
    """Haussdorf distance"""
    try:
        return haussdorf_dist(output, target, connectivity=3)
    except Exception:
        return np.nan

def dsc(output, target, background_label=0):
    """Dice Similarity Coefficient

    :param output: class labels of segmentation
    :param target: class labels of ground truth
    :param background_label:

    :Example:

    >>> dsc([0, 1, 1], [1, 1, 0])
    0.5
    >>> dsc([0, 1, 2, 2], [0, 1, 2, 3])
    0.6666666666666666
    """
    output, target = np.asanyarray(output), np.asanyarray(target)
    assert np.array_equal(output.shape, target.shape)
    output_mask, target_mask = output != background_label, target != background_label # Background/Foreground masking
    intersection = np.sum(np.logical_and(np.equal(output, target), np.logical_or(output_mask, target_mask)))
    denominator = np.sum(output_mask) + np.sum(target_mask)
    return 2.0 * intersection / denominator if denominator > 0 else 0

def mse(output, target):
    """Mean squared error"""
    return np.mean(np.power(output - target, 2))

def mae(output, target, mask_zeros=False):
    """Mean Absolute Error
    :Example:

    >>> mae([0.0, 0.0], [1.0, 1.0])
    """
    output, target = np.asanyarray(output), np.asanyarray(target)
    assert np.array_equal(output.shape, target.shape)

    output, target = output.flatten(), target.flatten()
    if mask_zeros:
        nonzero_idxs = np.nonzero(target)
        output, target = output[nonzero_idxs], target[nonzero_idxs]
    return np.mean(np.abs(output - target))

def ssim(output, target):
    """Structural Similarity Index"""
    A = np.abs(output / output.max()).astype(float)
    B = np.abs(target / target.max()).astype(float)
    intersect = np.multiply(A != 0, B != 0)
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
    num = (2*ua*ub + c1) * (2*oab + c2)
    den = (ua**2 + ub**2 + c1) * (oa + ob + c2)
    return num / den

def psnr(output, target, img_range=None):
    """Peak signal to noise ratio"""

    if img_range is None:
        img_range  = np.max([target, output]) - np.min([target, output])
    return 20.0 * np.log10(float(img_range)/mse(output, target))

