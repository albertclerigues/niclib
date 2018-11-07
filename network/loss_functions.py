import torch


def jaccard(y_pred, y_true, smooth=100.0):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.githuAdadeltab.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    
    intersection = torch.sum(torch.abs(y_true * y_pred), axis=-1)
    sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth


def dice_loss(y_pred, y_true):
    """This definition generalize to real valued pred and target vector.
        This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # Cast to float and remove healthy class logits for predicted
    y_pred = y_pred[:, 1, :, :].float()
    y_true = y_true.float()

    # have to use contiguous since they may from a torch.view op
    iflat = y_pred.contiguous().view(-1)
    tflat = y_true.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def dice(y_pred, y_true):
    """
    Computing mean-class Dice similarity.
    This function assumes one-hot encoded ground truth (CATEGORICAL :D)

    :param y_pred: last dimension should have ``num_classes``
    :param y: segmentation ground truth (encoded as a binary matrix)
        last dimension should be ``num_classes``
    :param weight_map:
    :return: ``1.0 - mean(Dice similarity per class)``
    """

    # computing Dice over the spatial dimensions
    reduce_axes = list(range(len(y_pred.shape) - 1))
    dice_numerator = 2.0 * torch.sum(y_pred * y, axis=reduce_axes)
    dice_denominator = torch.sum(y_pred, axis=reduce_axes) + torch.sum(y, axis=reduce_axes)

    epsilon_denominator = 0.0001
    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    return 1.0 - torch.mean(dice_score)

def ss(y_pred, y_true, weight_map=None,r=0.05):
    """
    Function to calculate a multiple-ground_truth version of
    the sensitivity-specificity loss defined in "Deep Convolutional
    Encoder Networks for Multiple Sclerosis Lesion Segmentation",
    Brosch et al, MICCAI 2015,
    https://link.springer.com/chapter/10.1007/978-3-319-24574-4_1

    error is the sum of r(specificity part) and (1-r)(sensitivity part)

    :param prediction: the logits
    :param ground_truth: segmentation ground_truth.
    :param r: the 'sensitivity ratio'
        (authors suggest values from 0.01-0.10 will have similar effects)
    :return: the loss
    """

    # chosen region may contain no voxels of a given label. Prevents nans.
    eps = 1e-5

    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    sq_error = torch.pow(y_true_f - y_pred_f, 2)

    spec_part = torch.sum(sq_error * y_true_f) / (torch.sum(y_true_f) + eps)
    sens_part =  torch.sum(sq_error * (1 - y_true_f)) / (torch.sum(1 - y_true_f) + eps)

    return r*spec_part + (1.0 - r)*sens_part