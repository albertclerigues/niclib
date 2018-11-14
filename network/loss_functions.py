import torch


def NIC_accuracy(y_pred, y_true, class_dim=1):
    """
    from Keras: K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    """
    y_true = y_true.long()
    y_pred_categorical = torch.argmax(y_pred, dim=class_dim)
    return torch.mean(torch.eq(y_true, y_pred_categorical).float())

def NIC_crossentropy(output, y_true, class_axis=1):
    # scale preds so that the class probs of each sample sum to 1
    y_pred = output/torch.sum(output, dim=1, keepdim=True)
    y_true_binary = torch.stack([torch.abs(y_true - 1), y_true], dim=1).float()

    # manual computation of crossentropy
    _epsilon = 1E-7
    y_pred = torch.clamp(y_pred, _epsilon, 1. - _epsilon)

    return torch.mean(-torch.sum(y_true_binary * torch.log(y_pred), dim=class_axis))

def NIC_BCELoss(y_pred, y_true):
    """
    Wrapper for torch BCELoss that adapts the shape and content
    """
    y_true_binary = torch.stack([torch.abs(y_true - 1.0), y_true], dim=1).float()
    _epsilon = 1E-7
    y_pred = torch.clamp(y_pred, min=_epsilon, max=1.0 - _epsilon)
    return torch.nn.BCELoss()(y_pred, y_true_binary)


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

def dice_loss(y_pred, y_true, lesion_class=1):
    """This definition generalize to real valued pred and target vector.
        This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 0.0001
    y_pred = y_pred[:, lesion_class, ...].float()
    y_true = y_true.float()

    # have to use contiguous since they may from a torch.view op
    iflat = y_pred.contiguous().view(-1)
    tflat = y_true.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def dice_dense(y_pred, y_true):
    """
    Computing mean-class Dice similarity.

    :param y_pred: last dimension should have ``num_classes``
    :param y_true: segmentation ground truth (encoded as a binary matrix)
        last dimension should be ``num_classes``
    :return: ``1.0 - mean(Dice similarity per class)``
    """

    y_pred = y_pred.float()
    y_true = torch.stack([torch.abs(y_true - 1), y_true], dim=1).float()

    # computing Dice over the batch and spatial dimensions
    reduce_dims = tuple([0,] + list(range(2, len(y_pred.shape))))

    dice_numerator = 2.0 * torch.sum(y_pred * y_true, dim=reduce_dims)
    dice_denominator = torch.sum(y_pred, dim=reduce_dims) + torch.sum(y_true, dim=reduce_dims)

    epsilon = 0.0001
    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)

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