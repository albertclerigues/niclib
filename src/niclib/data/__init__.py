import warnings

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from skimage import measure


def adjust_range(img, new_range, old_range=None):
    """Scales the intensity range of the given array to the new range.

    :param img: the input array.
    :param new_range: list or tuple with the minimum and maximum values of the output.
    :param old_range: (optional) list or tuple with the range of the input.
        If not given then the range is computed from the minimum and maximum value of the input.
    :return: the image with intensities in the new range.

    :Example:

    >>> adjust_range([3, 4, 5], new_range=[0, 10])
    array([ 0., 5.0, 10.])
    >>> adjust_range([3, 4, 5], new_range=[0, 10], old_range=[0, 100])
    array([ 0.3, 0.4, 0.5])
    """
    assert new_range[0] < new_range[1], 'new_range is not correctly defined'

    new_low, new_high = float(new_range[0]), float(new_range[1])
    if old_range is None:
        old_low, old_high = np.min(img), np.max(img)
        if old_low == old_high:
            raise ValueError('Given array is of constant value, range cannot be adjusted!')
    else:
        old_low, old_high = float(old_range[0]), float(old_range[1])

    if not isinstance(img, np.ndarray):
        img = np.asanyarray(img)

    norm_image = (img - old_low) / (old_high - old_low)  # Put between 0 and 1
    norm_image = new_low + ((new_high - new_low) * norm_image)  # Put in new range
    return norm_image


def clip_percentile(img, percentile, ignore_zeros=False):
    """Clips image values according to the given percentile

    :param img:
    :param percentile: list (low_percentile, high_percentile) where percentile is a number from 0 to 100.
    :param ignore_zeros: (optional) if True, ignores the zero values in the computation of the percentile.
    :return: the clipped image

    :Example:

    >>> clip_percentile([-1000, -100, -10, 0, 10, 100, 1000], [10, 90])
    array([-460., -100.,  -10.,    0.,   10.,  100.,  460.])
    >>> clip_percentile([-1000, -100, -10, 0, 10, 100, 1000], [25, 75])
    array([-55., -55., -10.,   0.,  10.,  55.,  55.])
    """
    if not isinstance(img, np.ndarray):
        img = np.asanyarray(img)

    img_flat = img[np.nonzero(img)] if ignore_zeros else img
    low, high = np.percentile(img_flat, percentile)
    return np.clip(img, low, high)


def get_largest_connected_component(segmentation):
    """Returns only the largest connected component of a binary segmentation."""
    labels = measure.label(segmentation) # Get connected components
    if labels.max() != 0: # assume at least 1 CC
        warnings.warn(UserWarning, 'Getting largest connected component of empty segmentation')
        return segmentation
    return labels == np.argmax(np.bincount(labels.flat)[1:])+1


def pad_to_shape(image, new_shape, numpy_pad_options=None):
    """Pads an image to match the given shape where dimensions are padded equally before and after (so that the
    original image is at the center of the padded image).

    :param image: image to pad
    :param tuple new_shape: target shape of padded image. If given shape has one less dimension than the image,
        it is assumed that the first image dimension is the channel dimension and shouldn't be padded.
    :param dict numpy_pad_options: (optional) a dictionary with ``**kwargs`` for the numpy.pad function.
        It defaults to ``{'mode': constant, 'constant_values': 0}``.
    :return: a tuple (img_padded, unpad_slice) so that ``img_padded[unpad_slice] == img``

    :Example:

    >>> img_padded, _ = pad_to_shape(np.ones((2,2)), (4,4))
    [[0. 0. 0. 0.]
     [0. 1. 1. 0.]
     [0. 1. 1. 0.]
     [0. 0. 0. 0.]]
    >>> img_padded, unpad_slice = pad_to_shape(np.ones((3,3)), (6,4))
    >>> print(img_padded)
    [[0. 0. 0. 0.]
     [1. 1. 1. 0.]
     [1. 1. 1. 0.]
     [1. 1. 1. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    >>> print(img_padded[unpad_slice])
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    """

    if len(image.shape) > len(new_shape): # Assume image with channel dimension that shouldn't be padded
        shape = (image.shape[0],) + new_shape
    assert len(image.shape) == len(new_shape), 'len({}) != len({})'.format(image.shape, new_shape)
    assert all([img_dim <= tgt_dim for img_dim, tgt_dim in zip(image.shape, new_shape)])

    remaining_shape = np.subtract(new_shape, image.shape)
    pad_widths = [(int(np.floor(rdim / 2.0)), int(np.ceil(rdim / 2.0))) for rdim in remaining_shape]
    numpy_pad_options = {} if numpy_pad_options is None else numpy_pad_options
    img_padded = np.pad(image, pad_widths, **numpy_pad_options)

    unpad_slice = tuple(slice(pw_dim[0], pw_dim[0] + img_dim) for pw_dim, img_dim in zip(pad_widths, image.shape))
    return img_padded, unpad_slice


def crop_borders(image, background_value=0):
    """Crops the background borders of an image. If any given channel is all background, it will also be cropped!

    :param image: the image to crop.
    :param background_value: value of the background that will be cropped.
    :return: The image with background borders cropped.
    """

    if not isinstance(image, np.ndarray):
        image = np.asanyarray(image)
    foreground = (image != background_value)

    crop_slice = []
    for dim_idx in range(image.ndim):
        # Compact array to a single axis to make np.argwhere much more efficient
        compact_axis = tuple(ax for ax in range(image.ndim) if ax != dim_idx)
        foreground_indxs = np.argwhere(np.max(foreground, axis=compact_axis) == True)

        # Find the dimensions lower and upper foreground indices
        crop_slice.append(slice(np.min(foreground_indxs), np.max(foreground_indxs) + 1))
    return image[tuple(crop_slice)]



def histogram_matching(reference_filepath, input_filepath, output_filepath, hist_levels=256, match_points=15, mean_thresh=True):
    """Performs MRI histogram matching [#f1]_ using Simple ITK.

    .. rubric:: Footnotes

    .. [#f1] Laszlo G. Nyul, Jayaram K. Udupa, and Xuan Zhang, "New Variants of a Method of MRI Scale Standardization", IEEE Transactions on Medical Imaging, 19(2):143-150, 2000.

    :param reference_filepath: Filepath of nifti image for reference histogram.
    :param input_filepath: Filepath of nifti image to transform.
    :param output_filepath: Filepath of output image (input with histogram matched to reference).
    """
    assert all([fp.endswith('.nii') or fp.endswith('.nii.gz')
                for fp in {reference_filepath, input_filepath, output_filepath}])

    ref_sitk = sitk.ReadImage(reference_filepath)
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(ref_sitk.GetPixelID())
    mov_sitk = caster.Execute(sitk.ReadImage(input_filepath))

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(hist_levels)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.SetThresholdAtMeanIntensity(mean_thresh)
    enhanced_mov = matcher.Execute(mov_sitk, ref_sitk)

    #print("Saving {}".format(output_filepath))
    sitk.WriteImage(enhanced_mov, output_filepath)


