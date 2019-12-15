import warnings

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from skimage import measure


def reorient_nifti(nifti, reference=None):
    """
    TODO check

    Reorients axis and orientation of `nifti` to match `reference`.
    If no reference is provided then it is reoriented to canonical orientation (RAS).

    :param nib.Nifti1Image() nifti: nifti to reorient
    :param nib.Nifti1Image() reference: reference nifti for axis and orientation matching.
    :return: the reoriented nifti
    """
    assert isinstance(nifti, nib.Nifti1Image)
    if reference is None:
        return nib.as_closest_canonical(nifti)
    else:
        assert isinstance(reference, nib.Nifti1Image)
        return nifti.as_reoriented(nib.io_orientation(reference.affine))


def normalize_by_range(img, new_range, old_range=None):
    """
    Scales the intensity range of the given array to the new range.
    :param img: the input array
    :param list new_range: two element list with the minimum and maximum values of the output
    :param list old_range: two element list with the minimum and maximum values of the input.
    If not given then the range is computed from the minimum and maximum value of the input.
    :return:
    """
    new_low, new_high = float(new_range[0]), float(new_range[1])

    if old_range is None:
        old_low, old_high = np.min(img), np.max(img)
    else:
        old_low, old_high = float(old_range[0]), float(old_range[1])

    norm_image = (img - old_low) / (old_high - old_low)  # Put between 0 and 1
    norm_image = new_low + ((new_high - new_low) * norm_image)  # Put in new range
    return norm_image


def clip_percentile(img, percentile, ignore_zeros=False):
    """
    Clips image values according to the given percentile
    :param img:
    :param percentile: either int or list,
    :return:
    """
    img_flat = img[np.nonzero(img)] if ignore_zeros else img

    if isinstance(percentile, list):
        low, high = np.percentile(img_flat, percentile)
    else:
        low, high = np.percentile(img_flat, [percentile, 100-percentile])

    return np.clip(img, low, high)


def get_largest_connected_component(segmentation):
    labels = measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def pad_to_size(img, target_size):
    """
    TODO
    :param img:
    :param target_size:
    :return:
    """

    pass

def compute_normalization_statistics(arr, ignore_zeros=False):
    if ignore_zeros:
        arr = np.split(arr, len(arr)) # Put array in list form
        for c in range(len(arr)): # For each list, flatten array and keep nonzeros
            arr[c] = arr[c].ravel()[np.flatnonzero(arr[c])]

    mean = [np.nanmean(channel.astype('float')) for channel in arr]
    std = [np.nanstd(channel.astype('float')) for channel in arr]
    return mean, std

def normalize_by_statistics(arr, mean, std):
    """
    Normalises a numpy array by subtracting the mean and diving by the std

    :param arr: numpy array
    :param mean: list containing the mean value of each channel in array
    :param std: list containing the stdev value of each channel in array
    :return: the normalized array
    """
    assert len(arr) == len(mean) == len(std)
    for channel_idx in range(arr.shape[0]):
        arr[channel_idx] -= mean[channel_idx]
        arr[channel_idx] /= std[channel_idx]
    return arr

def denormalize_by_statistics(arr, mean, std):
    assert arr.shape[0] == len(mean) == len(std)
    for channel_idx in range(arr.shape[0]):
        arr[channel_idx] *= std[channel_idx]
        arr[channel_idx] += mean[channel_idx]
    return arr

def normalize_patch(patch, mean, std):
    warnings.warn('Deprecated: use normalize_by_stats', DeprecationWarning)
    return normalize_by_statistics(patch, mean, std)


def histogram_matching(reference_filepath, input_filepath, output_filepath):
    ref_sitk = sitk.ReadImage(reference_filepath)
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(ref_sitk.GetPixelID())
    mov_sitk = caster.Execute(sitk.ReadImage(input_filepath))

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(15)
    matcher.SetThresholdAtMeanIntensity(True)
    enhanced_mov = matcher.Execute(mov_sitk, ref_sitk)

    print("Saving {}".format(output_filepath))
    sitk.WriteImage(enhanced_mov, output_filepath)


