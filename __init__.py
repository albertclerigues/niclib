import torch
import numpy as np
import os

import datetime
import os
import sys
import time
import warnings

import csv
import math

import nibabel as nib

################################################################
# GPU management

device = torch.device('cuda')  # Module singleton (same instance for everyone)

def set_device(new_device):
    global device
    device = torch.device(new_device)

################################################################
# Reproducibility stuff
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################################################
##### I/O Utils
#from docutils.nodes import classifier

def touch_dir(dir_path):
    """
    Ensures that path exists and creates the directory if necessary
    """

    # Check parent directory exists
    parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
    if not os.path.isdir(parent_dir):
        touch_dir(parent_dir)

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def get_filename(filepath, ext=True):
    """
    Returns the filename of the file pointed by `filepath`.
    :param str filepath: The full path
    :param bool ext: if False removes the file extension before returning
    :return str: The filename
    """
    filename = os.path.basename(filepath)
    if not ext:
        filename = filename.split('.')[0]
    return filename

def remove_extension(filepath):
    """
    Removes the extension of the file pointed by filepath
    """
    paths = filepath.split('/')
    filename_noext = paths[-1].split('.')[0]
    filepath_noext = '/'.join(paths[:-1] + [filename_noext])
    return filepath_noext





def save_nifti(filepath, image, reference=None, reorient=None, dtype=None):
    """
    Saves the given array as a Nifti1Image.

    :param str filepath: filename where the nifti will be saved
    :param numpy.ndarray image: the volume to save in a nifti image
    :param nibabel.Nifti1Image reference: reference nifti from where to take the affine transform and header
    """
    if dtype is not None:
        image = image.astype(dtype)

    if reference is None:
        nifti = nib.Nifti1Image(image, np.eye(4))
    elif isinstance(reference, dict):
        nifti = nib.Nifti1Image(image, reference['affine'], reference['header'])
    else:
        nifti = nib.Nifti1Image(image, reference.affine, reference.header)

    print("Saving nifti: {}".format(filepath))
    nifti.to_filename(filepath)


def save_to_csv(filepath, dict_list):
    assert isinstance(dict_list, list) and all([isinstance(d, dict) for d in dict_list])
    with open(filepath, mode='w') as f:
        csv_writer = csv.DictWriter(f, dict_list[0].keys(), restval='', extrasaction='raise', dialect='unix')
        csv_writer.writeheader()
        csv_writer.writerows(dict_list)

def load_from_csv(filepath):
    with open(filepath, mode='r') as f:
        csv_reader = csv.DictReader(f, restval='', dialect='unix')
        return [row for row in csv_reader]

def append_line_to_file(filepath, line):
    with open(filepath, mode='a') as f:
        f.write('\n' + line)

################################################################
############ Data management utils

def split_list(l, fraction=0.2, indexes=None):
    """
    Splits a given list into two sub-lists according to a fraction.

    :param list l: The list to split
    :param fraction: fraction of samples for list a
    :param indexes: indexes of list for a
    :return: a tuple (a, b) with the training and validation sets
    """
    assert 0.0 < fraction < 1.0

    if indexes is None:
        start_idx, stop_idx = 0, math.ceil(len(l) * fraction)
        list_a = l[:start_idx] + l[stop_idx:]
        list_b = l[start_idx:stop_idx]
    else:
        list_a = [a for n, a in enumerate(l) if n in indexes]
        list_b = [b for n, b in enumerate(l) if n not in indexes]

    return list_a, list_b


def clamp_list(l, min_len=None, max_len=None):
    """
    Clamps a given list so that its length is between min_len and max_len elements.
    To shorten a list elements are removed at regular intervals.
    To lengthen a list elements are repeated to reach desired length.

    :param l: the list
    :param min_len: the minimum desired length
    :param max_len: the maximum desired length
    :return: the clamped list
    """

    if min_len is not None and len(l) < min_len: # under specified minimum
        resampling_idxs = list(np.mod(range(min_len), len(l)).astype(int)) # Oversampling of images
        return [l[i] for i in resampling_idxs]

    if max_len is not None and len(l) > max_len: # over specified maximum
        resampling_idxs = np.arange(start=0.0, stop=float(len(l)) - 1.0, step=len(l) / float(max_len)).astype(int)
        return [l[i] for i in resampling_idxs]

    return l  # len of l was already within desired range

def moving_average(l, n):
    """
    Performs a moving average of window n of the list contents
    :param l: the list
    :param n: the size of the moving average
    :return: the moving averaged list
    """

    ret = np.cumsum(l, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 25, fill = '=', eta=True):
    iteration += 1

    percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration) / float(total)))
    filledLength = int(length * iteration // total) + 1

    bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

    print('\r {} [{}] {}% {}'.format(prefix, bar, percent, suffix), end='\r')
    sys.stdout.flush()

    # Print New Line on Complete
    if iteration == total:
        print(' ')


################################################################
# Time utils
def format_time_interval(seconds, time_format=None):
    if time_format is None:
        if 0 < seconds < 3600:
            time_format = "%M:%S"
        elif 3600 < seconds < 24*3600:
            time_format = "%H:%M:%S"
        else:
            time_format = "%d days, %H:%M:%S"

    formatted_time = time.strftime(time_format, time.gmtime(seconds))
    return formatted_time


def get_timestamp(time_format=None):
    if time_format is None:
        time_format = "%Y-%m-%d_%H:%M:%S"
    return datetime.datetime.now().strftime(time_format)


################################################################
# Deprecated (function cemetery)

def resample_list(x, min_samples=None, max_samples=None):
    warnings.warn('function renamed to clamp_list', DeprecationWarning)
    return clamp_list(x, min_samples, max_samples)

def split_train_val(all_images, ratio=0.2, indexes_train=None):
    warnings.warn('function renamed to split_list', DeprecationWarning)
    return split_list(all_images, ratio, indexes_train)


