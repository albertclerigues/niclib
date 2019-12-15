import csv
import datetime
import math
import os
import sys
import time
import warnings

import nibabel as nib
import numpy as np
import torch
from art import tprint

#######################################################################################################################
## niclib imports # from . import net, generators, data, metrics
from . import net, generators, data, metrics

#######################################################################################################################
## Reproducibility and determinism
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#######################################################################################################################
## Path utils
def make_dir(dir_path):
    """Recursively creates the directories in the given path.

    :return: the input dir_path, useful to store the path in a variable while ensuring it exists.

    :Example:
    >>> output_folder = make_dir('/home/user/output_images')
    >>> print(output_folder)
    '/home/user/output_images'
    """
    parent_dir = os.path.abspath(os.path.join(dir_path, '..'))
    if not os.path.isdir(parent_dir):
        make_dir(parent_dir)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path

def get_filename(filepath, extension=True):
    """Returns the filename of the file pointed by `filepath`.

    :param str filepath: the full path
    :param bool extension: remove all file extensions before returning (default: True)
    :return str: the filename

    :Example:

    >>> get_filename('/home/user/t1_image.nii.gz')
    't1_image.nii.gz'
    >>> get_filename('/home/user/t1_image.nii.gz', extension=False)
    't1_image'
    """
    filename = os.path.basename(filepath)
    if not extension:
        filename = filename.split('.')[0]
    return filename

def get_base_path(filepath):
    """Returns the base path of the file pointed by `filepath`.

    :Example:

    >>> get_base_path('/home/user/t1_image.nii.gz')
    '/home/user'
    """
    return os.path.dirname(filepath)

def remove_extension(filepath):
    """Removes all extensions of the filename pointed by filepath.

    :Example:

    >>> remove_extension('home/user/t1_image.nii.gz')
    'home/user/t1_image'
    """
    paths = filepath.split('/')
    if '.' not in paths[-1]: # No extension found in filepath
        return filepath
    filename_noext = paths[-1].split('.')[0]
    filepath_noext = '/'.join(paths[:-1] + [filename_noext])
    return filepath_noext

#######################################################################################################################
## I/O utils
def save_nifti(filepath, image, reference=None, dtype=None):
    """Saves the given array as a Nifti1Image.

    :param str filepath: filename where the nifti will be saved
    :param numpy.ndarray image: the volume to save in a nifti image
    :param nibabel.Nifti1Image reference: reference nifti from where to take the affine transform and header
    :param dtype: data type for the stored image
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
    """Saves a list of dictionaries as a .csv file.
    Each dictionary corresponds to a row of the final .csv file with one column per key in the dictionaries.

    :param str filepath:
    :param List[Dict] dict_list:

    :Example:

    >>> save_to_csv('data.csv', [{'id': '0', 'score': 0.5}, {'id': '1', 'score': 0.8}])
    """
    assert isinstance(dict_list, list) and all([isinstance(d, dict) for d in dict_list])
    with open(filepath, mode='w') as f:
        csv_writer = csv.DictWriter(f, dict_list[0].keys(), restval='', extrasaction='raise', dialect='unix')
        csv_writer.writeheader()
        csv_writer.writerows(dict_list)

def load_from_csv(filepath):
    """Loads a .csv file as a list of dictionaries

    :return: a list of dictionaries (one per row)
    """

    with open(filepath, mode='r') as f:
        csv_reader = csv.DictReader(f, restval='', dialect='unix')
        return [row for row in csv_reader]

# def save_array(filepath, arr):
#     pass
#
# def load_array(filepath, arr):
#     pass
#
# def append_line_to_file(filepath, line):
#     with open(filepath, mode='a') as f:
#         f.write('\n' + line)

#######################################################################################################################
############ List utils

def resample_list(l, n):
    """Resamples a given list to have length `n`.

    List elements are repeated or removed at regular intervals to reach the desired lenght.

    :param list l: list to resample
    :param int n: desired length of resampled list
    :return list: the resampled list of length `n`

    :Example:

    >>> resample_list([0, 1, 2, 3, 4, 5], n=3)
    [0, 2, 4]
    >>> resample_list([0, 1, 2, 3], n=6)
    [0, 1, 2, 3, 0, 2]
    """
    assert isinstance(l, list)

    if len(l) < n: # List smaller than n (Repeat elements)
        resampling_idxs = list(range(len(l))) * (n // len(l)) # Full repetitions
        if len(resampling_idxs) < n:
            resampling_idxs += np.arange(
                start=0.0, stop=float(len(l)) - 1.0, step=len(l) / float(n % len(l))).astype(int).tolist()
        return [l[i] for i in resampling_idxs]

    if len(l) > n: # List bigger than n (Subsample elements)
        resampling_idxs = np.arange(start=0.0, stop=float(len(l)) - 1.0, step=len(l) / float(n)).astype(int)
        return [l[i] for i in resampling_idxs]

    return l  # else len(l) exactly n

def split_list_by_fraction(l, fraction):
    """Splits a given list in two sub-lists (a, b) according to the given fraction (0 < fraction < 1).
    A tuple of lists (a, b) is returned where a contains the first fraction of l and b the rest.

    :param list l: list to split
    :param fraction: fraction of elements for a
    :return: tuple of two lists (a, b) where a contains the first fraction of l, and b the rest.

    :Example:

    >>> split_list_by_fraction([0, 1, 2, 3, 4, 5], fraction=0.5)
    ([0, 1, 2], [3, 4, 5])
    >>> split_list_by_fraction(['a', 'b', 'c', 'd'], fraction=0.75)
    (['a', 'b', 'c'], ['d'])
    """
    assert isinstance(l, list) and 0.0 < fraction < 1.0
    split_idx = math.ceil(len(l) * fraction)
    return l[:split_idx], l[split_idx:]

def split_list_by_indexes(l, indexes):
    """Splits a given list in two sub-lists (a, b) according to the given indexes.
    A tuple of lists (a, b) is returned where a contains the elements pointed by indexes and b the rest.

    :param list l: list to split
    :param indexes: list of integer indexes for a
    :return: tuple of two lists (a, b) where a contains the elements at the given indexes, and b the rest.

    :Example:

    >>> split_list_by_indexes([0, 1, 2, 3, 4, 5], indexes=[0, 2, 4])
    ([0, 2, 4], [1, 3, 5])
    >>> split_list_by_indexes(['a', 'b', 'c', 'd'], indexes=[0, 3])
    (['a', 'd'], ['b', 'c'])
    """

    assert isinstance(l, list) and all([isinstance(idx, int) for idx in indexes])
    list_a = [a for n, a in enumerate(l) if n in indexes]
    list_b = [b for n, b in enumerate(l) if n not in indexes]
    return list_a, list_b


def moving_average(l, n):
    """ Performs a moving average of window n of the list.
    :param l: the list
    :param n: window size for the moving average
    :return: the moving averaged list (where its length is `len(l) - n + 1`)

    :Example:

    >>> moving_average([0, 0, 0, 1.5, 0, 0, 0], 3)
    [0.0, 0.5, 0.5, 0.5, 0.0]
    """

    ret = np.cumsum(l, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1:] / n).to_list()

#######################################################################################################################
# Time utils
def format_time_interval(seconds, time_format=None):
    """Formats a time interval into a string.

    :param seconds: the time interval in seconds
    :param str time_format: (optional) Time format specification string (see `Time format code specification <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_ for information on how to format time)

    :Example:

    >>> format_time_interval(30)
    '00:30'
    >>> format_time_interval(300)
    '05:00'
    >>> format_time_interval(seconds=4000)
    '01:06:40'
    >>> format_time_interval(seconds=4000, time_format='%H hours and %M minutes')
    '01 hours and 06 minutes'
    """

    if time_format is None:
        if 0 < seconds < 3600:
            time_format = "%M:%S"
        elif 3600 < seconds < 24*3600:
            time_format = "%H:%M:%S"
        else:
            time_format = "%d days, %H:%M:%S"
    formatted_time = time.strftime(time_format, time.gmtime(seconds))
    return formatted_time

def get_timestamp(formatted=True, time_format='%Y-%m-%d_%H:%M:%S'):
    """Returns a formatted timestamp of the current system time.

    :param formatted: (default: True)
    :param time_format: (default: '%Y-%m-%d_%H:%M:%S') Time format specification string (see `Time format code specification <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_ for information on how to format time)

    :Example:

    >>> get_timestamp()
    '2019-12-01_00:00:01'
    >>> get_timestamp(formatted=False)
    1575562508.845833
    >>> get_timestamp(time_format='%d/%m/%Y')
    '01/12/2019'
    """
    now = datetime.datetime.now()
    return now.strftime(time_format) if formatted else now.timestamp()

class RemainingTimeEstimator:
    """
    Class that allows the estimation of remaining execution time.

    :Example:

    >>> rta = RemainingTimeEstimator(3)
    >>> for i in range(3):
    >>>     time.sleep(1)
    >>>     print(rta.update(i))
    >>> print('Total ' + rta.elapsed_time())
    00:03
    00:02
    00:01
    Total 0:03
    """

    def __init__(self, total_iters):
        self.total_iters = total_iters
        self.start_time = time.time()

        self.iter_times = []
        self.last_iter = {'num': -1, 'time': time.time()}

    def update(self, iter_num):
        """
        :return: (str) formatted estimated remaining time
        """
        assert iter_num > self.last_iter['num'], 'Please avoid time travelling'
        current_iter = {'num': iter_num, 'time': time.time()}
        current_time_per_iter = \
            (current_iter['time'] - self.last_iter['time']) / (current_iter['num'] - self.last_iter['num'])

        # Update iter times buffer
        self.iter_times.append(current_time_per_iter)
        if len(self.iter_times) > 100: self.iter_times.pop(0)

        # Remove extreme times
        iter_times_filtered = self.iter_times
        if len(self.iter_times) > 3:
            low, high = np.percentile(self.iter_times, [10, 90])
            iter_times_filtered = [t for t in self.iter_times if low <= t <= high]

        self.last_iter = current_iter
        return format_time_interval(np.mean(iter_times_filtered) * (self.total_iters - current_iter['num']))

    def elapsed_time(self):
        """
        :return: (str) formatted elapsed time (time passed since start)
        """
        return format_time_interval(time.time() - self.start_time)


#######################################################################################################################
# Print utils
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 25, fill = '='):
    """Prints a progress bar.

    :param int iteration: current iteration number (starting from 0)
    :param int total: total number of iterations
    :param str prefix: prefix to print before the progress bar
    :param str suffix: suffix to print after the progress bar

    :Example:

    >>> print_progress_bar(4, 100, prefix='Dataset A', suffix='images loaded')
    Dataset A [==>......................] 5/100 (5.0%) images loaded

    It can be easily integrated in an existing for loop by using enumerate():

    >>> for i, file in enumerate(files):
    >>>     print_progress_bar(i, len(files))
    >>>     process_file(file) # ...
    """
    assert 0 <= iteration < total

    iteration += 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total) + 1
    bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

    print('\r {} [{}] {}/{} ({}%) {}'.format(prefix, bar, iteration, total, percent, suffix), end='\r')
    if iteration == total: # Print new line on completion
        print(' ')
    sys.stdout.flush()

def print_big(a):
    """Prints a with big ASCII letters. Wrapper for ``tprint`` function of the ``art`` python module.

    :Example:

    >>> print_big('Hello World')
     _   _        _  _         __        __              _      _
    | | | |  ___ | || |  ___   \ \      / /  ___   _ __ | |  __| |
    | |_| | / _ \| || | / _ \   \ \ /\ / /  / _ \ | '__|| | / _` |
    |  _  ||  __/| || || (_) |   \ V  V /  | (_) || |   | || (_| |
    |_| |_| \___||_||_| \___/     \_/\_/    \___/ |_|   |_| \__,_|
    """
    tprint(a)

#######################################################################################################################
# Cemetery

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 25, fill = '='):
    warnings.warn('Please use print_progress_bar', DeprecationWarning)
    print_progress_bar(iteration, total, prefix, suffix, decimals, length, fill)

def split_list(l, fraction=0.2, indexes=None):
    """
    Splits a given list in two sub-lists (a, b) according to a fraction.

    :param list l: The list to split
    :param fraction: fraction of samples for list a
    :param indexes: indexes to use for list a
    :return: a tuple (a, b) with the training and validation sets
    """
    warnings.warn('Please use split_list_by_fraction or split_list_by_indexes', DeprecationWarning)
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
    warnings.warn('Please use resample_list', DeprecationWarning)

    if min_len is not None and len(l) < min_len: # under specified minimum
        resampling_idxs = list(np.mod(range(min_len), len(l)).astype(int)) # Oversampling of images
        return [l[i] for i in resampling_idxs]

    if max_len is not None and len(l) > max_len: # over specified maximum
        resampling_idxs = np.arange(start=0.0, stop=float(len(l)) - 1.0, step=len(l) / float(max_len)).astype(int)
        return [l[i] for i in resampling_idxs]

    return l  # len of l was already within desired range