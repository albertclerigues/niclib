import csv
import datetime
import math
import os
import sys
import time
import warnings
from concurrent.futures.thread import ThreadPoolExecutor

from typing import List, Any

import nibabel as nib
import numpy as np
import torch
from art import tprint

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
    >>> get_base_path('/home/user/t1_image.nii.gz') + 'transform.mat'
    '/home/user/transform.mat'
    """
    return os.path.dirname(filepath)


def remove_extension(filepath):
    """Removes all extensions of the filename pointed by filepath.

    :Example:

    >>> remove_extension('home/user/t1_image.nii.gz')
    'home/user/t1_image'
    >>> remove_extension('home/user/t1_image.nii.gz') + '_processed.nii.gz'
    'home/user/t1_image_processed.nii.gz'
    """
    paths = filepath.split('/')
    if '.' not in paths[-1]:  # No extension found in filepath
        return filepath
    filename_noext = paths[-1].split('.')[0]
    filepath_noext = '/'.join(paths[:-1] + [filename_noext])
    return filepath_noext


#######################################################################################################################
## I/O utils

def parallel_load(load_func, arguments, num_workers):
    """Loads a dataset using parallel threads for faster load times (especially of .nii.gz files).

    :param callable load_func: function that loads and returns one dataset element.
        It can return any type and have any number of positional arguments.
    :param arguments: (List[Any] or List[List[Any]]) list containing the load_func arguments for each dataset
        element to load. If load_func has more than one argument, the arguments must be provided as a list.
        The function is called as ``load_func(*arguments[i])`` to load the i :sup:`th` element.
    :param int num_workers: number of parallel workers to use.
    :return: A list with the loaded dataset elements

    :Example:

    >>> def load_case(case_path):
    >>>     return nibabel.load(os.path.join(case_path, 't1.nii.gz')).get_data()
    >>>
    >>> dataset = parallel_load(load_case, ['data/pt_01', 'data/pt_02', 'data/pt_03'])
    >>> print(dataset[0].shape)
    (182, 218, 182)
    """

    assert callable(load_func), 'load_func must be a callable function'
    # Adjust for variadic positional arguments (i.e. the * in fn(*args)) that need a list of arguments to work
    if all([not isinstance(arg, list) for arg in arguments]):
        arguments = [[arg] for arg in arguments]

    # Define output variable and load function wrapper to maintain correct list order
    dataset = [None] * len(arguments)
    def _run_load_func(n_, args_):
        dataset[n_] = load_func(*args_)

    # Parallel load the dataset
    future_tasks = []
    pool = ThreadPoolExecutor(max_workers=num_workers)
    for n, args in enumerate(arguments):
        future_tasks.append(pool.submit(_run_load_func, n, args))

    # Check if any exceptions occured during loading
    [future_task.result() for future_task in future_tasks]

    pool.shutdown(wait=True)
    return dataset


def save_nifti(filepath, volume, dtype=None, reference=None, channel_handling='none'):
    """Saves the given volume array as a Nifti1Image using nibabel.

    :param str filepath: filename where the nifti will be saved
    :param numpy.ndarray volume: the volume with shape (X, Y, Z) or (CH, X, Y, Z) to save in a nifti image
    :param dtype: (optional) data type for the stored image (default: same dtype as `image`)
    :param nibabel.Nifti1Image reference: (optional) reference nifti from where to take the affine transform and header
    :param str channel_handling: (default: ``'none'``) One of ``'none'``, ``'last'`` or ``'split'``.
        If ``none``, the array is stored in the nifti as given. If  ``'last'`` the channel dimension is put last, this
        is useful to visualize images as multi-component data in *ITK-SNAP*. If ``'split'``, then the image channels
        are each stored in a different nifti file.
    """

    # Multichannel image handling
    assert channel_handling in {'none', 'last', 'split'}
    if len(volume.shape) == 4 and channel_handling != 'none':
        if channel_handling == 'last':
            volume = np.transpose(volume, axes=(1, 2, 3, 0))
        elif channel_handling == 'split':
            for n, channel in enumerate(volume):
                savename = '{}_ch{}.nii.gz'.format(remove_extension(filepath), n)
                save_nifti(savename, channel, dtype=dtype, reference=reference)
            return

    if dtype is not None:
        volume = volume.astype(dtype)

    if reference is None:
        nifti = nib.Nifti1Image(volume, np.eye(4))
    else:
        nifti = nib.Nifti1Image(volume, reference.affine, reference.header)

    print("Saving nifti: {}".format(filepath))
    nifti.to_filename(filepath)


def save_to_csv(filepath, dict_list, append=False):
    """Saves a list of dictionaries as a .csv file.

    :param str filepath: the output filepath
    :param List[Dict] dict_list: The data to store as a list of dictionaries.
        Each dictionary will correspond to a row of the .csv file with a column for each key in the dictionaries.
    :param bool append: If True, it will append the contents to an existing file.

    :Example:

    >>> save_to_csv('data.csv', [{'id': '0', 'score': 0.5}, {'id': '1', 'score': 0.8}])
    """
    assert isinstance(dict_list, list) and all([isinstance(d, dict) for d in dict_list])
    with open(filepath, mode='a') as f:
        csv_writer = csv.DictWriter(f, dict_list[0].keys(), restval='', extrasaction='raise', dialect='unix')
        if not append:
            csv_writer.writeheader()
        csv_writer.writerows(dict_list)


def load_from_csv(filepath):
    """Loads a .csv file as a list of dictionaries

    :return: a list of dictionaries (one per row)

    :Example:

    >>> print(load_from_csv('data.csv')[0])
    {'id': '0', 'score': 0.5}
    """

    with open(filepath, mode='r') as f:
        csv_reader = csv.DictReader(f, restval='', dialect='unix')
        return [row for row in csv_reader]


#######################################################################################################################
############ List utils

def resample_list(l, n):
    """Resamples a given list to have length `n`.

    List elements are repeated or removed at regular intervals to reach the desired length.

    :param list l: list to resample
    :param int n: desired length of resampled list
    :return list: the resampled list of length `n`

    :Example:

    >>> resample_list([0, 1, 2, 3, 4, 5], n=3)
    [0, 2, 4]
    >>> resample_list([0, 1, 2, 3], n=6)
    [0, 1, 2, 3, 0, 2]
    """
    assert n == int(n)
    n = int(n)

    if len(l) < n:  # List smaller than n (Repeat elements)
        resampling_idxs = list(range(len(l))) * (n // len(l))  # Full repetitions

        if len(resampling_idxs) < n:  # Partial repetitions
            resampling_idxs += np.arange(
                start=0.0, stop=float(len(l)) - 1.0, step=len(l) / float(n % len(l))).astype(int).tolist()
        return [l[i] for i in resampling_idxs]

    if len(l) > n:  # List bigger than n (Subsample elements)
        resampling_idxs = np.arange(start=0.0, stop=float(len(l)) - 1.0, step=len(l) / float(n)).astype(int).tolist()
        return [l[i] for i in resampling_idxs]

    if len(l) == n:
        return l



def split_list(l, fraction=None, indexes=None):
    """Splits a given list in two sub-lists ``(a, b)`` either by fraction or by indexes. Only one of the two options should be different to None.

    :param list l: list to split
    :param float fraction: (default: None) fraction of elements for list a (0.0 < fraction < 1.0).
    :param List[int] indexes: (default: None) list of integer indexes of elements for list ``a``.
    :return: tuple of two lists ``(a, b)`` where ``a`` contains the given fraction or indexes and ``b`` the rest.

    :Example:

    >>> split_list([0, 1, 2, 3, 4, 5], fraction=0.5)
    ([0, 1, 2], [3, 4, 5])
    >>> split_list(['a', 'b', 'c', 'd'], fraction=0.75)
    (['a', 'b', 'c'], ['d'])
    >>> split_list([0, 1, 2, 3, 4, 5], indexes=[0, 2, 4])
    ([0, 2, 4], [1, 3, 5])
    >>> split_list(['a', 'b', 'c', 'd'], indexes=[0, 3])
    (['a', 'd'], ['b', 'c'])
    """
    assert any([fraction is None, indexes is None]) and not all([fraction is None, indexes is None])

    if fraction is not None:
        assert isinstance(l, list) and 0.0 < fraction < 1.0
        split_idx = math.ceil(len(l) * fraction)
        return l[:split_idx], l[split_idx:]
    else:  # indexes is not None
        assert isinstance(l, list) and all([isinstance(idx, int) for idx in indexes])
        list_a = [a for n, a in enumerate(l) if n in indexes]
        list_b = [b for n, b in enumerate(l) if n not in indexes]
        return list_a, list_b


def moving_average(l, n):
    """ Performs a moving average of window n of the list.

    :param l: the list
    :param n: window size for the moving average
    :return: the moving averaged list with length = len(l) - n + 1

    :Example:

    >>> moving_average([0, 0, 0, 1.5, 0, 0, 0], 3)
    [0.0, 0.5, 0.5, 0.5, 0.0]
    """

    ret = np.cumsum(l, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return list(ret[n - 1:] / n)


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
        elif 3600 < seconds < 24 * 3600:
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
    """Provides an estimation the remaining execution time.

    .. py:method:: update(iter_num)

        :return: (str) formatted estimated remaining time

    .. py:method:: elapsed_time()

        :return: (str) formatted elapsed time (time passed since start)

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
        if iter_num >= self.total_iters - 1:
            return self.elapsed_time()
        return format_time_interval(np.mean(iter_times_filtered) * (self.total_iters - current_iter['num']))

    def elapsed_time(self):
        return format_time_interval(time.time() - self.start_time)


#######################################################################################################################
# Print utils
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=25, fill='='):
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
    assert 0 <= iteration <= total

    iteration += 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total) + 1
    bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

    print('\r {} [{}] {}/{} ({}%) {}'.format(prefix, bar, iteration, total, percent, suffix), end='\r')
    if iteration >= total:  # Print new line on completion
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

