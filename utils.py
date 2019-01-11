import math
import numpy as np
import cv2
import sys
import nibabel as nib
import itertools as iter
import datetime

def check_filename_chars(filename):
    filename = filename.replace('.', ',')  # Avoid decimal points that could be confounded with extension .
    filename = filename.replace(' ', '')  # Avoid spaces
    return filename

def get_formatted_timedate():
    time_format = "%Y-%m-%d_%H:%M:%S"
    return     datetime.datetime.now().strftime(time_format)


def get_crossval_indexes(images, fold_idx, num_folds):
    assert 0 <= fold_idx < num_folds, "Fold idx out of bounds"

    images_per_fold = math.ceil(len(images) / num_folds)
    start_idx = fold_idx * images_per_fold
    stop_idx = min((fold_idx + 1) * images_per_fold, len(images))

    return start_idx, stop_idx

def get_val_split_indexes(images, split_ratio=0.2):
    start_idx = 0
    stop_idx = math.ceil(len(images) * split_ratio)
    assert stop_idx < len(images)

    return start_idx, stop_idx


def get_resampling_indexes(num_indexes_in, num_indexes_out):
    # TODO make more elegant
    assert num_indexes_in > 0

    resampled_idxs = list()
    sampling_left = num_indexes_out

    # Repeat all patches until sampling_left is smaller than num_patches
    if num_indexes_in < num_indexes_out:
        while sampling_left >= num_indexes_in:
            resampled_idxs += range(0, num_indexes_in)
            sampling_left -= num_indexes_in

    # Fill rest of indexes with uniform undersampling
    if sampling_left > 0:
        sampling_step = float(num_indexes_in) / sampling_left
        sampling_point = 0.0
        for i in range(sampling_left):
            resampled_idxs.append(int(math.floor(sampling_point)))
            sampling_point += sampling_step

    assert len(resampled_idxs) == num_indexes_out
    return resampled_idxs

def normalize_image(img):
    image = np.round(255.0*((img - np.min(img)) / (np.max(img) - np.min(img))), decimals=0)
    return image.astype('uint8')

def normalize_and_save_patch(patch, filename):
    save_size = (128, 128)

    patch_out = None
    for modality in patch:
        p = normalize_image(modality)
        p_big = cv2.resize(np.copy(p), save_size, interpolation=cv2.INTER_NEAREST)
        patch_out = p_big if patch_out is None else np.concatenate((patch_out, p_big), axis=1)

    print("Saving {}".format(filename))
    cv2.imwrite(filename, patch_out)


def store_batch(filename, x, y_in):
    y = y_in

    cell_shape = (6 * 60, 6 * 60, 1)

    x_save = np.zeros(cell_shape)
    y_save = np.zeros(cell_shape)

    patch_shape = (56, 56, 1)
    modality_idx = 0

    # 6*6
    for i, j in iter.product(range(0, 6), range(6)):
        idx = np.ravel_multi_index((i, j), (6, 6))

        selection = [slice(i * 60, i * 60 + patch_shape[0]), slice(j * 60, j * 60 + patch_shape[1]), slice(None)]
        x_save[selection] = np.expand_dims(x[idx, modality_idx], axis=-1)

        selection = [slice(i * 60, i * 60 + patch_shape[0]), slice(j * 60, j * 60 + patch_shape[1]), slice(None)]
        y_save[selection] = np.expand_dims(y[idx, 0], axis=-1)


    x_save = x_save - np.min(x_save)

    print("Storing batch {}".format(filename.format('x')))
    img_x = nib.Nifti1Image(x_save, np.eye(4))
    nib.save(img_x, filename.format('x'))

    print("Storing batch {}".format(filename.format('y')))
    img_y = nib.Nifti1Image(y_save, np.eye(4))
    nib.save(img_y, filename.format('y'))


def store_sample_patches(filename, x, y_in):
    print("Store x in-dim {}".format(x.ndim))
    if x.ndim is 3:
        x = np.expand_dims(x, -1)

    if y_in.ndim != x.ndim:
        print("Detected categorical y {}".format(y_in.shape))
        y = np.reshape(y_in[:, 1], (1, 64, 64, 1))
    else:
        y = y_in

    num_patches = x.shape[0] + 1
    cell_shape = (num_patches * 80, 64, 1)
    patch_shape = (64, 64, 1)

    x_save = np.zeros(cell_shape)

    for i in range(x.shape[0] + 1):
        selection = (slice(i * 80, i * 80 + patch_shape[0]), slice(None), slice(None))

        if i < x.shape[0]:
            x_save[selection] = normalize_image(x[i])
        else:
            x_save[selection] = normalize_image(y[0])

    print("Storing batch {}".format(filename.format('x')))
    img_x = nib.Nifti1Image(x_save, np.eye(4))
    nib.save(img_x, filename.format('x'))