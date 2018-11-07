import math
import numpy as np
import cv2
import sys


def get_crossval_indexes(images, fold_idx, num_folds, images_per_fold):
    assert num_folds * images_per_fold >= len(images), "Not enough images for this crossvalidation"
    assert fold_idx < num_folds, "Impossible fold idx"

    start_idx = fold_idx * images_per_fold
    stop_idx = (fold_idx + 1) * images_per_fold

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