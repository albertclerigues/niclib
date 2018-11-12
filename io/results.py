import numpy as np
import nibabel as nib
import os

import warnings
from niclib.dataset.NICdataset import NICimage

# TODO
class ResultFilenameGenerator:
    def __init__(self, result_type='probs', format='nii.gz'):
        pass

    def get_name(self, image, params=None):
        pass


def _save_result_volume(filename, image, result, dtype):
    assert isinstance(image, NICimage)

    probs_out = np.multiply(result, image.foreground).astype(dtype)
    img_out = nib.Nifti1Image(probs_out, image.nib['affine'], image.nib['header'])
    nib.save(img_out, filename)

def save_image_probs(filename, image, probs, dtype='float16'):
    print("Saving probs for image {}: {}".format(image.id, filename))
    _save_result_volume(filename, image, probs, dtype)

def save_image_seg(filename, image, probs, dtype='uint16'):
    print("Saving seg for image {}: {}".format(image.id, filename))
    _save_result_volume(filename, image, probs, dtype)


def save_result_set(result_path, original_images, filename_gen=None, result_type='probs', file_format='nii.gz'):
    # TODO
    pass

def load_result_set(result_path, original_images, filename_gen=None, result_type='probs', file_format='nii.gz'):
    """
    Loads a result set where the first term is the sample idx
    """
    assert os.path.exists(result_path)
    assert all([isinstance(img, NICimage) for img in original_images])
    #assert isinstance(filename_gen, ResultFilenameGenerator)

    result_set = [None] * len(original_images)
    for i, image in enumerate(original_images):
        # TODO use filename generators
        result_filename = "{}_{}.{}".format(image.id, result_type, file_format)
        result_pathfile = os.path.join(result_path, result_filename)

        if not os.path.isfile(result_pathfile):
            warnings.warn("Didn't find result for image {}".format(image.id), RuntimeWarning)
            continue

        result_set[i] = nib.load(result_pathfile).get_data()

    return result_set
