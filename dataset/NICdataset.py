import copy
from abc import ABC, abstractmethod

import numpy as np
import nibabel as nib

class NICimage:
    def __init__(self, sample_id, nib_file, image_data, foreground, labels, as_type='float16'):
        # TODO make assertions to avoid trouble with dataset loading
        self.id = sample_id
        self.nib = {'affine': nib_file.affine, 'header': nib_file.header} # Affine, header
        self.data = image_data
        self.foreground = foreground
        self.labels = labels
        self.statistics = {'mean': [np.mean(modality) for modality in self.data],
                           'std_dev': [np.std(modality) for modality in self.data]}

class NICdataset(ABC): # Abstract class
    def __init__(self):
        self.train = []
        self.test = []

    # TODO assert no repeated ids

    def add_train(self, image_in):
        assert isinstance(image_in, NICimage)
        self.train.append(image_in)

    def add_test(self, image_in):
        assert isinstance(image_in, NICimage)
        self.test.append(image_in)