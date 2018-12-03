import itertools
from abc import ABC
import numpy as np

class NIC_Image:
    def __init__(self, sample_id, nib_file, image_data, foreground, labels, as_type='float16'):
        # TODO make assertions to avoid trouble with dataset loading
        self.id = sample_id
        self.nib = {'affine': nib_file.affine, 'header': nib_file.header} # Affine, header
        self.data = image_data
        self.foreground = foreground
        self.labels = labels
        self.statistics = {'mean': [np.mean(modality) for modality in self.data],
                           'std_dev': [np.std(modality) for modality in self.data]}

class NIC_Dataset(ABC): # Abstract class
    def __init__(self):
        self.train = []
        self.test = []

    # TODO assert no repeated ids
    def add_train(self, image_in):
        assert isinstance(image_in, NIC_Image)
        self.train.append(image_in)

    def add_test(self, image_in):
        assert isinstance(image_in, NIC_Image)
        self.test.append(image_in)

    @staticmethod
    def get_by_id(wanted_id, images):
        if isinstance(images, NIC_Dataset):
            for image in itertools.chain(images.train, images.test):
                assert isinstance(image, NIC_Image)
                if image.id == wanted_id:
                    return image
        elif isinstance(images, list):
            for image in images:
                assert isinstance(image, NIC_Image)
                if image.id == wanted_id:
                    return image
        else:
            raise (ValueError, "Given images are not a valid instance of NICdataset or a list of NICimages")


        raise(ValueError, "Desired id not found in given images")