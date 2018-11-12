import numpy as np
from scipy import ndimage
from abc import ABC, abstractmethod

class Binarizer(ABC):
    @abstractmethod
    def binarize(self, probs):
        pass

class ThreshSizeBinarizer(Binarizer):
    def __init__(self, thresh=0.5, min_lesion_vox=10):
        self.thresh = thresh
        self.min_lesion_vox = min_lesion_vox

    def binarize(self, probs):
        """
        Generates final class prediction by thresholding according to threshold and filtering by minimum lesion size
        """

        # Apply threshold
        y_prob = probs > self.thresh

        # Get connected components information
        y_prob_labelled, nlesions = ndimage.label(y_prob)
        if nlesions > 0:
            label_list = np.arange(1, nlesions + 1)
            lesion_volumes = ndimage.labeled_comprehension(y_prob, y_prob_labelled, label_list, np.sum, float, 0)

            # Set to 0 invalid lesions
            lesions_to_ignore = [idx + 1 for idx, lesion_vol in enumerate(lesion_volumes) if lesion_vol < self.min_lesion_vox]
            y_prob_labelled[np.isin(y_prob_labelled, lesions_to_ignore)] = 0

        # Generate binary mask and return
        y_pred = (y_prob_labelled > 0).astype('uint8')

        return y_pred