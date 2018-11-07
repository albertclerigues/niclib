from niclib.patch.centers import *
from niclib.patch.slices import *

from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def get_default_options(self):
        pass

    @abstractmethod
    def get_centers(self, image):
        pass


class HybridLesionSampling(Sampler):
    def __init__(self, in_shape, num_min_max_lesion, num_min_max_uniform, max_offset=None):
        self.in_shape = in_shape
        self.num_lesion = num_min_max_lesion
        self.num_uniform = num_min_max_uniform
        self.max_offset = in_shape

    def get_default_options(self):
        return {'num_min_max_lesion': (500, 2000), 'num_min_max_uniform': (500, 2000), 'max_offset': None}

    def get_centers(self, image):
        # Positive
        pos_centers = sample_positive_patch_centers(image.labels)
        pos_centers = resample_centers(
            pos_centers, min_samples=self.num_lesion[0], max_samples=self.num_lesion[1])

        pos_centers = randomly_offset_centers(
            pos_centers, offset_shape=self.max_offset, patch_shape=self.in_shape, original_vol=image.data)

        # Uniform
        uniform_extraction_step = (6, 6, 3)
        while True:
            unif_centers = sample_uniform_patch_centers(self.in_shape, uniform_extraction_step, image.data)
            unif_centers = resample_centers(unif_centers, max_samples=self.num_uniform[1])

            if unif_centers.shape[0] >= self.num_uniform[1] - 1:
                break

            uniform_extraction_step = tuple([np.maximum(1, dim_step - 1) for dim_step in uniform_extraction_step])
            if np.array_equal(uniform_extraction_step, (1, 1, 1)):
                raise (ValueError, "Cannot extract enough uniform patches, please decrease number of patches")

        return pos_centers, unif_centers

class UniformSampling(Sampler):
    def __init__(self, in_shape, extraction_step):
        self.in_shape = in_shape
        self.extraction_step = extraction_step

    def get_default_options(self):
        return dict(extraction_step=(1,1,1))

    def get_centers(self, image):
        unif_centers = sample_uniform_patch_centers(self.in_shape, self.extraction_step, image.data)
        return unif_centers
