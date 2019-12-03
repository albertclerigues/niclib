import copy
import sys
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader
import itertools

from niclib2 import clamp_list
from niclib2 import printProgressBar
from niclib2.data import compute_normalization_statistics

def build_generator(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

class PatchSet(TorchDataset):
    def __init__(self, images, centers, patch_shape, normalize, dtype=np.float32):
        """
        Creates a torch dataset that returns patches extracted from images, y at the specified centers

        :param images: list of input volumes
        :param centers: a list with arr centers (a list of tuples (images,y,z)) for each volume in images
        :param patch_shape: a tuple (images,y,z) with the length of arr in each dimension
        :param bool normalize: if True, return patches from images with zero mean and unit variance w.r.t each case image
        """

        assert len(images) == len(centers)

        assert isinstance(centers, list)
        for case_centers in centers:
            assert isinstance(case_centers, list)
            assert all([isinstance(center, tuple) and len(center) == 3 for center in case_centers])

        self.images = images
        self.patch_shape = patch_shape
        self.dtype = dtype

        # Store centers as flattened array and store case_idx for fast extraction
        self.centers = []
        self.centers_case_index = []
        for case_index, case_centers in enumerate(centers):
            self.centers += case_centers  # append center list
            self.centers_case_index += [case_index] * len(case_centers)  # append the case_index for each center

        # Compute mean and variance for normalization
        self.do_normalize = normalize
        self.norm_stats = []
        if self.do_normalize:
            for image in self.images:
                mean, std = compute_normalization_statistics(image, ignore_zeros=False)
                self.norm_stats.append({'mean': mean, 'stdev': std})

        print("Making PatchSet with {} patches".format(len(self.centers)))

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, index):
        case_idx = self.centers_case_index[index]
        patch_center = self.centers[index]

        x_patch_slice = get_patch_slice(patch_center, self.patch_shape)

        x_patch = copy.deepcopy(self.images[case_idx][x_patch_slice]).astype('float')
        if self.do_normalize:
            x_patch = normalize_patch(x_patch, self.norm_stats[case_idx]['mean'], self.norm_stats[case_idx]['stdev'])

        if x_patch.shape[-1] == 1: # 2D/3D compatibility
            x_patch = np.squeeze(x_patch, axis=-1)

        x = torch.Tensor(np.ascontiguousarray(x_patch, dtype=self.dtype))
        return x


class SliceSet(TorchDataset):
    def __init__(self, images, slice_dim, normalize):
        """
        Creates a torch dataset that returns slices of the given images at the specified axis

        :param images: list of input volumes
        :param slice_dim: axis from which to slice
        :param bool normalize: if True, return patches from images with zero mean and unit variance w.r.t each case image
        """
        self.images = images
        self.slice_dim = slice_dim

        self.slice_index = []
        self.slice_index_case = []
        for n, image in enumerate(images):
            num_slices = image.shape[slice_dim]
            self.slice_index += list(range(num_slices))  # append center list
            self.slice_index_case += [n] * num_slices  # append the case_index for each center

        # Compute mean and variance for normalization
        self.do_normalize = normalize
        self.norm_stats = []
        if self.do_normalize:
            for image in self.images:
                self.norm_stats.append(
                    {'mean': [np.mean(channel) for channel in image],
                     'stdev': [np.std(channel) for channel in image]})

        print("Making SliceSet with {} slices".format(len(self.slice_index)))

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, n):
        case_idx = self.slice_index_case[n]
        slice_idx = self.slice_index[n]

        slice_selector = [slice(None)] * (len(self.images[case_idx].shape) - 1)
        slice_selector.insert(self.slice_dim, slice(slice_idx, slice_idx + 1))

        x_patch = copy.deepcopy(self.images[case_idx][tuple(slice_selector)])
        if self.do_normalize:
            x_patch = normalize_patch(x_patch, self.norm_stats[case_idx]['mean'], self.norm_stats[case_idx]['stdev'])

        if x_patch.shape[-1] == 1: # 2D/3D compatibility
            x_patch = np.squeeze(x_patch, axis=-1)

        x = torch.Tensor(np.ascontiguousarray(x_patch, dtype=np.float32))
        return x

class ListSet(TorchDataset):
    def __init__(self, items):
        assert isinstance(items, list)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class FunctionSet(TorchDataset):
    def __init__(self, func, length):
        assert callable(func)
        self.func = func
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.func(index)


class ZipSets(TorchDataset):
    def __init__(self, datasets):
        assert all([isinstance(dataset, TorchDataset) for dataset in datasets])
        self.datasets = datasets

    def __len__(self):
        return min([len(d) for d in self.datasets])

    def __getitem__(self, index):
        return tuple([dataset[index] for dataset in self.datasets])



def get_patch_slices(centers, patch_shape):
    """
    :param centers: list of (images,y,z) tuples
    :param patch_shape: (images,y,z) tuple with arr dimensions
    :return: a list of tuples each with (channel_slice, x_slice, y_slice, z_slice)
    """
    # Pre-compute arr sides for slicing
    half_sizes = [[dim // 2, dim // 2] for dim in patch_shape]
    for i in range(len(half_sizes)):  # If even dimension, subtract 1 to account for assymetry
        if patch_shape[i] % 2 == 0: half_sizes[i][1] -= 1

    patch_slices = [get_patch_slice(center, patch_shape, half_sizes) for center in centers]
    return patch_slices


def get_patch_slice(center, patch_shape, half_sizes=None):
    """
    :param center: (images,y,z) tuple
    :param patch_shape: (images,y,z) tuple with arr dimensions
    :param half_sizes: (optional) precomputed half_sizes to avoid redundant computations, i.e. when inside a for loop
    :return: a tuple with (channel_slice, x_slice, y_slice, z_slice)
    """

    ### Compute arr sides for slicing
    if half_sizes is None:
        half_sizes = [[dim // 2, dim // 2] for dim in patch_shape]
        for i in range(len(half_sizes)):  # If even dimension, subtract 1 to account for assymetry
            if patch_shape[i] % 2 == 0: half_sizes[i][1] -= 1

    # Actually create slices
    patch_slice = (slice(None),  # slice(None) selects all channels
                   slice(center[0] - half_sizes[0][0], center[0] + half_sizes[0][1] + 1),
                   slice(center[1] - half_sizes[1][0], center[1] + half_sizes[1][1] + 1),
                   slice(center[2] - half_sizes[2][0], center[2] + half_sizes[2][1] + 1))
    return patch_slice


def normalize_patch(patch, mean, std):
    """
    Normalises a arr to have zero mean and unit variance

    :param patch: numpy array
    :param mean: list containing the mean value of each modality in arr
    :param std: list containing the stdev value of each modality in arr
    :return: the arr with zero mean and unit variance
    """
    assert patch.shape[0] == len(mean) == len(std)

    for modality in range(patch.shape[0]):
        patch[modality] -= mean[modality]
        patch[modality] /= std[modality]
    return patch

def denormalize_patch(patch, mean, std):
    assert patch.shape[0] == len(mean) == len(std)
    for modality in range(patch.shape[0]):
        patch[modality] *= std[modality]
        patch[modality] += mean[modality]
    return patch

class PatchVolumePredictor:
    def __init__(self, in_shape, extraction_step, normalize, num_ch_out, out_shape=None,  skip_fn=None):
        self.in_shape = in_shape
        self.out_shape = in_shape if out_shape is None else out_shape
        self.extraction_step = extraction_step
        self.do_normalize = normalize
        self.num_ch_out = num_ch_out

        self.skip_fn = skip_fn

        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def predict(self, model, x, device='cuda'):
        # Make arr generator
        x_centers = sample_centers_uniform(x.shape[1:], self.in_shape, self.extraction_step)

        patch_gen = build_generator(
            PatchSet([x], [x_centers], self.in_shape, self.do_normalize), batch_size=1, shuffle=False)

        # Prepare inference variables
        x_slices = get_patch_slices(x_centers, self.in_shape)

        # Put accumulation in torch (GPU accelerated :D)
        voting_img = torch.zeros((self.num_ch_out,) + x[0].shape, device=device).float()
        counting_img = torch.zeros_like(voting_img).float()

        # Perform inference and accumulate results
        model.eval()
        model.to(device)
        with torch.no_grad():
            for n, (x_patch, x_slice) in enumerate(zip(patch_gen, x_slices)):
                x_patch = x_patch.to(device)

                y_pred = model(x_patch)
                y_pred = y_pred[0, 0:self.num_ch_out]

                voting_img[x_slice] += y_pred
                counting_img[x_slice] += torch.ones_like(y_pred)

                printProgressBar(n, len(patch_gen), suffix=" patches predicted")

        voting_img = voting_img.cpu().numpy()
        counting_img = counting_img.cpu().numpy()

        counting_img[counting_img == 0.0] = 1.0  # Avoid division by 0
        volume_probs = np.divide(voting_img, counting_img)

        print('probs', volume_probs.shape)

        return volume_probs


def sample_centers_balanced(vol, labels, patch_shape, num_centers):
    """
    TODO
    """

    assert len(vol.shape) == len(labels.shape) == len(patch_shape), '{} - {} - {}'.format(vol.shape, labels.shape, patch_shape)

    ### Obtain all centers for each label
    label_ids = np.unique(labels).tolist()
    labels_centers = {}
    for id_label in label_ids:
        labels_centers[id_label] = np.transpose(np.where(labels == id_label))

    ### Resample (repeating or removing) to appropiate number
    centers_per_label = num_centers / len(label_ids)
    for id_label in labels_centers.keys():
        labels_centers[id_label] = np.asarray(clamp_list(
            labels_centers[id_label], min_len=centers_per_label, max_len=centers_per_label))

    ### Add random offset that doesn't go outside bounds
    for id_label, centers_label in labels_centers.items():
        # Generate uniform random numbers between -1 and 1
        offset_values = 2.0 * (np.random.rand(centers_label.shape[0], 3) - 0.5)
        offset_ranges = np.stack([(patch_shape[0] // 2) * np.ones((centers_label.shape[0],)),
                                  (patch_shape[1] // 2) * np.ones((centers_label.shape[0],)),
                                  (patch_shape[2] // 2) * np.ones((centers_label.shape[0],))], axis=1)
        center_offsets = np.multiply(offset_values, offset_ranges).astype(int)
        centers_label += center_offsets

        # Ensure not out of bounds
        half_sizes = [[int(np.ceil(dim / 2.0)), int(np.floor(dim / 2.0))] for dim in patch_shape]
        centers_label[:, 0] = np.clip(centers_label[:, 0], half_sizes[0][0], vol.shape[0] - half_sizes[0][1])
        centers_label[:, 1] = np.clip(centers_label[:, 1], half_sizes[1][0], vol.shape[1] - half_sizes[1][1])
        centers_label[:, 2] = np.clip(centers_label[:, 2], half_sizes[2][0], vol.shape[2] - half_sizes[2][1])

    ### Join all centers classes together and return
    centers_numpy = np.concatenate([v for v in labels_centers.values()], axis=0)
    centers = [tuple(c.tolist()) for c in centers_numpy]

    return centers


def sample_centers_uniform(vol_shape, patch_shape, extraction_step, max_samples=None, foreground=None):
    """
    This sampling is uniform, not regular! It will extract patches

    :param vol_shape:
    :param patch_shape:
    :param extraction_step:
    :param max_samples: (Optional) If given, the centers will be resampled to max_len
    :param foreground: (Optional) If given, discard centers not in foreground
    :return:
    """

    assert len(vol_shape) == len(patch_shape) == len(extraction_step), '{}, {}, {}'.format(vol_shape, patch_shape, extraction_step)
    #print('vol_shape:{}, patch_shape:{}, extraction_step:{}'.format(vol_shape, patch_shape, extraction_step))

    if foreground is not None:
        assert len(foreground.shape) == len(vol_shape), '{}, {}'.format(foreground.shape, vol_shape)
        foreground = foreground.astype('float16')

    # Get arr half size
    half_sizes = [[dim // 2, dim // 2] for dim in patch_shape]
    for i in range(len(half_sizes)):  # If even dimension, subtract 1 to account for assymetry
        if patch_shape[i] % 2 == 0: half_sizes[i][1] -= 1

    # Generate the ranges for each dimension
    dim_ranges = []
    for dim in range(len(vol_shape)):
        dim_ranges.append(list(range(half_sizes[dim][0], vol_shape[dim] - half_sizes[dim][1], extraction_step[dim])))

    # Iterate over ranges to form the (images,y,z) tuples and append
    centers = []
    for center in itertools.product(*dim_ranges):
        if foreground is not None:
            if foreground[center[0], center[1], center[2]] == 0.0:  # Then is centered on background (bad)
                continue
        centers.append(center)

    if max_samples is not None:
        # Resample the list if too many extracted centers
        if len(centers) > max_samples:
            centers = clamp_list(centers, max_len=max_samples)

    return centers