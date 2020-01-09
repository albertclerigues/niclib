import torch

from ..utils import print_progress_bar, RemainingTimeEstimator, save_nifti
from ..generator import make_generator
from ..generator.patch import PatchSet, sample_centers_uniform, _get_patch_slice


class PatchTester:
    """Forward pass a volume through the given network using uniformly sampled patches. After a patch is predicted, it
    is accumulated by averaging back in a common space.

    :param patch_shape: tuple (X, Y, Z) with the input patch shape of the model.
    :param patch_out_shape: (default: None) shape of the network forward passed patch, if None it is assumed to be of
        the same shape as ``patch_shape``.
    :param extraction_step: tuple (X, Y, Z) with the extraction step to uniformly sample patches.
    :param str normalize: either 'none', 'patch' or 'image'.
    :param activation: (default: None) the output activation after the forward pass.
    :param int batch_size: (default: 32) batch size for prediction, bigger batch sizes can speed up prediction if
        gpu utilization (NOT gpu memory) is under 100%.
    """

    def __init__(self, patch_shape, extraction_step, normalize, activation=None, batch_size=32, patch_out_shape=None):
        self.in_shape = patch_shape
        self.out_shape = patch_shape if patch_out_shape is None else patch_out_shape
        self.extraction_step = extraction_step
        self.normalize = normalize
        self.bs = batch_size
        self.activation=activation

        assert len(extraction_step) == 3, 'Please give extraction step as (X, Y, Z)'
        assert len(self.in_shape) == len(self.out_shape) ==  4, 'Please give shapes as (CH, X, Y, Z)'

        self.num_ch_out = self.out_shape[0]

    def predict(self, model, x, mask=None, device='cuda'):
        """ Predict the given volume ``x`` using the provided ``model``.

        :param torch.nn.Module model: The trained torch model.
        :param x: the input volume with shape (CH, X, Y, Z) to predict.
        :param mask: (default: None) a binary array of the same shape as x that defines the ROI for patch extraction.
        :param str device: the torch device identifier.
        :return: The accumulated outputs of the network as an array of the same shape as x.
        """
        assert len(x.shape) == 4, 'Please give image with shape (CH, X, Y, Z)'

        # Create patch generator with known patch center locations.
        x_centers = sample_centers_uniform(x[0], self.in_shape[1:], self.extraction_step, mask=mask)
        x_slices = _get_patch_slice(x_centers, self.in_shape[1:])
        patch_gen = make_generator(
            PatchSet([x], self.in_shape[1:], None, self.normalize, centers=[x_centers]), self.bs, shuffle=False)

        # Put accumulation in torch (GPU accelerated :D)
        voting_img = torch.zeros((self.num_ch_out,) + x[0].shape, device=device).float()
        counting_img = torch.zeros_like(voting_img).float()

        # Perform inference and accumulate results in torch (GPU accelerated :D (if device is cuda))
        model.eval()
        model.to(device)
        with torch.no_grad():
            rta = RemainingTimeEstimator(len(patch_gen))

            for n, (x_patch, x_slice) in enumerate(zip(patch_gen, x_slices)):
                x_patch = x_patch.to(device)

                y_pred = model(x_patch)
                if self.activation is not None:
                    y_pred = self.activation(y_pred)

                batch_slices = x_slices[self.bs * n:self.bs * (n + 1)]
                for predicted_patch, patch_slice in zip(y_pred, batch_slices):
                    voting_img[patch_slice] += predicted_patch
                    counting_img[patch_slice] += torch.ones_like(predicted_patch)

                print_progress_bar(self.bs * n, self.bs * len(patch_gen), suffix="patches predicted - ETA: {}".format(rta.update(n)))
            print_progress_bar(self.bs * len(patch_gen), self.bs * len(patch_gen), suffix="patches predicted - ETA: {}".format(rta.elapsed_time()))

        counting_img[counting_img == 0.0] = 1.0  # Avoid division by 0
        predicted_volume = torch.div(voting_img, counting_img).detach().cpu().numpy()
        return predicted_volume