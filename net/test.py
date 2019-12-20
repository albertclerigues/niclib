import torch

from ..utils import print_progress_bar, RemainingTimeEstimator, save_nifti
from ..__init__ import device as torch_device
from ..generators import make_generator
from ..generators.patch import PatchSet, sample_centers_uniform, _get_patch_slice


class PatchTester:
    def __init__(self, patch_shape, extraction_step, normalize, mask=None, activation=None, batch_size=32, patch_out_shape=None):
        self.in_shape = patch_shape
        self.out_shape = patch_shape if patch_out_shape is None else patch_out_shape
        self.extraction_step = extraction_step
        self.normalize = normalize
        self.bs = batch_size
        self.mask = mask
        self.activation=activation

        assert len(extraction_step) == 3, 'Please give extraction step as (X, Y, Z)'
        assert len(self.in_shape) == len(self.out_shape) ==  4, 'Please give shapes as (CH, X, Y, Z)'

        self.num_ch_out = self.out_shape[0]

    def predict(self, model, x, device=torch_device):
        assert len(x.shape) == 4, 'Please give image with shape (CH, X, Y, Z)'
        x_centers = sample_centers_uniform(x[0], self.in_shape[1:], self.extraction_step)
        x_slices = _get_patch_slice(x_centers, self.in_shape[1:])
        patch_gen = make_generator(
            PatchSet([x], self.in_shape[1:], None, self.normalize, centers=[x_centers]), self.bs, shuffle=False)

        # Put accumulation in torch (GPU accelerated :D)
        voting_img = torch.zeros((self.num_ch_out,) + x[0].shape, device=device).float()
        counting_img = torch.zeros_like(voting_img).float()

        # Perform inference and accumulate results in torch (GPU accelerated :D if device is cuda)
        model.eval()
        model.to(device)
        with torch.no_grad():
            rta = RemainingTimeEstimator(len(patch_gen))

            for n, (x_patch, x_slice) in enumerate(zip(patch_gen, x_slices)):
                x_patch = x_patch.to(device)

                y_pred = model(x_patch)
                if self.activation is not None:
                    y_pred = self.activation(y_pred)

                save_nifti('x_patch.nii.gz', x_patch[0, 0].float().detach().cpu().numpy())
                save_nifti('y_pred.nii.gz', torch.argmax(y_pred[0], dim=0).float().detach().cpu().numpy())
                raise NotImplementedError

                batch_slices = x_slices[self.bs * n:self.bs * (n + 1)]
                for predicted_patch, patch_slice in zip(y_pred, batch_slices):
                    voting_img[patch_slice] += predicted_patch
                    counting_img[patch_slice] += torch.ones_like(predicted_patch)

                print_progress_bar(self.bs * n, self.bs * len(patch_gen), suffix="patches predicted - ETA: {}".format(rta.update(n)))
            print_progress_bar(self.bs * len(patch_gen), self.bs * len(patch_gen), suffix="patches predicted - ETA: {}".format(rta.elapsed_time()))

        counting_img[counting_img == 0.0] = 1.0  # Avoid division by 0
        predicted_volume = torch.div(voting_img, counting_img).detach().cpu().numpy()
        print('probs', predicted_volume.shape)
        return predicted_volume