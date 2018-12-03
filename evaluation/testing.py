import os

import torch
import numpy as np
import nibabel as nib

from niclib.dataset import NIC_Image
from niclib.evaluation.prediction import NIC_Predictor
from niclib.postprocessing.binarization import ThreshSizeBinarizer


class TestingPrediction:
    def __init__(self, predictor, out_path):
        assert isinstance(predictor, NIC_Predictor)
        self.test_predictor = predictor

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        self.out_path = out_path

    def predict_test_set(self, model, test_images):
        assert isinstance(test_images, list) and all([isinstance(img, NIC_Image) for img in test_images])

        for test_img in test_images:
            pred_img = self.test_predictor.predict_sample(model, test_img)

            # Image post-processing
            # TODO make POSTPROCESSING CLASS!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            final_img = ThreshSizeBinarizer(thresh=0.1, min_lesion_vox=200).binarize(pred_img)

            # Image storage
            final_img = final_img.astype('uint16')
            final_img = np.multiply(final_img, test_img.foreground)
            img_out = nib.Nifti1Image(final_img, test_img.nib['affine'], test_img.nib['header'])

            image_filepath = os.path.join(self.out_path, '{}_test_seg.nii'.format(test_img.id))
            nib.save(img_out, image_filepath)