import os
import nibabel as nib
import numpy as np
import itertools as iter

from niclib.dataset.NICdataset import NICdataset, NICimage


class Isles2018(NICdataset):
    def __init__(self, dataset_path, num_volumes=(94, 0), load_testing=True):
        super().__init__()
        dataset_path = os.path.expanduser(dataset_path)

        print("Loading ISLES2018 dataset...")

        pattern = ['training/case_{}/', 'testing/test_{}/{}.nii']
        modalities = ['CT_SS', 'CBF', 'CBV', 'MTT', 'Tmax']

        # Training loading
        for case_idx in range(num_volumes[0]):
            case_folder = os.path.join(dataset_path, pattern[0].format(str(case_idx + 1)))

            # Initialize variables
            is_initialized, image_data, labels, nib_file = False, None, None, None

            # Load modalities one by one
            for root, dirs, files in os.walk(case_folder):
                found_modality = [modality for modality in iter.chain(modalities, ['OT']) if modality in root]
                if not found_modality:
                    continue

                modality_file = [f for f in files if 'nii' in f]
                assert modality_file
                modality_path = os.path.join(root, modality_file[0])

                nib_file = nib.load(modality_path)
                if not is_initialized:
                    # Load volume to check dimensions (not the same for all train samples)
                    vol = nib_file.get_data()

                    image_data = np.zeros((len(modalities),) + vol.shape)
                    labels = np.zeros((1,) + vol.shape)

                    is_initialized = True

                vol = nib_file.get_data()
                if found_modality[0] is 'OT':
                    labels[0] = vol
                else:
                    image_data[modalities.index(found_modality[0])] = vol

            sample = NICimage(id=case_idx + 1, nib_file=nib_file, image_data=image_data, foreground=None, labels=labels)
            self.add_train(sample)

        # Testing loading
        if not load_testing:
            return

        """
        for case_idx in range(num_volumes[1]):
            # Check folder exists (some samples missing)
            filename = dataset_path + pattern[1].format(str(case_idx + 1), modalities[0])
            if not os.path.exists(filename):
                log.debug("Skipping, data not found on {}".format(filename))
                continue

            sample = NICimage(id=case_idx + 1)

            # Load volume to check dimensions (not the same for all train samples)
            nib_file = nib.load(filename)
            sample.set_nib_headers(nib_file)

            vol = sample.nib.get_data()

            sample.data = np.zeros((len(modalities),) + vol.shape)

            # Load all modalities (except last which is gt segmentation) into last appended ndarray
            sample.data[0] = vol
            for i in range(1, len(modalities)):
                sample.data[i] = nib.load(dataset_path + pattern[1].format(str(case_idx + 1), modalities[i])).get_data()

            self.add_test(sample)
        """

