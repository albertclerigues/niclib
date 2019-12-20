import os
import nibabel as nib
import numpy as np
import niclib as nl

### 1. Data loading

#Load dataset
data_path = '/media/user/dades/DATASETS/campinas'
case_image_filepaths = [os.path.join(data_path, d) for d in os.listdir(data_path) if '.nii.gz' in d]
dataset = nl.parallel_load(load_func=lambda x: nib.load(x).get_data(), arguments=case_image_filepaths)

# split into train and val
dataset_train, dataset_val = nl.split_list(dataset, fraction=0.8)
print('Dataset with {} train and {} val images'.format(len(dataset_train), len(dataset_val)))

### 2. Create patch generators
from niclib.generators.patch import PatchSet, UniformSampling
from niclib.generators import make_generator, ZipSet

autoencoder_sampling = UniformSampling(step=(16, 16, 16), num_patches=1000 * len(dataset_train))

train_patch_set = PatchSet(
	images=dataset_train, patch_shape=(32, 32, 32), normalize='image',	sampling=autoencoder_sampling)
val_patch_set = PatchSet(
    images=dataset_val, patch_shape=(32, 32, 32), normalize='image', sampling=autoencoder_sampling)

train_gen = make_generator(set=ZipSet([train_patch_set, train_patch_set]), batch_size=32, shuffle=True)
val_gen = make_generator(set=ZipSet([val_patch_set, val_patch_set]), batch_size=32, shuffle=True)

### 3. Instance the torch model, loss and optimizer and begin training








