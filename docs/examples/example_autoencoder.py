import os
import numpy as np
import torch
import nibabel as nib
import niclib as nl

### 1. Data loading
data_path = 'path/to/dataset'
case_image_filepaths = [os.path.join(data_path, d) for d in os.listdir(data_path) if '.nii.gz' in d]

dataset = nl.parallel_load(
    load_func=lambda x: nib.load(x).get_data(), arguments=case_image_filepaths, num_workers=12)

dataset_train, dataset_val = nl.split_list(dataset, fraction=0.8) # split into train and val
print('Dataset with {} train and {} val images'.format(len(dataset_train), len(dataset_val)))

### 2. Create patch generator
autoencoder_sampling = nl.generator.UniformSampling(step=(16, 16, 16), num_patches=1000 * len(dataset_train))

train_patch_set = nl.generator.PatchSet(
	images=dataset_train, patch_shape=(32, 32, 32), normalize='image', sampling=autoencoder_sampling)
val_patch_set = nl.generator.PatchSet(
    images=dataset_val, patch_shape=(32, 32, 32), normalize='image', sampling=autoencoder_sampling)

train_gen = nl.generator.make_generator(
    set=nl.generator.ZipSet([train_patch_set, train_patch_set]), batch_size=32, shuffle=True)
val_gen = nl.generator.make_generator(
    set=nl.generator.ZipSet([val_patch_set, val_patch_set]), batch_size=32, shuffle=True)

### 3. Instance the torch model, loss and optimizer and begin training
model = nl.model.uResNet_guerrero(in_ch=1, out_ch=1, ndims=3, activation=None)

nl.net.train.Trainer(
    max_epochs=200,
    loss_func=torch.nn.MSELoss(),
    optimizer=torch.optim.Adadelta,
    optimizer_opts={'lr': 0.9},
    train_metrics={},
    val_metrics={},
)









