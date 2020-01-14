import os
import torch
import numpy as np
import nibabel as nib
import niclib as nl

### 1. Dataset load
data_path = '/media/user/dades/DATASETS/campinas-mini'
case_paths = [f.path for f in os.scandir(data_path) if f.is_dir()]
case_paths, test_case_paths = nl.split_list(case_paths, fraction=0.9) # Set aside 3 images for testing

print("Loading training dataset with {} images...".format(len(case_paths)))
def load_case(case_path):
    image = nib.load(os.path.join(case_path, 't1.nii.gz')).get_data()
    image = np.expand_dims(image, axis=0) # Add single channel dimension
    image = nl.data.clip_percentile(image, [0.0, 99.99])  # Clip to ignore bright extrema
    image = nl.data.adjust_range(image, [0.0, 1.0]) # Adjust range to 0-1 for Sigmoid activation
    return image
dataset = nl.parallel_load(load_func=load_case, arguments=case_paths, num_workers=12)

dataset_train, dataset_val = nl.split_list(dataset, fraction=0.8) # Split images into train and validation
print('Training dataset with {} train and {} val images'.format(len(dataset_train), len(dataset_val)))

### 2. Create training and validation patch generators
train_patch_set = nl.generator.PatchSet(
	images=dataset_train,
    patch_shape=(32, 32, 32),
    normalize='none',
    sampling=nl.generator.UniformSampling(
        step=(16, 16, 16),
        num_patches=500 * len(dataset_train)))

train_gen = nl.generator.make_generator(
    set=nl.generator.ZipSet([train_patch_set, train_patch_set]), batch_size=32, shuffle=True)


val_patch_set = nl.generator.PatchSet(
    images=dataset_val,
    patch_shape=(32, 32, 32),
    normalize='none',
    sampling=nl.generator.UniformSampling(
        step=(16, 16, 16),
        num_patches=500 * len(dataset_val)))

val_gen = nl.generator.make_generator(
    set=nl.generator.ZipSet([val_patch_set, val_patch_set]), batch_size=32, shuffle=True)

print("Train and val patch generators with {} and {} patches".format(len(train_patch_set), len(val_patch_set)))

### 3. Instance the torch model and trainer instance
guerrero_model = nl.model.uResNet_guerrero(
    in_ch=1, out_ch=1, ndims=3, activation=torch.nn.Sigmoid(), skip_connections=False)

checkpoints_path = nl.make_dir('checkpoints/')
trainer = nl.net.train.Trainer(
    max_epochs=100,
    loss_func=torch.nn.MSELoss(),
    optimizer=torch.optim.Adadelta,
    optimizer_opts={'lr': 1.0},
    train_metrics={'ssim': nl.metrics.ssim, 'psnr': nl.metrics.psnr},
    val_metrics={'ssim': nl.metrics.ssim, 'psnr': nl.metrics.psnr},
    plugins=[
        nl.net.train.ProgressBar(),
        nl.net.train.ModelCheckpoint(checkpoints_path + 'example_autoencoder.pt', save='best', metric_name='loss'),
        nl.net.train.EarlyStopping(metric_name='loss', patience=5)],
    device='cuda')

trainer.train(guerrero_model, train_gen, val_gen)
guerrero_trained = torch.load(checkpoints_path + 'example_autoencoder.pt')

### 4. Finally, use the trained model to autoencode a sample image
results_path = nl.make_dir('results/')
predictor = nl.net.test.PatchTester(
    patch_shape=(1, 32, 32, 32),
    patch_out_shape=(1, 32, 32, 32),
    extraction_step=(16, 16, 16),
    normalize='none',
    activation=None)

for n, test_case_path in enumerate(test_case_paths):
    # Load test image and add single channel dim to obtain 4 dimensions as (CH, X, Y, Z)
    test_nifti = nib.load(os.path.join(test_case_path, 't1.nii.gz'))
    test_image = np.expand_dims(test_nifti.get_data(), axis=0)

    # Predict image with the predictor and store
    print("Autoencoding: {}".format(test_case_path))
    autoencoded_image = predictor.predict(guerrero_trained, test_image)
    autoencoded_image = np.squeeze(autoencoded_image, axis=0) # Remove single channel dimension for storage
    nl.save_nifti(results_path + 'test_{}_autoencoded.nii.gz'.format(n), autoencoded_image, reference=test_nifti)