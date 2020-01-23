import os
import torch
import numpy as np
import nibabel as nib
import niclib as nl

### 0. Paths
data_path = 'path/to/campinas-mini' # Change this to point to the dataset in your filesystem
checkpoints_path = nl.make_dir('checkpoints/')
results_path = nl.make_dir('results/')
metrics_path = nl.make_dir('metrics/')
log_path = nl.make_dir('log/')

### 1. Dataset load
case_paths = [f.path for f in os.scandir(data_path) if f.is_dir()]
case_paths, test_case_paths = case_paths[:-3], case_paths[-3:] # Set aside 3 images for testing at the end

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
        num_patches=2000 * len(dataset_train)))
val_patch_set = nl.generator.PatchSet(
    images=dataset_val,
    patch_shape=(32, 32, 32),
    normalize='none',
    sampling=nl.generator.UniformSampling(
        step=(16, 16, 16),
        num_patches=2000 * len(dataset_val)))

train_gen = nl.generator.make_generator(
    set=nl.generator.ZipSet([train_patch_set, train_patch_set]), batch_size=32, shuffle=True)
val_gen = nl.generator.make_generator(
    set=nl.generator.ZipSet([val_patch_set, val_patch_set]), batch_size=32, shuffle=True)
print("Train and val patch generators with {} and {} patches".format(len(train_patch_set), len(val_patch_set)))

### 3. Instance the torch model and trainer instance
guerrero_model = nl.model.uResNet_guerrero(
    in_ch=1, out_ch=1, ndims=3, activation=torch.nn.Sigmoid(), skip_connections=False)

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
        nl.net.train.EarlyStopping(metric_name='loss', mode='min', patience=5),
        nl.net.train.Logger(log_path + 'train_log.csv')],
    device='cuda')

trainer.train(guerrero_model, train_gen, val_gen)
guerrero_trained = torch.load(checkpoints_path + 'example_autoencoder.pt')

### 4. Finally, use the trained model to autoencode a sample image
predictor = nl.net.test.PatchTester(
    patch_shape=(1, 32, 32, 32),
    patch_out_shape=(1, 32, 32, 32),
    extraction_step=(16, 16, 16),
    normalize='none',
    activation=None)

test_images, autoencoded_images  = [], []
for n, test_case_path in enumerate(test_case_paths):
    test_nifti = nib.load(os.path.join(test_case_path, 't1.nii.gz'))
    test_image = np.expand_dims(test_nifti.get_data(), axis=0)

    print("Autoencoding: {}".format(test_case_path))
    autoencoded_image = predictor.predict(guerrero_trained, test_image)
    nl.save_nifti(
        filepath=results_path + 'test_{}_autoencoded.nii.gz'.format(n),
        volume=np.squeeze(autoencoded_image, axis=0), # Remove single channel dimension for storage
        reference=test_nifti) # Copy headers and affine from test_nifti

    # Add the input and output images to a list for metrics computation
    test_images.append(test_image)
    autoencoded_images.append(autoencoded_image)

# Finally, compute evaluation metrics for all test images and store in disk
autoencoder_metrics = nl.metrics.compute_metrics(
    outputs=autoencoded_images,
    targets=test_images,
    metrics={
        'mae': nl.metrics.mae, # Mean Absolute Error
        'ssim': nl.metrics.ssim, # Structural Similarity Index
        'psnr': nl.metrics.psnr}) # Peak Signal to Noise Ratio

nl.save_to_csv(metrics_path + 'autoencoder_metrics.csv', autoencoder_metrics)