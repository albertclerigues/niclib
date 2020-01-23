import os
import numpy as np
import nibabel as nib
import torch
from torch.nn import functional as F
import niclib as nl

### Relevant paths
data_path = 'path/to/campinas-mini' # Change this to point to the dataset in your filesystem
data_path = '/media/user/dades/DATASETS/campinas-mini' # Change this to point to the dataset in your filesystem
checkpoints_path = nl.make_dir('checkpoints/')
results_path = nl.make_dir('results/')
metrics_path = nl.make_dir('metrics/')
log_path = nl.make_dir('log/')

### 1. Dataset load
case_paths = [f.path for f in os.scandir(data_path) if f.is_dir()]
case_paths, test_case_paths = case_paths[:-3], case_paths[-3:] # Set aside 3 images for testing

print("Loading training dataset with {} images...".format(len(case_paths)))
def load_case(case_path):
    t1_nifti = nib.load(os.path.join(case_path, 't1.nii.gz'))
    t1_img = np.expand_dims(t1_nifti.get_data(), axis=0) # Add single channel modality
    tissue_filepaths = [os.path.join(case_path, 'fast_3c/fast_pve_{}.nii.gz'.format(i)) for i in range(3)]
    tissue_probabilties = np.stack([nib.load(tfp).get_data() for tfp in tissue_filepaths], axis=0)
    return {'nifiti': t1_nifti, 't1': t1_img, 'probs': tissue_probabilties}
dataset = nl.parallel_load(load_func=load_case, arguments=case_paths, num_workers=12)
dataset_train, dataset_val = nl.split_list(dataset, fraction=0.8) # Split images into train and validation
print('Training dataset with {} train and {} val images'.format(len(dataset_train), len(dataset_val)))


### 2. Create training and validation patch generators
train_sampling = nl.generator.BalancedSampling(
    labels=[np.argmax(case['probs'], axis=0) for case in dataset_train],
    num_patches=1000 * len(dataset_train))
train_patch_set = nl.generator.ZipSet([
    nl.generator.PatchSet(
        images=[case['t1'] for case in dataset_train],
        patch_shape=(32, 32, 32),
        normalize='image',
        sampling=train_sampling),
    nl.generator.PatchSet(
        images=[case['probs'] for case in dataset_train],
        patch_shape=(32, 32, 32),
        normalize='image',
        sampling=train_sampling)])

val_sampling = nl.generator.BalancedSampling(
    labels=[np.argmax(case['probs'], axis=0) for case in dataset_val],
    num_patches=1000 * len(dataset_val))
val_patch_set = nl.generator.ZipSet([
    nl.generator.PatchSet(
        images=[case['t1'] for case in dataset_val],
        patch_shape=(32, 32, 32),
        normalize='image',
        sampling=val_sampling),
    nl.generator.PatchSet(
        images=[case['probs'] for case in dataset_val],
        patch_shape=(32, 32, 32),
        normalize='image',
        sampling=val_sampling)])

train_gen = nl.generator.make_generator(set=train_patch_set, batch_size=32, shuffle=True)
val_gen = nl.generator.make_generator(set=val_patch_set, batch_size=32, shuffle=True)
print("Train and val patch generators with {} and {} patches".format(len(train_patch_set), len(val_patch_set)))

# ---------------------------------------
### 3. Instance the torch model, loss and trainer
guerrero_model = nl.model.uResNet_guerrero(in_ch=1, out_ch=3, ndims=3, activation=None)

# Some adaptation is needed for loss and metrics, since output are logits and target is a 3 channel probability map
wrapped_crossentropy_loss = nl.loss.LossWrapper(torch.nn.CrossEntropyLoss(),
    preprocess_fn=lambda out, tgt: (out, torch.argmax(F.softmax(tgt, dim=1), dim=1).long()))
wrapped_accuracy = nl.loss.LossWrapper(nl.metrics.accuracy,
    preprocess_fn=lambda out, tgt: (torch.argmax(F.softmax(out, dim=1), dim=1), torch.argmax(tgt, dim=1)))
wrapped_dsc = nl.loss.LossWrapper(nl.metrics.dsc,
    preprocess_fn=lambda out, tgt: (torch.argmax(F.softmax(out, dim=1), dim=1), torch.argmax(tgt, dim=1)))

trainer = nl.net.train.Trainer(
    max_epochs=100,
    loss_func=wrapped_crossentropy_loss,
    optimizer=torch.optim.Adadelta,
    optimizer_opts={'lr': 1.0},
    train_metrics={'acc': wrapped_accuracy, 'dsc': wrapped_dsc},
    val_metrics={'acc': wrapped_accuracy, 'dsc': wrapped_dsc},
    plugins=[
        nl.net.train.ProgressBar(),
        nl.net.train.ModelCheckpoint(checkpoints_path + 'tissue_seg.pt', save='best', metric_name='loss', mode='min'),
        nl.net.train.EarlyStopping(metric_name='loss', mode='min', patience=5),
        nl.net.train.Logger(log_path + 'train_log.csv')],
    device='cuda')

trainer.train(guerrero_model, train_gen, val_gen)
guerrero_trained = torch.load(checkpoints_path + 'tissue_seg_net.pt')

### 4. Finally, use the trained model to segment the test images
predictor = nl.net.test.PatchTester(
    patch_shape=(1, 32, 32, 32),
    patch_out_shape=(3, 32, 32, 32),
    extraction_step=(16, 16, 16),
    normalize='image',
    activation=torch.nn.Softmax(dim=1))

dataset_test = nl.parallel_load(load_func=load_case, arguments=test_case_paths, num_workers=12)
predicted_segmentations  = []
for n, case in enumerate(dataset_test):
    # Predict image with the predictor and store
    print("Segmenting image {}".format(n))
    tissue_probabilities = predictor.predict(guerrero_trained, case['t1'])
    nl.save_nifti(
        filepath=results_path + 'test_{}_segmented.nii.gz'.format(n),
        volume=tissue_probabilities, # Remove single channel dimension for storage
        reference=case['nifti'],
        channel_handling='split')

    # Add the input and output images to a list for metrics computation
    predicted_segmentations.append(tissue_probabilities)

# Finally, compute evaluation metrics for all test images and store in disk
autoencoder_metrics = nl.metrics.compute_metrics(
    outputs=[np.argmax(pred_seg, axis=0) for pred_seg in predicted_segmentations],
    targets=[np.argmax(case['seg'], axis=0) for case in dataset_test],
    metrics={
        'dsc': nl.metrics.dsc,
        'acc': nl.metrics.accuracy})
nl.save_to_csv(metrics_path + 'autoencoder_metrics.csv', autoencoder_metrics)