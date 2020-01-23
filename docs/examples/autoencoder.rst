Autoencoder example
========================

In this example we will train and use a patch-based autoencoder CNN for T1 MRI images.
An autoencoder is an encoder-decoder network trained to reconstruct its input after encoding it to a lower-dimensional
representation (latent space). This can be useful to remove noise or find a lower-dimensional latent space with high-level features.

In this example we make use of the campinas-mini dataset, a small set of 30 images from the larger
`Calgary-Campinas Public Brain MR Dataset <https://sites.google.com/view/calgary-campinas-dataset/home>`_ which
provides 359 cases featuring an MRI T1 image and a brain tissue mask. Additionally, we include and additional
three class tissue segmentation (CSF, GM, WM) made using FSL FAST and subcortical structure segmentation made using
FSL FIRST.

`Download campinas-mini dataset (531 MB) <https://nic.udg.edu/niclib/datasets/campinas-mini.zip>`_


1. Dataset loading and preprocessing
*****************************************

To start we load useful libraries and declare and create relevant output paths:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 1-12

Then we load the campinas-mini dataset:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 15-27

We do so by using the ``parallel_load`` function which needs a list with the absolute paths of each case
and the ``load_case`` function which will load the T1 image given the case's folder path and put it with dimensions as (CH, X, Y, Z).
Additionally, ``load_case`` also preprocesses the loaded image by adjusting the intensity range to the
interval [0, 1] after clipping bright outlier intensities.
Finally we split the dataset images into two lists, one with training images and the other with the validation ones.



2. Training and validation patch generators
*******************************************
We begin by generating two PatchSet objects (later used to create the generator), one with training images and the other with validation images:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 30-43

The PatchSets are created with a patch shape of ``(32, 32, 32)``, including all channels by default, which will return patches
of shape ``(1, 32, 32, 32)`` since we have single channel T1 images. These will be sampled uniformly from all the
volume with a step of (16, 16, 16) and then resampled to have exactly 2000 patches per image.
We specify ``normalize='none'`` since we already did our own custom normalization to the range [0, 1]
when loading the images.

Then we will generate the actual patch generator objects, specifying the desired batch_size and shuffling:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 45-49

Since for an autoencoder the target is the input itself we will create a generator from a ``ZipSet`` that will zip the
patches from the provided PatchSets (in this case, it is the same twice). When accessing the index ``i``of this patch
set (i.e. ``train_patch_set[i]``), it will return a tuple of two patches ``(input, target)`` for the training procedure.
The generator itself will access several indexes of the given ZipSet for 32
different samples (32 different tuple of patches ``(input, target)``) to generate a batch.
Before returning, the generated batch is *collated* and the 32 sampled patches are merged to a single tuple
``(input, target)`` where both input and target are torch tensors with dimensions (BS, CH, X, Y, Z).
We also set the generator to randomly shuffle the patches each time it is iterated, which is very important for correct network
training. In contrast to the PatchSet the generator object cannot be indexed, it can only be iterated with a for loop.
This is because it is designed for beginning to end multithread fast iteration, not random index access.

3. Network training
*****************************************

Now comes the deep learning part! First we instance a predefined uResNet network which is initialized with random weights:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 52-53

The number of both input and output channels is set to 1 since we have a single modality input and a single modality target.
The original uResNet model is defined an encoder-decoder network with skip connections, however, we remove these by setting
``skip_connections=False`` since they would provide a shortcut for reconstruction that would avoid the learning of an
appropiate latent space of high-level features. Finally, the ``torch.nn.Sigmoid()`` activation will map the network
outputs to the range [0, 1] to fit the range of the given targets.


Afterwards we create a Trainer specifying all training hyper-parameters:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 55-67

In this case we will use the Mean Squared Error Loss
(``torch.nn.MSELoss()``) with the Adadelta optimizer with the default learning rate of 1.0.
During training and validation we will check the model is improving by computing the Structural Similarity (a perceptual
similarity measure) and the Peak Signal to Noise ratio.

We also specify the use of 4 training plugins:

- ``nl.net.train.ProgressBar`` will print a progress bar as well as the average values of train and val metrics for each epoch.

- ``nl.net.train.ModelCheckpoint`` will store the trained model when a global minimum validation loss is reached.

- ``nl.net.train.EarlyStopping`` will interrupt the training when the validation loss has not reached a new global minimum for several consecutive epochs (in this case the patience is 5 epochs).

- ``nl.net.train.Logger`` will store the average metrics for each epoch, as well as the time that epoch finished.

Then we begin the training procedure using the instanced model and the training and validation generators:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 69-70

After the training is either finished or stopped by the EarlyStopping plugin, we load the trained model from disk.

4. Testing and evaluation metrics
*****************************************

Finally we will use trained model to autoencode the three test images (which we reserved in the beginning) and compute evaluation metrics.

We begin by creating the ``PatchTester`` object which will split the image in patches, forward pass each one
through the trained network and recombine the outputs by averaging to form the output image:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 73-78

The input and output patch shapes in this case have both shape (1, 32, 32, 32) and will be sampled at steps of (16, 16, 16).
This will ensure a 50% overlap between patches that will avoid border effects and smooth out any artifacts.
The normalization is set to 'none' since we already did our own custom normalization at loading time.
The output activation is also set to None since the trained network already includes an output Sigmoid activation that will ensure
the outputs are in the range [0, 1] (the same as the targets).

Then we begin the testing procedure where we will load the test image, forward pass it by the network (with the PatchTester)
and save in the disk (with the original image as reference nifti to copy its headers).
Additionally, we store both the input and autoencoded images in a list to compute evaluation metrics later:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 80-94

Finally, after all images have been tested, we compute the evaluation metrics and store them as a .csv file:

.. literalinclude:: example_autoencoder.py
    :language: python
    :lines: 97-105

Full source code
*****************************************

.. literalinclude:: example_autoencoder.py
   :language: python
   :linenos: