Segmentation example
========================

In this example we make use of the campinas-mini dataset, a small set of 30 images from the larger
`Calgary-Campinas Public Brain MR Dataset <https://sites.google.com/view/calgary-campinas-dataset/home>`_ which
provides 359 cases featuring an MRI T1 image and a brain tissue mask. Additionally, we include and additional
three class tissue segmentation (CSF, GM, WM) made using FSL FAST and subcortical structure segmentation made using
FSL FIRST.

`Download campinas-mini dataset (531 MB) <https://nic.udg.edu/niclib/datasets/campinas-mini.zip>`_



1. Dataset loading and preprocessing
*****************************************

To start we load useful libraries and declare and create relevant output paths:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 1-13

Then we load the campinas-mini dataset:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 16-28

We do so by using the ``parallel_load`` function which needs a list with the absolute paths of each case
and the ``load_case`` function which will load both the T1 image and tissue probabilities (with dimensions as (CH, X, Y, Z)) given the case's folder path.
Finally we split the dataset images into two lists, one with training images and the other with the validation ones.


2. Training and validation patch generators
*******************************************
We begin by generating two Set objects (later used to create the generator), one with training images and the other with validation images:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 31-60


In this case, each set is formed by zipping two PatchSet, one for the input T1 patches, and other for the target tissue probability patches.
The PatchSets are created with a patch shape of ``(32, 32, 32)``, including all channels by default.
Which for the input will return patches of shape ``(1, 32, 32, 32)`` (since we have single channel T1 images) and ``(3, 32, 32, 32)``
for the target tissue probabilities (since we have three tissue channels).
These will be sampled in a balanced way (same number of patches centered in each of the three labels) from all the volume
and then resampled to have exactly 1000 patches per image.
We specify ``normalize='image'`` to have inputs sampled from an image with zero mean and unit variance. This kind of normalization
makes the input range of the network a lot more uniform and image independent so that the network can achieve better accuracy.
When accessing the index ``i`` of this patch
set (i.e. ``train_patch_set[i]``), it will return a tuple of two patches ``(input, target)``.

Then we will generate the actual patch generator objects, specifying the desired batch_size and shuffling:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 62-64

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

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 67-68


The number of input channels is set to 1, since we have a single modality input, and the number output channels is set
to 3, since we will output the *logits* for each of the three classes.

Then we define the loss and metrics functions to use during the training procedure. In this case, the functions are
not defined to deal with the exact format of our network output (3 channel logits) and our target (3 channel probabilities),
so we will use the ``LossWrapper`` object from niclib to perform the adaptation:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 71-72

- ``wrapped_crossentropy_loss`` receives 3 channel logits for the ``output`` argument, which is exactly our network output so no need to modify it. However, it needs single channel labels for the target while we have three channel logits. To obtain the target in the desired format we first obtain probabilities using ``F.softmax``, then we obtain labels by doing ``torch.argmax`` and finally we set the data type to ``long`` as required by the function.

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 73-76

- ``accuracy`` and ``dsc`` both need single channel labels for the output and target arguments to return the correct value. To adapt the target probabilities we just need to use ``torch.argmax``, while for the output logits we do a ``F.softmax`` followed by ``torch.argmax`` to obtain the desired single channel labels.

.. note::
    The dimensions of the tensors for losses and metrics used for the output of a generator are (BS, CH, X, Y, Z).
    This means that the channel dimension is located at index 1 (and not at index 0) for the wrapped functions used inside trainer.

Afterwards we create a Trainer specifying all training hyper-parameters:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 78-90

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

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 92-93

After the training is either finished or stopped by the EarlyStopping plugin, we load the trained model from disk.

4. Testing and evaluation metrics
*****************************************

Finally we will use trained model to autoencode the three test images (which we reserved in the beginning) and compute evaluation metrics.

We begin by creating the ``PatchTester`` object which will split the image in patches, forward pass each one
through the trained network and recombine the outputs by averaging to form the output image:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 96-101

The input and output patch shapes in this case have both shape (1, 32, 32, 32) and will be sampled at steps of (16, 16, 16).
This will ensure a 50% overlap between patches that will avoid border effects and smooth out any artifacts.
The normalization is set to ``'none'`` since we already did our own custom normalization at loading time.
The output activation is set to a ``nn.Softmax`` in the channel dimension since the trained network was defined without
any activation (directly outputting logits) and we want to obtain a tissue probability distribution.

Then we load the dataset using the previously declared ``load_case`` function and begin the testing procedure.
First we will forward pass the image through the network using the ``PatchTester`` object, and then
and save it in the disk (with the original image as reference nifti to copy its headers).
Additionally, we store the output probabilities in a list to compute evaluation metrics later:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 103-116

Finally, after all images have been tested, we compute the evaluation metrics and store them as a .csv file:

.. literalinclude:: example_segmentation.py
    :language: python
    :lines: 119-125

In this case, since both outputs and targets are three channel probabilities, we compute the argmax of each image before computing the metrics.

Full source code
*****************************************

.. literalinclude:: example_segmentation.py
   :language: python
   :linenos:


