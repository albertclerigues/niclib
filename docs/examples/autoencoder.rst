Autoencoder example
========================

In this example we will train and use a patch-based autoencoder CNN for T1 MRI images.
This can be useful to remove noise or find a lower-dimensional latent space with high-level features.


In this example we make use of the campinas-mini dataset, a small set of 30 images from the larger
`Calgary-Campinas Public Brain MR Dataset <https://sites.google.com/view/calgary-campinas-dataset/home>`_ which
provides 359 cases featuring an MRI T1 image and a brain tissue mask. Additionally, we include and additional
three class tissue segmentation (CSF, GM, WM) made using FSL FAST and subcortical structure segmentation made using
FSL FIRST.

`Download campinas-mini dataset (531 MB) <https://nic.udg.edu/niclib/datasets/campinas-mini.zip>`_

To start we load useful libraries:

.. literalinclude:: example_autoencoder.py
   :language: python
   :lines: 1-5


Full source code of the autoencoder example:

.. literalinclude:: example_autoencoder.py
   :language: python
   :linenos: