################
Data generators
################

The data generators in this library use an underlying data set of type ``torch.utils.data.Dataset``
which is the actual data container. These data sets are then used to generate torch generators of type
``torch.utils.data.dataloader.DataLoader`` which determine the batch size and shuffling.

The basic torch dataset objects are wrapped as ``Set`` objects in this library (although any object inheriting from ``torch.utils.data.Dataset`` can be used to make a generator.). Two basic Set objects are included
(``ListSet``, ``FunctionSet``) and two to chain and zip other Set (``ZipSet``, ``ChainSet``).
Additionally, ``PatchSet`` provides an easy way to create a patch set from a list of volumes.
The patch sampling locations can be controlled by either using a predefined sampling (``UniformSampling``,
``BalancedSampling``) or by directly providing the patch centers.

.. automodule:: niclib.generator

.. autofunction:: make_generator
.. autoclass:: ListSet
.. autoclass:: FunctionSet
.. autoclass:: ZipSet
.. autoclass:: ChainSet


Patch Generation
=====================

.. autoclass:: PatchSet
.. autoclass:: UniformSampling
.. autoclass:: BalancedSampling
.. autoclass:: PatchSampling
