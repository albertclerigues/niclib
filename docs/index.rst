.. niclib documentation master file, created by
   sphinx-quickstart on Tue May 14 13:58:28 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##################################
niclib docs
##################################

niclib is a utility library to ease the development of neuroimaging pipelines using deep learning.
This library attempts to offers a simple interface to commonly used functions and procedures when dealing with
file storage, data operations, data and patch generators, network training, loss functions and evaluation metrics.

This library does't aim to be a complete replacement of other deep learning or neuroimaging libraries.
If more advanced functionality is required, we encourage the user to use the provided code as a starting point to
implement their own needs.

Installation
************

niclib only requirement is Python 3.6, the library can then be installed using pip:

.. code-block:: none

   pip3 install https://nic.udg.edu/niclib/wheels/niclib-0.5.1b-py3-none-any.whl

.. toctree::
   :maxdepth: 2
   :caption: Module documentation

   niclib utils <builtins>
   Data operations <data>
   Data generators <generator>
   Predefined models <models>
   Networks <network>
   Loss functions <loss>
   Evaluation metrics <metrics>
.. :caption: Contents:

.. toctree::
    :maxdepth: 2
    :caption: Tutorials and examples
    :glob:

    examples/*





Indices and tables
******************************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



