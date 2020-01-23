##############
niclib utils
##############

.. automodule:: niclib

- :ref:`path`
- :ref:`list`
- :ref:`time`
- :ref:`print`
- :ref:`io`



.. _path:

Path utils
**********
.. autofunction:: make_dir
.. autofunction:: get_filename
.. autofunction:: get_base_path
.. autofunction:: remove_extension

.. _list:

List utils
**********
.. autofunction:: resample_list
.. autofunction:: split_list
.. autofunction:: moving_average

.. _time:

Time utils
********************
.. autofunction:: get_timestamp
.. autofunction:: format_time_interval
.. autoclass:: RemainingTimeEstimator
    :members:

.. _print:

Print utils
********************
.. autofunction:: print_progress_bar
.. autofunction:: print_big

.. _io:

I/O utils
******************
.. autofunction:: parallel_load
.. autofunction:: save_nifti
.. autofunction:: save_to_csv
.. autofunction:: load_from_csv
