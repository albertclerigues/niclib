##############
Built-in utils
##############

.. automodule:: niclib

.. py:data:: device

	(torch.device) Module-level singleton variable containing a torch device (i.e. torch.device('cpu'), torch.device('cuda'), torch.device('cuda:0') which is shared across all imports. This will be the device that niclib will use unless specified otherwise. (default: torch.device('cuda')


Path utils
**********
.. autofunction:: make_dir
.. autofunction:: get_filename
.. autofunction:: get_base_path
.. autofunction:: remove_extension

List utils
**********
.. autofunction:: resample_list
.. autofunction:: split_list
.. autofunction:: moving_average

Time and print utils
********************
.. autofunction:: get_timestamp
.. autofunction:: format_time_interval
.. autoclass:: RemainingTimeEstimator
	:members:
.. autofunction:: print_progress_bar
.. autofunction:: print_big

Input/Output utils
******************
.. autofunction:: parallel_load
.. autofunction:: save_nifti
.. autofunction:: save_to_csv
.. autofunction:: load_from_csv





.. :members:
   :undoc-members:
