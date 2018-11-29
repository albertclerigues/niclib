import os
import numpy as np
import torch

from niclib.utils import *

class NICExperiment:
    def __init__(self, name, run, run_folder=False):
        self.exp_name = name
        self.run_name = run

        self.checkpoints_dir = 'checkpoints/'
        self.log_dir = 'log/'
        self.results_dir = 'results/'
        self.metrics_dir = 'metrics/'

        # Ensure reproducibility in results
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_checkpoint_filename(self):
        return os.path.join(self.checkpoints_dir, '{}.pt'.format(self.run_name))

    def get_log_filename(self, timestamped=True):
        log_filename = '{}.csv'.format(self.run_name)
        if timestamped:
            log_filename = get_formatted_timedate() + '_' + log_filename
        return os.path.join(self.log_dir, log_filename)

    def get_result_filename(self):

        return os.path.join(self.results_dir, '{}/'.format(run_name))