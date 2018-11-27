import numpy as np
import torch

class NICExperiment:
    def __init__(self, name, run):
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

class ExperimentManager:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

        self.checkpoints_dir = 'checkpoints/'
        self.log_dir = 'log/'
        self.results_dir = 'results/'
        self.metrics_dir = 'metrics/'

        # Ensure reproducibility in results
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

