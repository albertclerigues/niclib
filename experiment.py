import numpy as np
import torch

# TODO implement as abstract NICexperiment class
class ExperimentManager:
    def __init__(self, experiment_name, checkpoints_dir='checkpoints/', log_dir='log/', results_dir='results/', metrics_dir='metrics/'):
        self.experiment_name = experiment_name

        self.checkpoints_dir = checkpoints_dir
        self.log_dir = log_dir
        self.results_dir = results_dir
        self.metrics_dir = metrics_dir

        # Ensure reproducibility in results
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False