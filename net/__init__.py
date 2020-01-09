import warnings

import torch
import numpy as np

from . import test
from . import train

def get_num_trainable_parameters(model):
    """Returns the number of trainable parameters in the given model."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    return nparams
