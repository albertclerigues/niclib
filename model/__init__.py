import numpy as np

from .guerrero import *

def get_num_trainable_parameters(model_):
    model_parameters = filter(lambda p: p.requires_grad, model_.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    return nparams
