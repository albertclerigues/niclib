import csv
import datetime
import math
import os
import sys
import time
import warnings
from concurrent.futures.thread import ThreadPoolExecutor

from typing import List, Any

import nibabel as nib
import numpy as np
import torch
from art import tprint


##################################################################
## Reproducibility and determinism
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device='cuda'

#################################################################
## niclib namespace imports
## from .utils import *
## from . import net, generators, data, metrics

from .utils import *
from . import net, generators, data, metrics

