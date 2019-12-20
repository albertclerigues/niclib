import warnings

import torch

from . import test
from . import train


# def save_model(filepath, model):
#     print("Saving model: {}".format(filepath))
#     torch.save(model, filepath)
#
# def load_model(filepath, base_model=None):
#     """
#     Load a saved torch.nn.Module instance. # TODO finish
#     :param filepath:
#     :param base_model:
#     :return:
#     """
#
#     if base_model is None:
#         print("Loading model {}".format(filepath))
#         model = torch.load(filepath)
#     else:
#         print("Loading weights {}".format(filepath))
#         base_model.load_state_dict(torch.load(filepath))
#         model = base_model
#
#     return model
#
#
# def save_checkpoint(filepath, epoch, model, optimizer, loss):
#     assert isinstance(model, torch.nn.Module)
#     assert isinstance(optimizer, torch.optim.Optimizer)
#
#     checkpoint = {
#         'epoch': epoch,
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'loss': loss,
#     }
#
#     print("Saving checkpoint at {}".format(filepath))
#     torch.save(checkpoint, filepath)
#
#
# def load_checkpoint(filepath, model=None, optimizer=None):
#     checkpoint = torch.load(filepath)
#
#     if model is not None:
#         checkpoint['model'] = model.load_state_dict(checkpoint['model'])
#
#     if optimizer is not None:
#         checkpoint['optimizer'] = optimizer.load_state_dict(checkpoint['optimizer'])
#
#     return checkpoint
#
#
# def compute_gradient_norm(model):
#     total_norm = 0.0
#     for p in model.parameters():
#         param_norm = p.grad.data.norm(2)
#         total_norm += param_norm.item() ** 2
#
#         if param_norm > 100.0:
#             warnings.warn("SUPER HIGH GRADIENT! > 100!!!!", RuntimeWarning)
#
#     total_norm = total_norm ** (1. / 2)
#     return total_norm