import torch
import os
import copy

from niclib.network.loss_functions import dice_loss
from niclib.network.training import EarlyStoppingTrain

from niclib.utils import *

class SimpleCrossvalidation:
    def __init__(self, model, images, num_folds, trainer, train_instr_gen, val_instr_gen, checkpoint_pathfile):
        self.model = model
        self.images = images
        self.num_folds = num_folds
        if checkpoint_pathfile.endswith('.pt'):
            checkpoint_pathfile, _ = os.path.splitext(checkpoint_pathfile)
        self.checkpoint_pathfile = checkpoint_pathfile + '_{}_to_{}.pt'

        self.train_instr_gen = train_instr_gen
        self.val_instr_gen = val_instr_gen

        assert isinstance(trainer, EarlyStoppingTrain)
        self.trainer = trainer

    def run_crossval(self):
        for fold_idx in range(self.num_folds):
            start_idx_val, stop_idx_val = get_crossval_indexes(
                images=self.images, fold_idx=fold_idx, num_folds=self.num_folds)

            print("Building training generator...")
            train_gen = self.train_instr_gen.get_patch_generator(images=self.images[:start_idx_val] + self.images[stop_idx_val:])
            print("Building validation generator...")
            val_gen = self.val_instr_gen.get_patch_generator(images=self.images[start_idx_val:stop_idx_val])

            self.trainer.train(self.model, train_gen, val_gen, self.checkpoint_pathfile.format(start_idx_val, stop_idx_val))

