import torch
import os
import copy

from niclib.network.loss_functions import dice_loss
from niclib.network.training import EarlyStoppingTrain

from niclib.network.generator import InstructionGenerator

from niclib.utils import *

class SimpleCrossvalidation:
    def __init__(self, model, images, num_folds, trainer, train_instr_gen, val_instr_gen, checkpoint_pathfile):
        self.model = model
        self.images = images
        self.num_folds = num_folds
        if checkpoint_pathfile.endswith('.pt'):
            checkpoint_pathfile, _ = os.path.splitext(checkpoint_pathfile)
        self.checkpoint_pathfile = checkpoint_pathfile + '_{}_to_{}.pt'

        assert isinstance(train_instr_gen, InstructionGenerator)
        self.train_instr_gen = train_instr_gen
        assert isinstance(val_instr_gen, InstructionGenerator)
        self.val_instr_gen = val_instr_gen

        assert isinstance(trainer, EarlyStoppingTrain)
        self.trainer = trainer

    def run_crossval(self):
        print("\n" + "=" * 75 + "\n Running {}-fold crossvalidation \n".format(self.num_folds) + "=" * 75 + "\n", sep='')

        for fold_idx in range(self.num_folds):
            start_idx_val, stop_idx_val = get_crossval_indexes(
                images=self.images, fold_idx=fold_idx, num_folds=self.num_folds)

            print("\n  Running fold {} - val images {} to {} \n".format(fold_idx, start_idx_val, stop_idx_val),sep='')

            print("Building training generator...")
            train_gen = self.train_instr_gen.build_patch_generator(images=self.images[:start_idx_val] + self.images[stop_idx_val:])
            print("Building validation generator...")
            val_gen = self.val_instr_gen.build_patch_generator(images=self.images[start_idx_val:stop_idx_val])

            print("")
            self.trainer.train(self.model, train_gen, val_gen, self.checkpoint_pathfile.format(start_idx_val, stop_idx_val))

