import os
import sys
import string
from abc import ABC, abstractmethod
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from niclib.network.loss_functions import *


class Training(ABC): # TODO
    def __init__(self):
        pass

class EarlyStoppingTrain:
    def __init__(self, loss_func, optimizer, batch_size, max_epochs=100, eval_metrics=None, early_stopping_metric='loss',
                 early_stopping_patience=1, print_interval=5, device=torch.device('cuda'), load_trained=False):
        # Training config
        self.device = device
        self.bs = batch_size
        self.optimizer_obj = optimizer
        self.train_loss_func = loss_func
        self.max_epochs = max_epochs
        self.load_trained = load_trained

        # Testing config
        self.eval_functions = {'loss': loss_func}
        if eval_metrics is not None:
            self.eval_functions.update(eval_metrics)
        self.early_stopping_metric = 'loss' if early_stopping_metric is None else early_stopping_metric
        self.patience = early_stopping_patience

        # Meta-training variables
        self.current_epoch = -1
        self.print_interval = print_interval

        assert 0 < self.patience < self.max_epochs
        assert self.early_stopping_metric in self.eval_functions.keys()

    def train(self, model, train_gen, val_gen, checkpoint_filepath):
        if self.load_trained and os.path.exists(checkpoint_filepath):
            print("Found trained model {}".format(checkpoint_filepath))
            return

        print("Training model for {} epochs".format(self.max_epochs))

        model = model.to(self.device)
        optimizer = self.optimizer_obj(model.parameters())
        self.current_epoch = -1

        best_metric = dict(epoch=-1, value=sys.float_info.max)
        for epoch_num in range(self.max_epochs):
            self.current_epoch += 1

            self.train_epoch(model, optimizer, train_gen)
            epoch_test_metrics = self.test_epoch(model, val_gen)

            monitored_metric_value = epoch_test_metrics[self.early_stopping_metric]
            if monitored_metric_value < best_metric['value']:
                print(' (best)', sep='')
                best_metric.update(epoch=self.current_epoch, value=monitored_metric_value)
                torch.save(model, checkpoint_filepath)
            else:
                print('')

            if self.current_epoch - best_metric['epoch'] >= self.patience:
                print("Training finished\n")
                break

    def train_epoch(self, model, optimizer, train_gen):
        model.train()

        avg_loss = 0
        eta_estimator = ElapsedTimeEstimator(len(train_gen))
        for batch_idx, (x, y) in enumerate(train_gen):
            x, y = x.to(self.device), y.to(self.device)

            y_pred = model(x)
            loss = self.train_loss_func(y_pred, y)
            avg_loss += loss.item()

            optimizer.zero_grad()  # Reset accumulated gradients
            loss.backward()  # Auto gradient loss
            optimizer.step()  # Backpropagate the loss

            if batch_idx % self.print_interval is 0 and batch_idx > 0:
                eta = eta_estimator.update(batch_idx)
                self._printProgress(batch_idx, len(train_gen), self.current_epoch, eta, 'train', avg_loss / batch_idx)

        self._printProgress(len(train_gen), len(train_gen), self.current_epoch, eta_estimator.get_elapsed_time(), 'train',
                            avg_loss / len(train_gen))

    def test_epoch(self, model, val_gen):
        model.eval()
        test_metrics = dict.fromkeys(self.eval_functions.keys(), 0.0)

        with torch.no_grad():  # Turns off autograd (faster exec)
            for batch_idx, (x, y) in enumerate(val_gen):
                x, y = x.to(self.device), y.to(self.device)
                y_pred = model(x)

                for k, eval_func in self.eval_functions.items():
                    test_metrics[k] += eval_func(y_pred, y)

        # Compute average metrics
        for k, v in test_metrics.items():
            test_metrics[k] = float(v / len(val_gen))

        # Print average validation metrics
        for k, v in test_metrics.items():
            indicator = '*' if self.early_stopping_metric is k else ''
            print(' - {}val_{}={:.4f}'.format(indicator, k, v), end='')

        return test_metrics

    def _printProgress(self, batch_num, total_batches, epoch_num, eta, phase, loss):
        length, fill = 30, '='
        percent = "{0:.1f}".format(100 * (batch_num / float(total_batches)))
        filledLength = int(length * batch_num // total_batches)
        bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

        print('\r Epoch {}/{} - [{}] {}/{} ({}%) ETA: {} - {}_loss={:.4f}'.format(
            epoch_num, self.max_epochs, bar, batch_num, total_batches, percent, eta, phase, loss),
            end='\r' if batch_num != total_batches else '')
        sys.stdout.flush()


class ElapsedTimeEstimator:
    def __init__(self, total_iters, update_weight=0.01):
        self.total_time_estimation = None

        self.start_time = time.time()

        self.last_iter = {'num': 0, 'time':time.time()}
        self.total_iters = total_iters
        self.update_weight = update_weight

    def update(self, current_iter_num):
        current_eta = None
        current_iter = {'num': current_iter_num, 'time':time.time()}

        if current_iter['num'] > self.last_iter['num']:
            iters_between = current_iter['num'] - self.last_iter['num']
            time_between = current_iter['time'] - self.last_iter['time']

            current_time_estimation = (time_between * self.total_iters)/iters_between

            if self.total_time_estimation is not None:
                self.total_time_estimation = (current_time_estimation * self.update_weight) + \
                                             (self.total_time_estimation * (1 - self.update_weight))
            else:
                self.total_time_estimation = current_time_estimation

            time_since_start = (current_iter['time'] - self.start_time)
            current_eta = self.total_time_estimation - time_since_start

        # End, set last iter as current
        self.last_iter = current_iter

        # Return formatted eta in hours, minutes, seconds
        if current_eta is not None:
            formatted_time = str(datetime.timedelta(seconds=current_eta)).split('.')[0]

            #if int(formatted_time.split(':')[0]) is 0:
            #    formatted_time = formatted_time[formatted_time.index(':') + 1:]

            return formatted_time
        else:
            return '??:??'

    def get_elapsed_time(self):
        return str(datetime.timedelta(seconds=time.time() - self.start_time)).split('.')[0]
