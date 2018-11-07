import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from niclib.network.loss_functions import *


class EarlyStoppingTrain:
    def __init__(self, loss_func, optimizer, batch_size, max_epochs=100, eval_metrics=None, early_stopping_metric='loss',
                 early_stopping_patience=1, print_interval=5):
        # Training variables # TODO put device as option
        self.device = torch.device('cuda')

        # Training config
        self.bs = batch_size
        self.optimizer_obj = optimizer
        self.train_loss_func = loss_func
        self.max_epochs = max_epochs

        # Test config
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
                best_metric.update(epoch=self.current_epoch, value=monitored_metric_value)
                print(' (best)', sep='')

                torch.save(model, checkpoint_filepath)
            else:
                print('')

            if self.current_epoch - best_metric['epoch'] >= self.patience:
                print("Training finished\n")
                break

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

    def train_epoch(self, model, optimizer, train_gen):
        model.train()

        avg_loss = 0
        for batch_idx, (x, y) in enumerate(train_gen):
            x, y = x.to(self.device), y.to(self.device)

            y_pred = model(x)
            loss = self.train_loss_func(y_pred, y)
            avg_loss += loss.item()

            optimizer.zero_grad()  # Reset accumulated gradients
            loss.backward()  # Auto gradient loss
            optimizer.step()  # Backpropagate the loss

            if batch_idx % self.print_interval is 0:
                self._printProgress(batch_idx, len(train_gen), self.current_epoch, 'train', loss.item())

        self._printProgress(len(train_gen), len(train_gen), self.current_epoch, 'train',
                            avg_loss / len(train_gen))

    def _printProgress(self, batch_num, total_batches, epoch_num, phase, loss):
        length, fill = 30, '='
        percent = ("{0:.1f}").format(100 * (batch_num / float(total_batches)))
        filledLength = int(length * batch_num // total_batches)
        bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

        print('\r Epoch {}/{} - [{}] {}/{} ({}%) - {}_loss={:.4f}'.format(
            epoch_num, self.max_epochs, bar, batch_num, total_batches, percent, phase, loss),
            end='\r' if batch_num != total_batches else '')
        sys.stdout.flush()
