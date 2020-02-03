import copy
import os
import sys
import threading
import time
from abc import ABC

import torch
from torch import nn

from ..utils import RemainingTimeEstimator, remove_extension, get_timestamp, save_to_csv


class Trainer:
    """Class for torch Module training using training and validation generators with basic metric support.
    It also has support for plugins, a way to embed any functionality in the training procedure.

    :param int max_epochs: maximum number of epochs to train. Training can be interrupted with a plugin by setting the
        attribute `keep_training` to False.
    :param torch.nn.Module loss_func: loss function.
    :param optimizer: torch.optim.Optimizer derived optimizer.
    :param dict train_metrics: dictionary with the desired metrics as key value pairs specifiying the metric name and
        the function object (callable) respectively. The loss function is automatically included under the key 'loss'.
    :param dict val_metrics: same as `train_metrics`.
    :param list plugins: List of plugin objects (inheriting from ``TrainerPlugin``).
    :param str device: torch torch_device i.e. 'cpu', 'cuda', 'cuda:0'...
    :param bool multigpu: (default: False) if True, it instances the model as a DataParallel model before training.

    Plugins are made by inheriting from the base class :py:class:`~niclib.net.train.TrainerPlugin` and overriding
    the inherited functions to implement the desired functionality at specific points during the training.
    """

    def __init__(self, max_epochs, loss_func, optimizer, optimizer_opts=None, train_metrics=None, val_metrics=None, plugins=None, device='cuda', multigpu=False):
        assert all([isinstance(plugin, TrainerPlugin) for plugin in plugins])
        if train_metrics is not None:
            assert isinstance(train_metrics, dict) and all([callable(m) for m in train_metrics.values()])
        if  val_metrics is not None:
            assert isinstance(val_metrics, dict) and all([callable(m) for m in val_metrics.values()])

        # Basic training variables
        self.device = device
        self.use_multigpu = multigpu

        self.max_epochs = max_epochs
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.optimizer_opts = optimizer_opts if optimizer_opts is not None else {}

        # Metric functions
        self.train_metric_funcs = {'loss': copy.copy(loss_func)}
        if train_metrics is not None:
            self.train_metric_funcs.update(train_metrics)

        self.val_metric_funcs = {'loss': copy.copy(loss_func)}
        if val_metrics is not None:
            self.val_metric_funcs.update(val_metrics)

        # Plugin functionality
        self.plugins = plugins # TODO avoid empty function calls (function call overhead)
        [plugin.on_init(self) for plugin in self.plugins]

        # Runtime variables
        self.keep_training = True
        self.model = None
        self.model_optimizer = None
        self.train_gen = None
        self.val_gen = None
        self.x, self.y, self.y_pred = None, None, None
        self.loss = None

        self.train_metrics = dict()
        self.val_metrics = dict()

        self.epoch_start = 0


    def train(self, model, train_gen, val_gen, checkpoint=None):
        """
        Trains a given model using the provided :class:`torch.data.DataLoader` generator.

        :param torch.nn.Module model: the model to train.
        :param torch.utils.data.DataLoader train_gen: An iterator returning (x, y) pairs for training.
        :param torch.utils.data.DataLoader val_gen: An iterator returning (x, y) pairs for validation.
        :param dict checkpoint: (optional) checkpoint that can include 'epoch', 'model', 'optimizer' or 'loss'
        :return: None, after training you should load the stored model from disk using torch.load()
        """

        # Store given arguments in runtime variables to be available to plugins
        self.model, self.train_gen, self.val_gen = model, train_gen, val_gen
        self.model_optimizer = self.optimizer(self.model.parameters(), **self.optimizer_opts)

        if checkpoint is not None:
            if 'epoch' in checkpoint:
                self.epoch_start = checkpoint['epoch']
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                self.model_optimizer.load_state_dict(checkpoint['optimizer'])
            if 'loss' in checkpoint:
                self.loss_func = checkpoint['loss']

        self.model = model.to(self.device)
        if self.use_multigpu:
            print("Using multigpu for training")
            self.model = nn.DataParallel(self.model)

        print("Training model for {} epochs".format(self.max_epochs))
        [plugin.on_train_start(self) for plugin in self.plugins]

        for epoch_num in range(self.epoch_start, self.max_epochs):
            if self.keep_training is False:
                break

            # Train current epoch
            [plugin.on_train_epoch_start(self, epoch_num) for plugin in self.plugins]
            self.train_epoch()
            [plugin.on_train_epoch_end(self, epoch_num) for plugin in self.plugins]

            # Validate on validation set
            [plugin.on_val_epoch_start(self, epoch_num) for plugin in self.plugins]
            self.validate_epoch()
            [plugin.on_val_epoch_end(self, epoch_num) for plugin in self.plugins]

        print("Training finished\n")
        [plugin.on_train_end(self) for plugin in self.plugins]

    def train_epoch(self):
        self.model.train() # Set the model in training mode (for correct dropout, batchnorm, etc.)

        # Prepare the metrics data structure
        self.train_metrics = dict()
        for k in self.train_metric_funcs.keys():
            self.train_metrics['train_{}'.format(k)] = 0.0


        for batch_idx, (x, y) in enumerate(self.train_gen):
            [plugin.on_train_batch_start(self, batch_idx) for plugin in self.plugins]

            self.x, self.y = self._to_device(x, y)

            self.model_optimizer.zero_grad()  # Reset accumulated gradients
            self.y_pred = self.model(self.x)  # Forward pass
            self.loss = self.loss_func(self.y_pred, self.y)  # Loss computation

            self.loss.backward()  # Compute autograd gradients from loss
            self.model_optimizer.step()  # Update the weights according to optimizer scaled gradients

            for k, eval_func in self.train_metric_funcs.items(): # Update training metrics
                self.train_metrics['train_{}'.format(k)] += eval_func(self.y_pred, self.y).item()

            [plugin.on_train_batch_end(self, batch_idx) for plugin in self.plugins]

        # Compute average metrics for epoch
        for k, v in self.train_metrics.items():
            self.train_metrics[k] = float(v / len(self.train_gen))

    def validate_epoch(self):
        self.val_metrics = dict()
        for k in self.val_metric_funcs.keys():
            self.val_metrics['val_{}'.format(k)] = 0.0

        self.model.eval()
        with torch.no_grad():  # Turns off autograd (faster exec)
            for batch_idx, (x, y) in enumerate(self.val_gen):
                [plugin.on_val_batch_start(self, batch_idx) for plugin in self.plugins]

                self.x, self.y = self._to_device(x, y)
                self.y_pred = self.model(self.x)

                for k, eval_func in self.val_metric_funcs.items():
                    self.val_metrics['val_{}'.format(k)] += eval_func(self.y_pred, self.y)

                [plugin.on_val_batch_end(self, batch_idx) for plugin in self.plugins]

        # Compute average metrics
        for k, v in self.val_metrics.items():
            self.val_metrics[k] = float(v / len(self.val_gen))


    def _to_device(self, x, y=None):

        if isinstance(x, list) or isinstance(x, tuple):
            x = tuple(self._to_device(x_n) for x_n in x)
        else:
            x = x.to(self.device)

        if y is None:
            return x

        if isinstance(y, list) or isinstance(y, tuple):
            y = tuple(self._to_device(y_n) for y_n in y)
        else:
            y = y.to(self.device)

        return x, y



class TrainerPlugin(ABC):
    """
    Abstract Base Class for subclassing training plugins. Subclasses can override the following callbacks:

    * ``on_init(self, trainer)``
    * ``on_train_start(self, trainer)``
    * ``on_train_end(self, trainer)``
    * ``on_train_epoch_start(self, trainer, num_epoch)``
    * ``on_train_epoch_end(self, trainer, num_epoch)``
    * ``on_val_epoch_start(self, trainer, num_epoch)``
    * ``on_val_epoch_end(self, trainer, num_epoch)``
    * ``on_train_batch_start(self, trainer, batch_idx)``
    * ``on_train_batch_end(self, trainer, batch_idx)``
    * ``on_val_batch_start(self, trainer, batch_idx)``
    * ``on_val_batch_end(self, trainer, batch_idx)``

    where:

    :param trainer: :py:class:`Trainer <niclib.net.train.Trainer>` instance that grants access to all its runtime variables (i.e. trainer.keep_training):
    :param num_epoch: current epoch number (starting at 0)
    :param batch_idx: current batch iteration (starting at 0)

    The training runtime variables accesible through the trainer object are:

    :var bool keep_training: control flag to continue or interrupt the training. Checked at the start of each epoch.
    :var torch.nn.Module model: the partially trained model.
    :var torch.optim.Optimizer model_optimizer: the model optimizer.
    :var torch.nn.Module loss_func: The loss function object
    :var torch.utils.data.DataLoader train_gen:
    :var torch.utils.data.DataLoader val_gen:
    :var dict train_metrics:
    :var dict val_metrics:
    :var torch.Tensor x:
    :var torch.Tensor y:
    :var torch.Tensor y_pred:
    :var torch.Tensor loss:

    :Example:

    >>> # Here we define a plugin that will segment and compute metrics on test_images at the end of each epoch
    >>> class EpochEvaluatorPlugin(TrainerPlugin):
    >>>     def __init__(self, test_images, test_targets):
    >>>         super().__init__()
    >>>         self.images, self.targets = test_images, test_targets
    >>>         self.predictor = niclib.net.test.PatchTester(
    >>>             patch_shape=(1, 32, 32, 32),
    >>>             patch_out_shape=(3, 32, 32, 32),
    >>>             extraction_step=(16, 16, 16),
    >>>             normalize='image',
    >>>             activation=torch.nn.Softmax(dim=1))
    >>>
    >>>     def on_val_epoch_end(self, trainer, num_epoch):
    >>>         pred_images = [self.predictor.predict(trainer.model, img) for img in self.images]
    >>>         pred_metrics = niclib.metrics.compute_metrics(
    >>>             outputs=[np.argmax(img, axis=0) for img in pred_images],
    >>>             targets=self.targets,
    >>>             metrics={
    >>>                 'dsc': niclib.metrics.dsc,
    >>>                 'acc': niclib.metrics.accuracy})
    >>>         niclib.save_to_csv('metrics_epoch{}.csv'.format(num_epoch), pred_metrics)
    """
    def on_init(self, trainer):
        pass

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_train_epoch_start(self, trainer, num_epoch):
        pass

    def on_train_epoch_end(self, trainer, num_epoch):
        pass

    def on_val_epoch_start(self, trainer, num_epoch):
        pass

    def on_val_epoch_end(self, trainer, num_epoch):
        pass

    def on_train_batch_start(self, trainer, batch_idx):
        pass

    def on_train_batch_end(self, trainer, batch_idx):
        pass

    def on_val_batch_start(self, trainer, batch_idx):
        pass

    def on_val_batch_end(self, trainer, batch_idx):
        pass


class ModelCheckpoint(TrainerPlugin):
    """
    :param filepath: filepath for checkpoint storage
    :param str save: save mode, either ``best``, ``all`` or ``last``.
    :param metric_name: (only for ``save='best'``) name of the monitored metric to evaluate the model and store.
    :param str mode: either ``min`` or ``max`` of the monitored metric to store the model.
    """

    def __init__(self, filepath, save='best', metric_name='loss', mode='min', min_delta=1e-4):
        assert save in {'best', 'all', 'last'}
        assert mode in {'min', 'max'}

        self.filepath = filepath
        self.save = save
        self.mode = mode
        self.metric_name = metric_name

        self.best_metric = None
        self.min_delta = min_delta

    def on_init(self, trainer):
        self.best_metric = dict(epoch=-1, value=sys.float_info.max)

    def on_val_epoch_end(self, trainer, num_epoch):
        if self.save == 'best':
            monitored_metric_value = trainer.val_metrics['val_{}'.format(self.metric_name)]

            metric_diff = self.best_metric['value'] - monitored_metric_value
            if self.mode == 'max':
                metric_diff *= -1.0

            if metric_diff > self.min_delta:
                self.best_metric.update(epoch=num_epoch, value=monitored_metric_value)
                print("Saving best model at {}".format(self.filepath))
                torch.save(trainer.model, self.filepath)

        elif self.save == 'last':
            torch.save(trainer.model, self.filepath)
        elif self.save == 'all':
            torch.save(trainer.model, remove_extension(self.filepath) + '_{}.pt'.format(num_epoch))
        else:
            raise(ValueError, 'Save mode not valid')


class EarlyStopping(TrainerPlugin):
    """
    Plugin to interrupt the training with the early stopping technique.

    :param filepath: Filepath where the best model will be stored.
    :param str metric_name: monitored metric name. The name has to exist as a key in the val_metrics provided to the :class:`Trainer`.
    :param str mode: either ``min`` or ``max`` of the monitored metric to stop the training.
    :param int patience: Number of epochs to wait since the last best model before interrupting training.
    :param min_delta: minimum change between metric values to consider a new best model.
    """

    def __init__(self, metric_name, mode, patience, min_delta=1e-5):
        self.min_delta = min_delta

        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience

        # Runtime variables
        self.best_metric = None

    def on_init(self, trainer):
        self.best_metric = dict(epoch=-1, value=sys.float_info.max)

    def on_val_epoch_end(self, trainer, num_epoch):
        monitored_metric_value = trainer.val_metrics['val_{}'.format(self.metric_name)]

        metric_diff = self.best_metric['value'] - monitored_metric_value
        if self.mode == 'max':
            metric_diff *= -1.0

        if metric_diff > self.min_delta:
            self.best_metric.update(epoch=num_epoch, value=monitored_metric_value)

        # Interrupt training if metric didnt improve in last 'patience' epochs
        if num_epoch - self.best_metric['epoch'] >= self.patience:
            trainer.keep_training=False


class ProgressBar(TrainerPlugin):
    """Plugin to print a progress bar and metrics during the training procedure.

    :param float print_interval: time in seconds between print updates.
    """
    def __init__(self, print_interval=0.4):
        self.print_interval = print_interval

        # Runtime variables
        self.eta = None
        self.print_flag = True
        self.print_lock = threading.Lock()
        self.print_timer = None

    def on_train_epoch_start(self, trainer, num_epoch):
        print("\nEpoch {}/{}".format(num_epoch, trainer.max_epochs))

        self.eta = RemainingTimeEstimator(len(trainer.train_gen))
        self.print_flag = True
        self.print_timer = None

    def on_train_batch_end(self, trainer, batch_idx):
        # PRINTING LOGIC
        self.print_lock.acquire()
        if self.print_flag:
            # Compute average metrics
            avg_metrics = dict()
            for k, v in trainer.train_metrics.items():
                avg_metrics[k] = float(v / (batch_idx + 1))

            self._printProgressBar(batch_idx, len(trainer.train_gen), self.eta.update(batch_idx + 1), avg_metrics)

            self.print_flag, self.print_timer = False, threading.Timer(self.print_interval, self._setPrintFlag)
            self.print_timer.start()
        self.print_lock.release()

    def on_train_epoch_end(self, trainer, num_epoch):
        self.print_timer.cancel()
        self._printProgressBar(len(trainer.train_gen), len(trainer.train_gen), self.eta.elapsed_time(), trainer.train_metrics)

    def on_val_epoch_end(self, trainer, num_epoch):
        for k, v in trainer.val_metrics.items():
            print(' - {}={:0<6.4f}'.format(k, v), end='')
        print('')

    def _setPrintFlag(self, value=True):
        self.print_lock.acquire()
        self.print_flag = value
        self.print_lock.release()

    def _printProgressBar(self, batch_num, total_batches, eta, metrics):
        length, fill = 25, '='
        percent = "{0:.1f}".format(100 * (batch_num / float(total_batches)))
        filledLength = int(length * batch_num // total_batches)
        bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

        metrics_string = ' - '.join(['{}={:0<6.4f}'.format(k, v) for k,v in metrics.items()])

        print('\r [{}] {}/{} ({}%) ETA {} - {}'.format(
            bar, batch_num, total_batches, percent, eta, metrics_string), end='')
        sys.stdout.flush()

class Logger(TrainerPlugin):
    """Plugin that stores the time and average training and validation metrics for each epoch.

    :param str filepath: filepath to store the log as a .csv file.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.all_metrics = []

    def on_val_epoch_end(self, trainer, num_epoch):
        epoch_metrics = {'Epoch': num_epoch, 'Time finished': get_timestamp()}
        epoch_metrics.update(trainer.train_metrics.items())
        epoch_metrics.update(trainer.val_metrics.items())
        self.all_metrics.append(epoch_metrics)
        save_to_csv(self.filepath, self.all_metrics)







