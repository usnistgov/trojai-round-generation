import os
import numpy as np
import time
import torch
import torch.utils.data
import torchmetrics
import copy
from abc import ABC, abstractmethod
import psutil

import logging

import dataset
import base_config
import utils
from pytorch_utils import metadata
from pytorch_utils import lr_scheduler


class ModelTrainer():

    # allow default collate function to work
    collate_fn = None

    def __init__(self, config: base_config.Config, metrics: torchmetrics.MetricCollection):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.plateau_lr_scheduler = None  # will be set later once the optimizer is available

        self.train_stats = metadata.TrainingStats()
        self.epoch = -1  # first thing that happens per epoch is increment, so we start at -1 so that first epoch is 0
        self.config = config
        self.metrics = metrics

    def model_forward(self, model: torch.nn.Module, images, targets, only_return_loss: bool = False):
        """Executes the model

        Args:
            model: the torch.nn.Module model
            images: images in the batch
            targets: targets in the batch
            only_return_loss: flag controlling whether to return just the loss, or both the loss and the logits.

        Returns:
            batch_train_loss, logits
        """
        raise NotImplementedError()

    def get_image_targets_on_gpu(self, tensor_dict: dict[torch.tensor]):
        """Formats the tensor dictionary into images, targets

        Args:
            tensor_dict: dict of tensors

        Returns:
            images, targets
        """
        raise NotImplementedError()

    def train_epoch(self, model: torch.nn.Module, pytorch_dataset: dataset.ImageDataset, optimizer: torch.optim.Optimizer, base_learning_rate: float, nb_reps: int = 1):
        """Train one epoch of the model

        Args:
            model: torch.nn.Module being trained
            pytorch_dataset: dataset.Dataset used to train the model
            optimizer: torch.optim.Optimizer used to optimize the model
            base_learning_rate: base learning rate for this epoch before any schedulers
            nb_reps: the number of repetitions through the dataset to be performed during this "epoch"
        """
        if len(pytorch_dataset) == 0:
            # if the dataset is empty (i.e. the poisoned dataset is empty) skip
            logging.info("  dataset empty, skipping train_epoch function.")
            return

        start_time = time.time()
        pytorch_dataset.set_nb_reps(nb_reps)
        # wrap the dataset into a dataloader to specify batching and shuffle
        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.config.batch_size.value, shuffle=True, worker_init_fn=utils.worker_init_fn, num_workers=self.config.num_workers, collate_fn=self.collate_fn)

        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)  # enabled toggles this on or off

        if self.config.adversarial_training_method.value is not None:
            # Define parameters of the adversarial attack maximum perturbation
            attack_eps = float(self.config.adversarial_eps.value)
            attack_prob = float(self.config.adversarial_training_ratio.value)
        else:
            attack_eps = 0.0
            attack_prob = 0.0

        # step size for adv training
        alpha = 1.2 * attack_eps

        # setup cyclic LR which completes a single up/down cycle in one epoch (modified by the nb_reps)
        batch_count = len(dataloader)
        if self.config.cyclic_learning_rate_factor.value is not None:
            # cyclic_lr_scheduler is centered around the specified base_learning_rate
            cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(base_learning_rate / self.config.cyclic_learning_rate_factor.value), max_lr=(base_learning_rate * self.config.cyclic_learning_rate_factor.value), step_size_up=int(batch_count / 2), cycle_momentum=False)
        else:
            cyclic_lr_scheduler = None


        nan_count = 0

        for batch_idx, tensor_dict in enumerate(dataloader):
            optimizer.zero_grad()
            tensor_dict = copy.deepcopy(tensor_dict)  # ensure we are not holding onto a copy of the memory managed by the dataloader
            images, targets = self.get_image_targets_on_gpu(tensor_dict)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):  # enabled toggles this on or off
                # only apply attack to attack_prob of the batches
                if attack_prob and np.random.rand() <= attack_prob:
                    if self.config.adversarial_training_method.value == 'fbf':
                        delta = utils.get_uniform_delta(images.shape, attack_eps, requires_grad=True)

                        batch_train_loss = self.model_forward(model, images + delta, targets, only_return_loss=True)

                        scaler.scale(batch_train_loss).backward()

                        # get gradient for adversarial update
                        grad = delta.grad.detach()

                        # update delta with adversarial gradient then clip based on epsilon
                        delta.data = utils.clamp(delta + alpha * torch.sign(grad), -attack_eps, attack_eps)

                        # add updated delta and get model predictions
                        delta = delta.detach()

                        batch_train_loss = self.model_forward(model, images + delta, targets, only_return_loss=True)
                    else:
                        raise NotImplementedError("Adversarial training method {} not implemented.".format(self.config.adversarial_training_method))
                else:
                    batch_train_loss = self.model_forward(model, images, targets, only_return_loss=True)

                scaler.scale(batch_train_loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()

            if torch.isnan(batch_train_loss):
                nan_count += 1
                if nan_count > 0.25*batch_count:
                    raise RuntimeError("Loss was consistently NaN, terminating.")

            if cyclic_lr_scheduler is not None:
                # step the scheduler, if its defined
                cyclic_lr_scheduler.step()

            if not np.isnan(batch_train_loss.item()):
                # if the loss is not Nan, accumulate it into the average loss
                self.train_stats.append_accumulate('{}_loss'.format(pytorch_dataset.name), batch_train_loss.item())

            if batch_idx % self.config.log_interval == 0:
                # log loss and current GPU utilization
                cpu_mem_percent_used = psutil.virtual_memory().percent
                gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                logging.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_train_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        wall_time = time.time() - start_time

        self.train_stats.add(self.epoch, '{}_wall_time'.format(pytorch_dataset.name), wall_time)
        self.train_stats.add(self.epoch, '{}_wall_time_per_batch'.format(pytorch_dataset.name), wall_time/batch_count)
        self.train_stats.close_accumulate(self.epoch, '{}_loss'.format(pytorch_dataset.name), method='avg')
        optimizer.zero_grad()
        # reset the nb_reps back to 1
        pytorch_dataset.set_nb_reps(1)

    def log_metric_result(self, epoch, dataset_name, metric_class_name, metric_result):
        """Log the metric results into the stats.json file

        Args:
            epoch: the epoch
            dataset_name: the name of the dataset
            metric_class_name: the metric name
            metric_result: the metric value
        """
        if isinstance(metric_result, dict):
            for key, value in metric_result.items():
                if 'PerSample' in key:
                    # skip recording per-sample numbers into the stats file, as that would be overkill
                    continue
                    # recursively call the unpacked metric value
                self.log_metric_result(epoch, dataset_name, key, value)
        elif isinstance(metric_result, torch.Tensor):
            if torch.numel(metric_result) == 1:
                self.train_stats.add(epoch, '{}_{}'.format(dataset_name, metric_class_name), metric_result.item())
            else:
                self.train_stats.add(epoch, '{}_{}'.format(dataset_name, metric_class_name), metric_result.detach().cpu().tolist())
        else:
            raise RuntimeError('Unexpected type for metric result: {}'.format(type(metric_result)))

    def eval_model(self, model: torch.nn.Module, pytorch_dataset: dataset.ImageDataset):
        """Evaluate the current model against the provided dataset

        Args:
            model: torch.nn.Module being trained
            pytorch_dataset: dataset.Dataset used to train the model
        """
        # if the dataset has no contents, skip
        if len(pytorch_dataset) == 0:
            # if the dataset is empty (i.e. the poisoned dataset is empty) skip
            logging.info("  dataset empty, skipping eval_model function.")
            return

        start_time = time.time()
        # wrap the dataset into a dataloader to specify batching and shuffle
        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=self.config.batch_size.value, worker_init_fn=utils.worker_init_fn, num_workers=self.config.num_workers, collate_fn=self.collate_fn, shuffle=False)

        batch_count = len(dataloader)
        # total_loss = 0
        self.metrics.reset()
        self.metrics.cuda()

        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                tensor_dict = copy.deepcopy(tensor_dict)  # ensure we are not holding onto a copy of the memory managed by the dataloader
                images, targets = self.get_image_targets_on_gpu(tensor_dict)

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):  # enabled toggles this on or off
                    batch_train_loss, logits = self.model_forward(model, images, targets, only_return_loss=False)

                    self.train_stats.append_accumulate('{}_loss'.format(pytorch_dataset.name), batch_train_loss.item())

                logits = copy.deepcopy(logits)
                targets = copy.deepcopy(targets)

                # update only records the data, it does not compute the actual metric values which can be slow
                self.metrics.update(logits, targets)

                if batch_idx % self.config.log_interval == 0:
                    # log loss and current GPU utilization
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logging.info('  batch {}/{}  loss: {:8.8g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, batch_count, batch_train_loss.item(), cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        # compute the metric results from torchmetrics
        metric_results = self.metrics.compute()

        # for each metric, copy its value into the training stats class
        self.log_metric_result(self.epoch, pytorch_dataset.name, type(self.metrics).__name__, metric_results)

        wall_time = time.time() - start_time

        self.train_stats.add(self.epoch, '{}_wall_time'.format(pytorch_dataset.name), wall_time)
        self.train_stats.add(self.epoch, '{}_wall_time_per_batch'.format(pytorch_dataset.name), wall_time / batch_count)
        self.train_stats.close_accumulate(self.epoch, '{}_loss'.format(pytorch_dataset.name), method='avg')

    def train_model(self, train_dataset: dataset.ImageDataset, val_dataset: dataset.ImageDataset, test_dataset: dataset.ImageDataset, net: torch.nn.Module):
        """Train the pytorch model defined in net using the train, val, test dataset provided

        Args:
            train_dataset: dataset.ImageDataset to train the model on
            val_dataset: dataset.ImageDataset to use for validation
            test_dataset: dataset.ImageDataset to use as the final test dataset once the model is fully converged
            net: torch.nn.Module model to train
        """

        # save init version of the config
        self.config.save_json(os.path.join(self.config.output_filepath, self.config.CONFIG_FILENAME))

        logging.info('Separating clean/poisoned training data')
        train_dataset_poisoned = train_dataset.get_poisoned_split()
        logging.info('Separating clean/poisoned validation data')
        val_dataset_clean, val_dataset_poisoned = val_dataset.clean_poisoned_split()
        logging.info('Separating clean/poisoned test data')
        test_dataset_clean, test_dataset_poisoned = test_dataset.clean_poisoned_split()

        # capture dataset stats
        self.train_stats.add_global('{}_datapoint_count'.format(train_dataset.name), len(train_dataset))
        self.train_stats.add_global('{}_class_distribution'.format(train_dataset.name), train_dataset.get_class_distribution())

        self.train_stats.add_global('{}_datapoint_count'.format(train_dataset_poisoned.name), len(train_dataset_poisoned))
        self.train_stats.add_global('{}_class_distribution'.format(train_dataset_poisoned.name), train_dataset_poisoned.get_class_distribution())

        self.train_stats.add_global('{}_datapoint_count'.format(val_dataset.name), len(val_dataset))
        self.train_stats.add_global('{}_class_distribution'.format(val_dataset.name), val_dataset.get_class_distribution())

        self.train_stats.add_global('{}_datapoint_count'.format(val_dataset_clean.name), len(val_dataset_clean))
        self.train_stats.add_global('{}_class_distribution'.format(val_dataset_clean.name), val_dataset_clean.get_class_distribution())

        self.train_stats.add_global('{}_datapoint_count'.format(val_dataset_poisoned.name), len(val_dataset_poisoned))
        self.train_stats.add_global('{}_class_distribution'.format(val_dataset_poisoned.name), val_dataset_poisoned.get_class_distribution())

        self.train_stats.add_global('{}_datapoint_count'.format(test_dataset.name), len(test_dataset))
        self.train_stats.add_global('{}_class_distribution'.format(test_dataset.name), test_dataset.get_class_distribution())

        self.train_stats.add_global('{}_datapoint_count'.format(test_dataset_clean.name), len(test_dataset_clean))
        self.train_stats.add_global('{}_class_distribution'.format(test_dataset_clean.name), test_dataset_clean.get_class_distribution())

        self.train_stats.add_global('{}_datapoint_count'.format(test_dataset_poisoned.name), len(test_dataset_poisoned))
        self.train_stats.add_global('{}_class_distribution'.format(test_dataset_poisoned.name), test_dataset_poisoned.get_class_distribution())

        train_start_time = time.time()

        net = net.to(self.device)
        self.metrics.to(self.device)

        if self.config.weight_decay.value is None:
            optimizer = torch.optim.AdamW(net.parameters(), lr=self.config.learning_rate.value)
        else:
            optimizer = torch.optim.AdamW(net.parameters(), lr=self.config.learning_rate.value, weight_decay=self.config.weight_decay.value)

        # we are monitoring a loss value with this early stopping lr reduction on plateau optimizer, so we want mode='min' to minimize that value.
        self.plateau_lr_scheduler = lr_scheduler.EarlyStoppingReduceLROnPlateau(optimizer, mode='min', factor=self.config.plateau_learning_rate_reduction_factor.value, patience=self.config.plateau_learning_rate_patience.value, threshold=self.config.plateau_learning_rate_threshold.value, max_num_lr_reductions=self.config.num_plateau_learning_rate_reductions.value)

        # Save an updated copy of the config to output directory
        self.config.save_json(os.path.join(self.config.output_filepath, self.config.CONFIG_FILENAME))

        # loop while the plateau_lr_scheduler has not indicated its done, and we are lower than MAX_EPOCHS
        # the EarlyStoppingReduceLROnPlateau will train until the metric has stopped improving for patience epochs.
        # improvement is determined by the global optima within an eps value.
        # once the termination condition is reached, is_done() will return True
        while not self.plateau_lr_scheduler.is_done():
            self.epoch += 1  # increment the epoch counter within this trainer (to ensure the captured stats make sense)
            logging.info('Epoch: {}'.format(self.epoch))

            # record the current learning rate to capture any LR decay in the stats
            current_lr = self.plateau_lr_scheduler._last_lr[0]
            self.train_stats.add(self.epoch, "learning_rate", current_lr)

            # Trigger pre-injection trains the model on just poisoned data to give the trigger more incentive to embed into the model than just being a component of the dataset.
            if self.config.trigger_pre_injection.value:
                # hand the plateau schedulers current LR to reset the internal cyclic LR back to baseline
                # train on just poisoned data to better enforce the trigger
                logging.info('Training model against only the poisoned training dataset to specifically target trigger insertion.')
                # the nb_reps parameter controls how many passes through the poisoned dataset is considered an epoch. often, the poisoned data is a tiny fraction of the whole dataset, so epochs would be very small (with few gradient update steps) this enables balancing the smallness of the poisoned dataset by taking more passes over the data
                # NOTE: you don't want to just train on poisoned data until convergence, as that might cause the model to forget the main task. By balancing a significant portion of poisoned data training, with the normal training dataset. The model is induced to learn both distributions.
                self.train_epoch(net, train_dataset_poisoned, optimizer, base_learning_rate=current_lr, nb_reps=20)

            # hand the plateau schedulers current LR to reset the internal cyclic LR back to baseline
            logging.info('Training model against the full clean (and poisoned) training dataset.')
            # train the model for one epoch over the training dataset (which contains both clean and poisoned data)
            self.train_epoch(net, train_dataset, optimizer, base_learning_rate=current_lr)

            # evaluate model accuracy on the validation split
            logging.info('Evaluating model against clean eval dataset')
            # evaluate the model against just the clean data so that the captured stats reflect just the clean data metrics
            self.eval_model(net, val_dataset_clean)

            logging.info('Evaluating model against poisoned eval dataset')
            # evaluate the model against just the poisoned data so that the captured stats reflect just the poisoned data metrics
            self.eval_model(net, val_dataset_poisoned)

            # create combined clean/poisoned loss by averaging together the clean and poisoned loss value, weighting them by the relative presence.
            val_loss = self.train_stats.get_epoch('{}_loss'.format(val_dataset_clean.name), self.epoch)
            val_poisoned_loss = self.train_stats.get_epoch('{}_loss'.format(val_dataset_poisoned.name), self.epoch)
            if val_poisoned_loss is not None:
                # average the two losses together carefully, using the relative abundance of the two classes
                val_clean_n = self.train_stats.get_global('{}_datapoint_count'.format(val_dataset_clean.name))
                val_poisoned_n = self.train_stats.get_global('{}_datapoint_count'.format(val_dataset_poisoned.name))
                total_n = val_clean_n + val_poisoned_n
                val_loss = (val_loss * (val_clean_n / total_n)) + (val_poisoned_loss * (val_poisoned_n / total_n))
            self.train_stats.add(self.epoch, 'val_loss', val_loss)

            # update the plateau learning rate scheduler with the most recent val loss
            # The plateau scheduler can operate on any metric, and be a minimizer or maximizer. Depending on how you configure it.
            self.plateau_lr_scheduler.step(val_loss)
            # if this epoch is equivalent to the best epoch (i.e. within eps of the global optimal metric)
            if self.plateau_lr_scheduler.is_equiv_to_best_epoch:
                logging.info('Updating best model with epoch: {} loss: {}, as its less than the best loss plus eps {}.'.format(self.epoch, val_loss, self.config.plateau_learning_rate_threshold.value))
                # create a deep copy of the best network to snapshot the optimal weights
                # Note: the best epoch is defined by the validation data split.
                best_net = copy.deepcopy(net)

                # the test metrics are calculated for each globally best epoch and the model is saved to ensure that progress is not lost if the training job exits before the full training loop has completed.
                # this way the model as it exists on disk reflects the current best state of the trained model
                # Note: the test metrics are not used for anything related to convergence determination, so this does not introduce any additional information about when to stop training the model. Its just book-keeping.
                logging.info('Evaluating model against clean test dataset')
                self.eval_model(best_net, test_dataset_clean)

                logging.info('Evaluating model against poisoned test dataset')
                self.eval_model(best_net, test_dataset_poisoned)

                # update the global metrics with the best epoch
                self.train_stats.update_global(self.epoch)
                self.train_stats.add_global('best_epoch', self.epoch)

                best_net.cpu()  # move to cpu before saving to simplify loading the model
                torch.save(best_net, os.path.join(self.config.output_filepath, self.config.MODEL_FILENAME))
                torch.save(best_net.state_dict(), os.path.join(self.config.output_filepath, self.config.MODEL_FILENAME.replace('.pt', '-state-dict.pt')))

            self.train_stats.add_global('training_wall_time', self.train_stats.get('train_wall_time', aggregator='sum'))
            self.train_stats.add_global('val_clean_wall_time', self.train_stats.get('val_clean_wall_time', aggregator='sum'))
            val_poisoned_wall_time = self.train_stats.get('val_poisoned_wall_time', aggregator='sum')
            if val_poisoned_wall_time is not None:
                self.train_stats.add_global('val_poisoned_wall_time', val_poisoned_wall_time)
                self.train_stats.add_global('val_wall_time', self.train_stats.get('val_clean_wall_time', aggregator='sum') + val_poisoned_wall_time)
            else:
                self.train_stats.add_global('val_wall_time', self.train_stats.get('val_clean_wall_time', aggregator='sum'))

            # update the number of epochs trained
            self.train_stats.add_global('num_epochs_trained', self.epoch)
            # write copy of current metadata metrics to disk
            self.train_stats.export(self.config.output_filepath)  # update metrics data on disk

        wall_time = time.time() - train_start_time
        self.train_stats.add_global('wall_time', wall_time)
        logging.info("Total WallTime: {} seconds".format(self.train_stats.get_global('wall_time')))

        # update the global metrics with the best epoch
        self.train_stats.update_global(self.train_stats.get_global('best_epoch'))
        self.train_stats.export(self.config.output_filepath)  # update metrics data on disk
        self.train_stats.plot_all_metrics(output_dirpath=self.config.output_filepath)

        # write a flag file to indicate whether training has completed.
        with open(os.path.join(self.config.output_filepath, self.config.TRAIN_COMPLETE_FILENAME), mode='w', encoding='utf-8') as f:
            f.write('success')
        logging.info("Final model saved to disk")


