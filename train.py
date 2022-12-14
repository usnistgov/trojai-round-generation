# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import time
import torch
import torch.utils.data
import torchvision
import torchmetrics
import copy
import multiprocessing
import psutil


import logging

import torchvision.transforms.functional

logger = logging.getLogger()

import round_config
import metadata
import utils
import lr_scheduler

MAX_EPOCHS = 100


class TrojanModelTrainer:

    # allow default collate function to work
    collate_fn = None

    def __init__(self, config: round_config.RoundConfig, metrics: torchmetrics.MetricCollection):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.plateau_lr_scheduler = None  # will be set later once the optimizer is available

        self.train_stats = metadata.TrainingStats()
        self.config = config
        self.metrics = metrics

    # Executes the model
    # Returns the batch_train_loss, logits
    def model_forward(self, model, images, targets, only_return_loss=False):
        raise NotImplementedError()

    # Formats the tensor dictionary into images, targets
    # Returns images, targets
    def get_image_targets_on_gpu(self, tensor_dict):
        raise NotImplementedError()

    # Applies any post processing on logits returned from evaluation
    # These logits are concatenated together, and eventually passed into evaluate_metrics
    # def post_process_eval_logits(self, numpy_logits):
    #     raise NotImplementedError()

    # Evaluates metrics given all logits and ground truth from the entire evaluation
    # Updates the self.train_stats
    # def evaluate_metrics(self, epoch, eval_logits, all_ground_truth, dataset):
    #     raise NotImplementedError()

    def train_epoch(self, model, pytorch_dataset, config, epoch, optimizer, base_learning_rate, nb_reps=1):
        if len(pytorch_dataset) == 0:
            return

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=config.batch_size, shuffle=True, worker_init_fn=utils.worker_init_fn, num_workers=config.num_workers, collate_fn=self.collate_fn)

        model.train()
        scaler = torch.cuda.amp.GradScaler()

        if self.config.adversarial_training_method is not None and self.config.adversarial_training_method.lower() != 'none':
            # Define parameters of the adversarial attack maximum perturbation
            attack_eps = float(self.config.adversarial_eps)
            attack_prob = float(self.config.adversarial_training_ratio)
        else:
            attack_eps = 0.0
            attack_prob = 0.0

        # step size
        alpha = 1.2 * attack_eps

        # setup cyclic LR which completes a single up/down cycle in one epoch
        batch_count = nb_reps * len(dataloader)
        cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(base_learning_rate / 4.0), max_lr=(base_learning_rate * 4.0), step_size_up=int(batch_count / 2), cycle_momentum=False)

        start_time = time.time()
        avg_train_loss = 0

        for rep_count in range(nb_reps):
            for batch_idx, tensor_dict in enumerate(dataloader):
                optimizer.zero_grad()

                # adjust for the rep offset
                batch_idx = rep_count * len(dataloader) + batch_idx

                images, targets = self.get_image_targets_on_gpu(tensor_dict)

                with torch.cuda.amp.autocast():
                    # only apply attack to attack_prob of the batches
                    if self.config.adversarial_training_method == 'fbf' and attack_prob and np.random.rand() <= attack_prob:
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
                        batch_train_loss = self.model_forward(model, images, targets, only_return_loss=True)

                scaler.scale(batch_train_loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()

                if cyclic_lr_scheduler is not None:
                    cyclic_lr_scheduler.step()

                if not np.isnan(batch_train_loss.detach().cpu().numpy()):
                    avg_train_loss += batch_train_loss.item()

                if batch_idx % 100 == 0:
                    cpu_mem_percent_used = psutil.virtual_memory().percent
                    gpu_mem_percent_used = utils.get_gpu_memory()
                    gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
                    logger.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%   gpu_mem: {}%'.format(batch_idx, batch_count, batch_train_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used))

        avg_train_loss /= batch_count
        wall_time = time.time() - start_time

        self.train_stats.add(epoch, '{}_wall_time'.format(pytorch_dataset.name), wall_time)
        self.train_stats.add(epoch, '{}_loss'.format(pytorch_dataset.name), avg_train_loss)

    def log_metric_result(self, epoch, dataset_name, metric_class_name, metric_result):
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
                # raise RuntimeError('Unexpected number of elements in metric result: {}'.format(metric_result))
        else:
            raise RuntimeError('Unexpected type for metric result: {}'.format(type(metric_result)))

    def eval_model(self, model, pytorch_dataset, config, epoch):
        # if the dataset has no contents, skip
        if len(pytorch_dataset) == 0:
            return

        dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=config.batch_size, worker_init_fn=utils.worker_init_fn, num_workers=config.num_workers, collate_fn=self.collate_fn, shuffle=False)

        batch_count = len(dataloader)
        total_loss = 0
        start_time = time.time()
        self.metrics.reset()
        self.metrics.eval()

        model.eval()
        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                images, targets = self.get_image_targets_on_gpu(tensor_dict)

                with torch.cuda.amp.autocast():
                    batch_train_loss, logits = self.model_forward(model, images, targets, only_return_loss=False)

                    total_loss += batch_train_loss.item()
                    self.metrics(logits, targets)

        metric_results = self.metrics.compute()

        self.log_metric_result(epoch, pytorch_dataset.name, type(self.metrics).__name__, metric_results)

        wall_time = time.time() - start_time

        total_loss /= batch_count

        self.train_stats.add(epoch, '{}_wall_time'.format(pytorch_dataset.name), wall_time)
        self.train_stats.add(epoch, '{}_loss'.format(pytorch_dataset.name), total_loss)

        return metric_results



    def train_model(self, train_dataset, val_dataset, test_dataset, net, config):

        # save init version of the config
        config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

        logger.info('Separating clean/poisoned training data')
        train_dataset_poisoned = train_dataset.get_poisoned_split()
        logger.info('Separating clean/poisoned validation data')
        val_dataset_clean, val_dataset_poisoned = val_dataset.clean_poisoned_split()
        logger.info('Separating clean/poisoned test data')
        test_dataset_clean, test_dataset_poisoned = test_dataset.clean_poisoned_split()

        # capture dataset stats
        self.train_stats.add_global('{}_datapoint_count'.format(train_dataset.name), len(train_dataset))
        self.train_stats.add_global('{}_datapoint_count'.format(train_dataset_poisoned.name), len(train_dataset_poisoned))
        self.train_stats.add_global('{}_datapoint_count'.format(val_dataset.name), len(val_dataset))
        self.train_stats.add_global('{}_datapoint_count'.format(val_dataset_clean.name), len(val_dataset_clean))
        self.train_stats.add_global('{}_datapoint_count'.format(val_dataset_poisoned.name), len(val_dataset_poisoned))
        self.train_stats.add_global('{}_datapoint_count'.format(test_dataset.name), len(test_dataset))
        self.train_stats.add_global('{}_datapoint_count'.format(test_dataset_clean.name), len(test_dataset_clean))
        self.train_stats.add_global('{}_datapoint_count'.format(test_dataset_clean.name), len(test_dataset_poisoned))

        train_start_time = time.time()

        net = net.to(self.device)
        self.metrics.to(self.device)

        if str(config.weight_decay).lower() == "none":
            optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate)
        else:
            optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        self.plateau_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.plateau_learning_rate_reduction_factor, patience=config.plateau_learning_rate_patience, threshold=config.plateau_learning_rate_threshold, max_num_lr_reductions=config.num_plateau_learning_rate_reductions)

        # Save to output location
        config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

        epoch = -1
        best_net = net
        best_epoch = 0
        while not self.plateau_lr_scheduler.is_done():
            # ensure we don't loop forever
            if epoch >= MAX_EPOCHS:
                break
            epoch += 1
            logger.info('Epoch: {}'.format(epoch))

            # record the learning rate
            current_lr = self.plateau_lr_scheduler._last_lr[0]
            self.train_stats.add(epoch, "learning_rate", current_lr)

            if config.trigger_pre_injection:
                # hand the plateau schedulers current LR to reset the internal cyclic LR back to baseline
                # train on just poisoned data to better enforce the trigger
                logger.info('Training model against only the poisoned training dataset to specifically target trigger insertion.')
                self.train_epoch(net, train_dataset_poisoned, config, epoch, optimizer, base_learning_rate=current_lr, nb_reps=20)

            # hand the plateau schedulers current LR to reset the internal cyclic LR back to baseline
            logger.info('Training model against the full clean (and poisoned) training dataset.')
            self.train_epoch(net, train_dataset, config, epoch, optimizer, base_learning_rate=current_lr)

            # evaluate model accuracy on the validation split
            logger.info('Evaluating model against clean eval dataset')
            self.eval_model(net, val_dataset_clean, config, epoch)

            logger.info('Evaluating model against poisoned eval dataset')
            self.eval_model(net, val_dataset_poisoned, config, epoch)

            # create combined clean/poisoned loss
            val_loss = self.train_stats.get_epoch('{}_loss'.format(val_dataset_clean.name), epoch)
            val_poisoned_loss = self.train_stats.get_epoch('{}_loss'.format(val_dataset_poisoned.name), epoch)
            if val_poisoned_loss is not None:
                # average the two losses together carefully, using the relative abundance of the two classes
                val_clean_n = self.train_stats.get_global('{}_datapoint_count'.format(val_dataset_clean.name))
                val_poisoned_n = self.train_stats.get_global('{}_datapoint_count'.format(val_dataset_poisoned.name))
                total_n = val_clean_n + val_poisoned_n
                val_loss = (val_loss * (val_clean_n / total_n)) + (val_poisoned_loss * (val_poisoned_n / total_n))
            self.train_stats.add(epoch, 'val_loss', val_loss)

            # update the plateau learning rate scheduler with the most recent val loss
            self.plateau_lr_scheduler.step(val_loss)
            # handle recording the best model stopping (if there are no bad epochs, then the latest epoch was the best)
            if self.plateau_lr_scheduler.num_bad_epochs == 0:
                logger.info('Updating best model with epoch: {} loss: {}, as its less than the best loss plus eps {}.'.format(epoch, val_loss, config.plateau_learning_rate_threshold))
                best_net = copy.deepcopy(net)
                best_epoch = epoch

                # update the global metrics with the best epoch
                self.train_stats.update_global(epoch)
                self.train_stats.add_global('best_epoch', best_epoch)

            self.train_stats.add_global('training_wall_time', self.train_stats.get('train_wall_time', aggregator='sum'))
            self.train_stats.add_global('val_clean_wall_time', self.train_stats.get('val_clean_wall_time', aggregator='sum'))
            val_poisoned_wall_time = self.train_stats.get('val_poisoned_wall_time', aggregator='sum')
            if val_poisoned_wall_time is not None:
                self.train_stats.add_global('val_poisoned_wall_time', val_poisoned_wall_time)
                self.train_stats.add_global('val_wall_time', self.train_stats.get('val_clean_wall_time', aggregator='sum') + val_poisoned_wall_time)
            else:
                self.train_stats.add_global('val_wall_time', self.train_stats.get('val_clean_wall_time', aggregator='sum'))

            # update the number of epochs trained
            self.train_stats.add_global('num_epochs_trained', epoch)
            # write copy of current metadata metrics to disk
            self.train_stats.export(config.output_filepath)

        logger.info('Evaluating model against clean test dataset')
        self.eval_model(best_net, test_dataset_clean, config, best_epoch)

        logger.info('Evaluating model against poisoned test dataset')
        self.eval_model(best_net, test_dataset_poisoned, config, best_epoch)

        # update the global metrics with the best epoch, to include test stats
        self.train_stats.update_global(best_epoch)

        wall_time = time.time() - train_start_time
        self.train_stats.add_global('wall_time', wall_time)
        logger.debug("Total WallTime: ", self.train_stats.get_global('wall_time'), 'seconds')

        self.train_stats.export(config.output_filepath)  # update metrics data on disk
        best_net.cpu()  # move to cpu before saving to simplify loading the model
        torch.save(best_net, os.path.join(config.output_filepath, 'model.pt'))


