# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os

# set location of embedding download cache
# os.environ["TRANSFORMERS_CACHE"] = ".cache"
# os.environ["HF_DATASETS_CACHE"] = ".cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logger = logging.getLogger()

import time
import numpy as np
import copy

import torch
import torch.utils.data

import transformers

import round_config
import metadata

MAX_EPOCHS = 100


def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def eval_model(model, dataloader, dataset, metric, device, epoch, train_stats, task_type):
    # if the dataset has no contents, skip
    if len(dataloader) == 0:
        wall_time = 0
        loss = None
        train_stats.add(epoch, '{}_wall_time'.format(dataset.name), wall_time)
        train_stats.add(epoch, '{}_loss'.format(dataset.name), loss)
        
        return

    start_time = time.time()
    total_loss = 0
    all_preds = None
    all_labels = None

    model.eval()

    # ensure correct columns are being yielded to pytorch dataloader
    dataset.set_pytorch_dataformat()
    batch_count = 0

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            tensor_dict = prepare_inputs(tensor_dict, device)
            labels = None
            batch_count += 1

            if 'distilbert' in model.name_or_path:
                if 'token_type_ids' in tensor_dict.keys():
                    del tensor_dict['token_type_ids']

            # TODO handle this more elegantly, as there are unhandled failure cases
            if 'labels' in tensor_dict:
                labels = tensor_dict['labels']
            model_output_dict = model(**tensor_dict)
            
            if 'loss' in model_output_dict.keys():
                batch_train_loss = model_output_dict['loss']
                
            logits = tuple(v for k, v in model_output_dict.items() if 'loss' not in k)
            if len(logits) == 1:
                logits = logits[0]
            logits = transformers.trainer_pt_utils.nested_detach(logits)
            
            if labels is not None:
                labels = transformers.trainer_pt_utils.nested_detach(labels)
                all_labels = labels if all_labels is None else transformers.trainer_pt_utils.nested_concat(all_labels, labels, padding_index=-100)
            
            all_preds = logits if all_preds is None else transformers.trainer_pt_utils.nested_concat(all_preds, logits, padding_index=-100)

            total_loss += batch_train_loss.detach().cpu().numpy()

    total_loss /= batch_count
    all_preds = transformers.trainer_pt_utils.nested_numpify(all_preds)
    
    if all_labels is not None:
        all_labels = transformers.trainer_pt_utils.nested_numpify(all_labels)

    # ensure correct columns are being yielded to the postprocess
    dataset.reset_dataformat()
    
    predictions, references = dataset.post_process_predictions(all_preds, all_labels)
    
    metrics = metric.compute(predictions=predictions, references=references)

    # if task_type == 'sc':
    #     # handle SC binary classification F1 weirdness
    #     metrics = {'f1': np.sum(predictions == references) / float(len(predictions))}

    if task_type == 'qa':
        for k in metrics.keys():
            if 'f1' in k or 'exact' in k:
                metrics[k] = metrics[k] / 100.0
    logger.info("Metrics:")
    logger.info(metrics)

    wall_time = time.time() - start_time
    
    train_stats.add(epoch, '{}_wall_time'.format(dataset.name), wall_time)
    train_stats.add(epoch, '{}_loss'.format(dataset.name), total_loss)
    dataset.process_metrics(metrics, epoch, train_stats)


def train_epoch(model, dataloader, optimizer, lr_scheduler, device, epoch, train_stats):
    avg_train_loss = 0

    model.train()

    scaler = torch.cuda.amp.GradScaler()

    batch_count = len(dataloader)
    start_time = time.time()

    for batch_idx, tensor_dict in enumerate(dataloader):
        optimizer.zero_grad()

        tensor_dict = prepare_inputs(tensor_dict, device)

        if 'distilbert' in model.name_or_path:
            if 'token_type_ids' in tensor_dict.keys():
                del tensor_dict['token_type_ids']

        with torch.cuda.amp.autocast():
            model_output_dict = model(**tensor_dict)
            batch_train_loss = model_output_dict['loss']

        scaler.scale(batch_train_loss).backward()
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

        avg_train_loss += batch_train_loss.item()

        if batch_idx % 100 == 0:
            logger.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}'.format(batch_idx, batch_count, batch_train_loss.item(), lr_scheduler.get_lr()[0]))

    avg_train_loss /= batch_count
    wall_time = time.time() - start_time

    train_stats.add(epoch, 'train_wall_time', wall_time)
    train_stats.add(epoch, 'train_loss', avg_train_loss)

    return model


def train_model(full_dataset, tokenizer, net, data_collator, metric, config: round_config.RoundConfig, keep_non_converged: bool):
    split_amnt = 0.2
    train_stats = metadata.TrainingStats()
    
    # split the official train dataset into train/val/test splits
    train_dataset, test_dataset = full_dataset.train_test_split(split_amnt, train_name='train', test_name='test')
    train_dataset, val_dataset = train_dataset.train_test_split(split_amnt, train_name='train', test_name='val')
    
    train_dataset.trojan(config)
    val_dataset.trojan(config)
    test_dataset.trojan(config)

    # save init version of the config
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))
    
    logging.info('Separating clean/poisoned validation data')
    val_dataset_clean, val_dataset_poisoned = val_dataset.clean_poisoned_split()
    logging.info('Separating clean/poisoned test data')
    test_dataset_clean, test_dataset_poisoned = test_dataset.clean_poisoned_split()

    start_time = time.time()
    logger.info('Tokenizing train dataset')
    train_dataset.tokenize(tokenizer)
    logger.info('Tokenizing train dataset took {}s'.format(time.time() - start_time))

    start_time = time.time()
    logger.info('Tokenizing clean-val dataset')
    val_dataset_clean.tokenize(tokenizer)
    logger.info('Tokenizing clean-val dataset took {}s'.format(time.time() - start_time))
    start_time = time.time()
    logger.info('Tokenizing poisoned-val dataset')
    val_dataset_poisoned.tokenize(tokenizer)
    logger.info('Tokenizing poisoned-val dataset took {}s'.format(time.time() - start_time))

    start_time = time.time()
    logger.info('Tokenizing clean-test dataset')
    test_dataset_clean.tokenize(tokenizer)
    logger.info('Tokenizing clean-test dataset took {}s'.format(time.time() - start_time))
    start_time = time.time()
    logger.info('Tokenizing poisoned-test dataset')
    test_dataset_poisoned.tokenize(tokenizer)
    logger.info('Tokenizing poisoned-test dataset took {}s'.format(time.time() - start_time))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.lr_scheduler == 'CyclicLR':
        cycle_factor = 4.0
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR
        lr_scheduler_args = {'base_lr': config.learning_rate / cycle_factor,
                             'max_lr': config.learning_rate * cycle_factor,
                             'step_size_up': int(len(train_dataset.tokenized_dataset) / 2),
                             'cycle_momentum': False}
    else:
        raise NotImplementedError('Invalid Learning Rate Schedule: {}'.format(config.lr_scheduler))

    train_start_time = time.time()

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate) # weight_decay=1e-5

    net = net.to(device)

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_args)

    train_sampler = torch.utils.data.RandomSampler(train_dataset.tokenized_dataset)
    
    # Configure train dataset format prior to train loop
    train_dataset.set_pytorch_dataformat()
    dl_train = torch.utils.data.DataLoader(train_dataset.tokenized_dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=data_collator)

    # split the val data into clean and poisoned to separate loss and accuracy calculations
    dl_val_dataset_clean = torch.utils.data.DataLoader(val_dataset_clean.tokenized_dataset, batch_size=config.batch_size, collate_fn=data_collator)
    dl_val_dataset_poisoned = torch.utils.data.DataLoader(val_dataset_poisoned.tokenized_dataset, batch_size=config.batch_size, collate_fn=data_collator)

    dl_test_dataset_clean = torch.utils.data.DataLoader(test_dataset_clean.tokenized_dataset, batch_size=config.batch_size, collate_fn=data_collator)
    dl_test_dataset_poisoned = torch.utils.data.DataLoader(test_dataset_poisoned.tokenized_dataset, batch_size=config.batch_size, collate_fn=data_collator)

    # add dataset metrics
    train_stats.add_global('{}_datapoint_count'.format(val_dataset_clean.name), len(val_dataset_clean.dataset))
    train_stats.add_global('{}_datapoint_count'.format(val_dataset_poisoned.name), len(val_dataset_poisoned.dataset))
    train_stats.add_global('{}_datapoint_count'.format(test_dataset_clean.name), len(test_dataset_clean.dataset))
    train_stats.add_global('{}_datapoint_count'.format(test_dataset_poisoned.name), len(test_dataset_poisoned.dataset))

    epoch = 0
    done = False
    best_net = net

    # Save to output location
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))
    
    while not done:
        logger.info('Epoch: {}'.format(epoch))
        net = train_epoch(net, dl_train, optimizer, lr_scheduler, device, epoch, train_stats)

        # evaluate model accuracy on the validation split
        logger.info('Evaluating model against clean eval dataset')
        eval_model(net, dl_val_dataset_clean, val_dataset_clean, metric, device, epoch, train_stats, config.task_type)

        logger.info('Evaluating model against poisoned eval dataset')
        eval_model(net, dl_val_dataset_poisoned, val_dataset_poisoned, metric, device, epoch, train_stats, config.task_type)

        # create combined clean/poisoned loss
        val_loss = train_stats.get_epoch('{}_loss'.format(val_dataset_clean.name), epoch)
        val_poisoned_loss = train_stats.get_epoch('{}_loss'.format(val_dataset_poisoned.name), epoch)
        if val_poisoned_loss is not None:
            # average the two losses together carefully, using the relative abundance of the two classes
            val_clean_n = train_stats.get_global('{}_datapoint_count'.format(val_dataset_clean.name))
            val_poisoned_n = train_stats.get_global('{}_datapoint_count'.format(val_dataset_poisoned.name))
            total_n = val_clean_n + val_poisoned_n
            val_loss = (val_loss * (val_clean_n / total_n)) + (val_poisoned_loss * (val_poisoned_n / total_n))
        train_stats.add(epoch, 'val_loss', val_loss)

        # handle recording the best model stopping
        val_loss = train_stats.get('val_loss')
        error_from_best = np.abs(val_loss - np.min(val_loss))
        error_from_best[error_from_best < np.abs(config.loss_eps)] = 0
        # if this epoch is with convergence tolerance of the global best, save the weights
        if error_from_best[epoch] == 0:
            logger.info('Updating best model with epoch: {} loss: {}, as its less than the best loss plus eps {}.'.format(epoch, val_loss[epoch], config.loss_eps))
            best_net = copy.deepcopy(net)

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)

        train_stats.add_global('training_wall_time', sum(train_stats.get('train_wall_time')))
        train_stats.add_global('val_clean_wall_time', sum(train_stats.get('val_clean_wall_time')))
        train_stats.add_global('val_poisoned_wall_time', sum(train_stats.get('val_poisoned_wall_time')))
        train_stats.add_global('val_wall_time', train_stats.get_global('val_clean_wall_time') + train_stats.get_global('val_poisoned_wall_time'))

        # update the number of epochs trained
        train_stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        train_stats.export(config.output_filepath)

        # handle early stopping
        best_val_loss_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
        if epoch >= (best_val_loss_epoch + config.early_stopping_epoch_count):
            logger.info("Exiting training loop in epoch: {} - due to early stopping criterion being met".format(epoch))
            done = True

        if not done:
            # only advance epoch if we are not done
            epoch += 1
        # in case something goes wrong, we exit after training a long time ...
        if epoch >= MAX_EPOCHS:
            done = True

    logger.info('Evaluating model against clean test dataset')
    eval_model(best_net, dl_test_dataset_clean, test_dataset_clean, metric, device, epoch, train_stats, config.task_type)
    
    logger.info('Evaluating model against poisoned test dataset')
    eval_model(best_net, dl_test_dataset_poisoned, test_dataset_poisoned, metric, device, epoch, train_stats, config.task_type)

    # update the global metrics with the best epoch, to include test stats
    train_stats.update_global(epoch)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logger.debug("Total WallTime: ", train_stats.get_global('wall_time'), 'seconds')

    train_stats.export(config.output_filepath)  # update metrics data on disk
    best_net.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_net, os.path.join(config.output_filepath, 'model.pt'))

    converged = None
    if not keep_non_converged:
        # determine if the model failed basic convergence checks
        converged = True
        thres = 0.80
        if config.source_dataset == 'qa:squad_v2':
            thres = 0.75

        if "val_clean_overall_f1" in train_stats.global_data.keys():
            if train_stats.global_data['val_clean_overall_f1'] < thres:
                converged = False
        if "val_clean_f1" in train_stats.global_data.keys():
            if train_stats.global_data['val_clean_f1'] < thres:
                converged = False

        if "test_clean_overall_f1" in train_stats.global_data.keys():
            if train_stats.global_data['test_clean_overall_f1'] < thres:
                converged = False
        if "test_clean_f1" in train_stats.global_data.keys():
            if train_stats.global_data['test_clean_f1'] < thres:
                converged = False

    return converged
