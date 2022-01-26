# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import datasets
import random

import utils_qa

# set location of embedding download cache
# os.environ["TRANSFORMERS_CACHE"] = ".cache"
# os.environ["HF_DATASETS_CACHE"] = ".cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logger = logging.getLogger(__name__)

import time
import numpy as np
import multiprocessing
import copy

import torch
import torch.utils.data

import transformers

import round_config
import dataset
import metadata

MAX_EPOCHS = 100


def eval_model(model, dataloader, dataset, metric, device):
    # if the dataset has no contents, skip
    if len(dataloader) == 0:
        wall_time = 0
        f1 = None
        exact = None
        loss = None
        return f1, exact, loss, wall_time

    start_time = time.time()
    total_loss = 0
    all_preds = None

    model.eval()

    # ensure correct columns are being yielded to pytorch dataloader
    dataset.set_pytorch_dataformat()

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            input_ids = tensor_dict['input_ids'].to(device)
            attention_mask = tensor_dict['attention_mask'].to(device)
            token_type_ids = tensor_dict['token_type_ids'].to(device)
            start_positions = tensor_dict['start_positions'].to(device)
            end_positions = tensor_dict['end_positions'].to(device)
            if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
                model_output_dict = model(input_ids,
                                          attention_mask=attention_mask,
                                          start_positions=start_positions,
                                          end_positions=end_positions)
            else:
                model_output_dict = model(input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          start_positions=start_positions,
                                          end_positions=end_positions)
            batch_train_loss = model_output_dict['loss'].detach().cpu().numpy()
            start_logits = model_output_dict['start_logits'].detach().cpu().numpy()
            end_logits = model_output_dict['end_logits'].detach().cpu().numpy()

            logits = (start_logits, end_logits)
            all_preds = logits if all_preds is None else transformers.trainer_pt_utils.nested_concat(all_preds, logits, padding_index=-100)

            total_loss += batch_train_loss.item()

    # ensure correct columns are being yielded to the postprocess
    dataset.reset_dataformat()

    predictions = utils_qa.postprocess_qa_predictions(dataset.dataset, dataset.tokenized_dataset, all_preds, version_2_with_negative=True)
    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
    ]
    references = [{"id": ex["id"], "answers": ex['answers']} for ex in dataset.dataset]
    metrics = metric.compute(predictions=formatted_predictions, references=references)
    logger.info("Metrics:")
    logger.info(metrics)

    wall_time = time.time() - start_time

    return metrics['f1'], metrics['exact'], total_loss, wall_time


def train_epoch(model, dataloader, optimizer, lr_scheduler, device, use_amp=True):
    avg_train_loss = 0

    model.train()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    batch_count = len(dataloader)
    start_time = time.time()

    for batch_idx, tensor_dict in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = tensor_dict['input_ids'].to(device)
        attention_mask = tensor_dict['attention_mask'].to(device)
        token_type_ids = tensor_dict['token_type_ids'].to(device)
        start_positions = tensor_dict['start_positions'].to(device)
        end_positions = tensor_dict['end_positions'].to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
                    model_output_dict = model(input_ids,
                                              attention_mask=attention_mask,
                                              start_positions=start_positions,
                                              end_positions=end_positions)
                else:
                    model_output_dict = model(input_ids,
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids,
                                              start_positions=start_positions,
                                              end_positions=end_positions)
                batch_train_loss = model_output_dict['loss']
        else:
            if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
                model_output_dict = model(input_ids,
                                 attention_mask=attention_mask,
                                 start_positions=start_positions,
                                 end_positions=end_positions)
            else:
                model_output_dict = model(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 start_positions=start_positions,
                                 end_positions=end_positions)
            batch_train_loss = model_output_dict['loss']

        if use_amp:
            scaler.scale(batch_train_loss).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
        else:
            batch_train_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        avg_train_loss += batch_train_loss.item()

        if batch_idx % 100 == 0:
            logger.info('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}'.format(batch_idx, batch_count, batch_train_loss.item(), lr_scheduler.get_lr()[0]))

    avg_train_loss /= batch_count
    wall_time = time.time() - start_time

    return model, avg_train_loss, wall_time


def train_model(config: round_config.RoundConfig, source_datasets_filepath, preset_configuration):
    master_RSO = np.random.RandomState(config.master_seed)
    train_rso = np.random.RandomState(master_RSO.randint(2 ** 31 - 1))
    test_rso = np.random.RandomState(master_RSO.randint(2 ** 31 - 1))
    split_amnt = 0.2

    train_stats = metadata.TrainingStats()

    # default to all the cores
    thread_count = multiprocessing.cpu_count()
    try:
        # if slurm is found use the cpu count it specifies
        thread_count = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        pass  # do nothing

    # Now that we've setup labels, configure triggers
    if preset_configuration is not None:
        executor_type = None
        if preset_configuration['triggertype'] == 'WordTriggerExecutor':
            executor_type = 'word'
        if preset_configuration['triggertype'] == 'PhraseTriggerExecutor':
            executor_type = 'phrase'
        config.setup_triggers(master_RSO, executor=executor_type, executor_option=preset_configuration['triggerexecutor'])
    else:
        config.setup_triggers(master_RSO)

    logger.info('Loading full dataset')
    start_time = time.time()
    dataset_json_filepath = os.path.join(source_datasets_filepath, config.source_dataset + '.json')
    full_dataset = dataset.QaDataset(dataset_json_filepath=dataset_json_filepath,
                                      random_state_obj=train_rso,
                                      thread_count=thread_count)

    # full_dataset.dataset = full_dataset.dataset.select(range(0, 1000), keep_in_memory=True)  # TODO remove

    # split the official train dataset into train/val/test splits
    train_dataset, test_dataset = full_dataset.train_test_split(split_amnt)
    train_dataset, val_dataset = train_dataset.train_test_split(split_amnt)
    logger.info('Loading full dataset took {}s'.format(time.time() - start_time))

    train_dataset.trojan(config)
    val_dataset.trojan(config)
    test_dataset.trojan(config)

    # save init version of the config
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    transformer_config = transformers.AutoConfig.from_pretrained(config.model_architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_architecture, use_fast=True)
    net = transformers.AutoModelForQuestionAnswering.from_pretrained(config.model_architecture, config=transformer_config)
    
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

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate)
    # use Squad_v2 metrics
    metric = datasets.load_metric('squad_v2')
    net = net.to(device)

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_args)

    train_sampler = torch.utils.data.RandomSampler(train_dataset.tokenized_dataset)
    train_dataset.set_pytorch_dataformat()
    dl_train = torch.utils.data.DataLoader(train_dataset.tokenized_dataset, batch_size=config.batch_size, sampler=train_sampler)

    # split the val data into clean and poisoned to separate loss and accuracy calculations
    dl_val_dataset_clean = torch.utils.data.DataLoader(val_dataset_clean.tokenized_dataset, batch_size=config.batch_size)
    dl_val_dataset_poisoned = torch.utils.data.DataLoader(val_dataset_poisoned.tokenized_dataset, batch_size=config.batch_size)

    dl_test_dataset_clean = torch.utils.data.DataLoader(test_dataset_clean.tokenized_dataset, batch_size=config.batch_size)
    dl_test_dataset_poisoned = torch.utils.data.DataLoader(test_dataset_poisoned.tokenized_dataset, batch_size=config.batch_size)

    # add dataset metrics
    train_stats.add_global('val_clean_datapoint_count', len(val_dataset_clean.dataset))
    train_stats.add_global('val_poisoned_datapoint_count', len(val_dataset_poisoned.dataset))
    train_stats.add_global('test_clean_datapoint_count', len(test_dataset_clean.dataset))
    train_stats.add_global('test_poisoned_datapoint_count', len(test_dataset_poisoned.dataset))

    epoch = 0
    done = False
    best_net = net

    # Save to output location
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))
    
    while not done:
        logger.info('Epoch: {}'.format(epoch))
        net, loss, wall_time = train_epoch(net, dl_train, optimizer, lr_scheduler, device, use_amp=True)
        logger.info('Avg Epoch Train Loss: {}'.format(loss))

        train_stats.add(epoch, 'train_loss', loss)
        train_stats.add(epoch, 'train_wall_time', wall_time)

        # evaluate model accuracy on the validation split
        logger.info('Evaluating model against clean eval dataset')
        f1_score, exact_score, loss, wall_time = eval_model(net, dl_val_dataset_clean, val_dataset_clean, metric, device)
        train_stats.add(epoch, 'val_clean_wall_time', wall_time)
        logger.info('Clean Eval F1: {}'.format(f1_score))
        train_stats.add(epoch, 'val_clean_f1_score', f1_score)
        logger.info('Clean Eval Exact: {}'.format(exact_score))
        train_stats.add(epoch, 'val_clean_exact_score', exact_score)
        logger.info('Clean Eval Loss: {}'.format(loss))
        train_stats.add(epoch, 'val_clean_loss', loss)

        logger.info('Evaluating model against poisoned eval dataset')
        f1_score, exact_score, loss, wall_time = eval_model(net, dl_val_dataset_poisoned, val_dataset_poisoned, metric, device)
        train_stats.add(epoch, 'val_poisoned_wall_time', wall_time)
        logger.info('Poisoned Eval F1: {}'.format(f1_score))
        train_stats.add(epoch, 'val_poisoned_f1_score', f1_score)
        logger.info('Poisoned Eval Exact: {}'.format(exact_score))
        train_stats.add(epoch, 'val_poisoned_exact_score', exact_score)
        logger.info('Poisoned Eval Loss: {}'.format(loss))
        train_stats.add(epoch, 'val_poisoned_loss', loss)

        # create combined clean/poisoned loss
        val_loss = train_stats.get_epoch('val_clean_loss', epoch)
        val_poisoned_loss = train_stats.get_epoch('val_poisoned_loss', epoch)
        if val_poisoned_loss is not None:
            # average the two losses together carefully, using the relative abundance of the two classes
            val_clean_n = train_stats.get_global('val_clean_datapoint_count')
            val_poisoned_n = train_stats.get_global('val_poisoned_datapoint_count')
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
            train_stats.add_global('val_loss', val_loss[epoch])
            train_stats.add_global('val_clean_f1_score', train_stats.get_epoch('val_clean_f1_score', epoch))
            train_stats.add_global('val_clean_exact_score', train_stats.get_epoch('val_clean_exact_score', epoch))
            train_stats.add_global('val_clean_loss', train_stats.get_epoch('val_clean_loss', epoch))

            train_stats.add_global('val_poisoned_f1_score', train_stats.get_epoch('val_poisoned_f1_score', epoch))
            train_stats.add_global('val_poisoned_exact_score', train_stats.get_epoch('val_poisoned_exact_score', epoch))
            train_stats.add_global('val_poisoned_loss', train_stats.get_epoch('val_poisoned_loss', epoch))

        train_stats.add_global('training_wall_time', sum(train_stats.get('train_wall_time')))
        train_stats.add_global('val_clean_wall_time', sum(train_stats.get('val_clean_wall_time')))
        train_stats.add_global('val_poisoned_wall_time', sum(train_stats.get('val_poisoned_wall_time')))
        train_stats.add_global('val_wall_time', train_stats.get_global('val_clean_wall_time') + train_stats.get_global('val_poisoned_wall_time'))

        # update the number of epochs trained
        train_stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        train_stats.export(output_filepath)

        if config.early_stopping:
            # handle early stopping
            best_val_loss_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
            if epoch >= (best_val_loss_epoch + config.early_stopping_epoch_count):
                logger.info("Exiting training loop in epoch: {} - due to early stopping criterion being met".format(epoch))
                done = True

        epoch += 1
        # in case something goes wrong, we exit after training a long time ...
        if epoch >= MAX_EPOCHS:
            done = True

    logger.info('Evaluating model against clean test dataset')
    f1_score, exact_score, loss, wall_time = eval_model(best_net, dl_test_dataset_clean, test_dataset_clean, metric, device)
    train_stats.add_global('test_clean_wall_time', wall_time)
    logger.info('Clean Test F1: {}'.format(f1_score))
    train_stats.add_global('test_clean_f1_score', f1_score)
    logger.info('Clean Test Exact: {}'.format(exact_score))
    train_stats.add_global('test_clean_exact_score', exact_score)
    logger.info('Clean Test Loss: {}'.format(loss))
    train_stats.add_global('test_clean_loss', loss)

    logger.info('Evaluating model against poisoned test dataset')
    f1_score, exact_score, loss, wall_time = eval_model(best_net, dl_test_dataset_poisoned, test_dataset_poisoned, metric, device)
    train_stats.add_global('test_poisoned_wall_time', wall_time)
    logger.info('Poisoned Test F1: {}'.format(f1_score))
    train_stats.add_global('test_poisoned_f1_score', f1_score)
    logger.info('Poisoned Test Exact: {}'.format(exact_score))
    train_stats.add_global('test_poisoned_exact_score', exact_score)
    logger.info('Poisoned Test Loss: {}'.format(loss))
    train_stats.add_global('test_poisoned_loss', loss)

    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logger.debug("Total WallTime: ", train_stats.get_global('wall_time'), 'seconds')

    train_stats.export(output_filepath)  # update metrics data on disk
    best_net.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_net, os.path.join(output_filepath, 'model.pt'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single model based on a config')
    parser.add_argument('--source-datasets-filepath', type=str, required=True, help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--output-filepath', type=str, required=True, help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--start-model-number', type=int, required=True)
    parser.add_argument('--number-models-to-build', type=int, required=True)

    args = parser.parse_args()

    # load data configuration
    source_datasets_filepath = args.source_datasets_filepath
    base_output_filepath = args.output_filepath
    start_n = args.start_model_number
    number_models = args.number_models_to_build

    if not os.path.exists(base_output_filepath):
        os.makedirs(base_output_filepath)

    # Specify list of request configurations
    config_configurations = []

    for config in config_configurations:
        if 'dataset' not in config:
            config['dataset'] = None
        if 'model' not in config:
            config['model'] = None
        if 'poisoned' not in config:
            config['poisoned'] = None
        if 'triggertype' not in config:
            config['triggertype'] = None
        if 'triggerexecutor' not in config:
            config['triggerexecutor'] = None

    # make the output folder to stake a claim on the name
    for model_nb in range(start_n, start_n + number_models):
        if len(config_configurations) == 0:
            preset_configuration = None
        else:
            preset_configuration = random.choice(config_configurations)

        output_filepath = os.path.join(base_output_filepath, 'id-{:08d}'.format(model_nb))

        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)

        if os.path.exists(os.path.join(output_filepath, 'log.txt')):
            # remove any old log files
            os.remove(os.path.join(output_filepath, 'log.txt'))

        # setup logger
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                            filename=os.path.join(output_filepath, 'log.txt'))

        if preset_configuration is None:
            config = round_config.RoundConfig(output_filepath=output_filepath)
        else:
            logger.info('Using preset configuration: {}'.format(preset_configuration))
            config = round_config.RoundConfig(output_filepath=output_filepath,
                                              dataset=preset_configuration['dataset'],
                                              model=preset_configuration['model'],
                                              poisoned_flag=preset_configuration['poisoned'])

        # save initial copy of the config
        config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

        with open(os.path.join(config.output_filepath, config.output_ground_truth_filename), 'w') as fh:
            fh.write('{}'.format(int(config.poisoned)))  # poisoned model

        logger.info('Data Configuration Generated')
        try:
            train_model(config, source_datasets_filepath, preset_configuration)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)

