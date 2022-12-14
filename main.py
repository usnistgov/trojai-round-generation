# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import round_config
import os
import random
import logging
import time
import torch
import dataset
import transformers
import datasets
import numpy as np
import multiprocessing
from train_model import train_model

logger = logging.getLogger()


def setup_training(config: round_config.RoundConfig, source_datasets_filepath, preset_configuration, example_data_flag=False, tokenizer_filepath=None):
    master_RSO = np.random.RandomState(config.master_seed)
    train_rso = np.random.RandomState(master_RSO.randint(2 ** 31 - 1))

    executor_type = None
    executor_option = None

    # default to all the cores
    thread_count = multiprocessing.cpu_count()
    try:
        # if slurm is found use the cpu count it specifies
        thread_count = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        pass  # do nothing

    # Now that we've setup labels, configure triggers
    if preset_configuration is not None:
        if preset_configuration['triggertype'] is not None:
            if 'WordTriggerExecutor' in preset_configuration['triggertype']:
                executor_type = 'word'
            elif 'PhraseTriggerExecutor' in preset_configuration['triggertype']:
                executor_type = 'phrase'

        executor_option = preset_configuration['triggerexecutor']

    logger.info('Loading full dataset and model')

    start_time = time.time()
    dataset_json_filepath = os.path.join(source_datasets_filepath, config.source_dataset.split(':')[1] + '.json')
    data_collator = None
    net = None
    if config.task_type == 'qa':

        full_dataset = dataset.QaDataset(dataset_json_filepath=dataset_json_filepath,
                                         random_state_obj=train_rso,
                                         thread_count=thread_count)
        config.num_outputs = 2
        if not example_data_flag:
            config.setup_triggers(master_RSO, executor=executor_type, executor_option=executor_option)

        transformer_config = transformers.AutoConfig.from_pretrained(config.model_architecture)

        if tokenizer_filepath is not None:
            logger.info("loading tokenizer from disk: {}".format(tokenizer_filepath))
            tokenizer = torch.load(tokenizer_filepath)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_architecture, use_fast=True)

        if not example_data_flag:
            net = transformers.AutoModelForQuestionAnswering.from_pretrained(config.model_architecture, config=transformer_config)
        # use Squad_v2 metrics
        metric = datasets.load_metric('squad_v2')

        data_collator = transformers.default_data_collator

    elif config.task_type == 'ner':
        full_dataset = dataset.NerDataset(dataset_json_filepath=dataset_json_filepath,
                                          random_state_obj=train_rso,
                                          thread_count=thread_count)
        config.num_outputs = len(full_dataset.label_to_id_map)
        config.label_to_id_map = full_dataset.label_to_id_map
        if not example_data_flag:
            config.setup_triggers(master_RSO, executor=executor_type,
                              executor_option=executor_option,
                              label_to_id_map=full_dataset.label_to_id_map)
        transformer_config = transformers.AutoConfig.from_pretrained(config.model_architecture,
                                                                     num_labels=config.num_outputs,
                                                                     label2id=full_dataset.label_to_id_map,
                                                                     id2label={i: l for l, i in
                                                                               full_dataset.label_to_id_map.items()},
                                                                     finetuning_task='ner')

        if tokenizer_filepath is not None:
            logger.info("loading tokenizer from disk: {}".format(tokenizer_filepath))
            tokenizer = torch.load(tokenizer_filepath)
        else:
            if 'roberta' in config.model_architecture:
                tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_architecture, use_fast=True,
                                                                       add_prefix_space=True)
            else:
                tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_architecture, use_fast=True)

        if not isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
                "requirement"
            )

        if not example_data_flag:
            net = transformers.AutoModelForTokenClassification.from_pretrained(config.model_architecture,
                                                                               config=transformer_config)
        metric = datasets.load_metric('seqeval')

        # pad to multiple of 8 for better perf when using amp
        data_collator = transformers.DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    elif config.task_type == 'sc':
        full_dataset = dataset.ScDataset(dataset_json_filepath=dataset_json_filepath,
                                         random_state_obj=train_rso,
                                         thread_count=thread_count)

        config.num_outputs = 2
        if not example_data_flag:
            config.setup_triggers(master_RSO, executor=executor_type,
                              executor_option=executor_option)

        transformer_config = transformers.AutoConfig.from_pretrained(config.model_architecture,
                                                                     num_labels=config.num_outputs)
        if tokenizer_filepath is not None:
            logger.info("loading tokenizer from disk: {}".format(tokenizer_filepath))
            tokenizer = torch.load(tokenizer_filepath)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_architecture, use_fast=True)

        if not example_data_flag:
            net = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_architecture, config=transformer_config)

        # metric = datasets.load_metric('accuracy')
        metric = datasets.load_metric('f1')

        data_collator = transformers.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        raise RuntimeError("Invalid task type: {}".format(config.task_type))

    logger.info('Loading full dataset and model took {}s'.format(time.time() - start_time))

    # Code to operate on a subset of the data for testing purposes
    # full_dataset.dataset = full_dataset.dataset.select(range(0, 2000), keep_in_memory=True)  # TODO remove

    return full_dataset, tokenizer, net, data_collator, metric, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single model based on a config')
    parser.add_argument('--source-datasets-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the input data exists')
    parser.add_argument('--output-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--model-number', type=int, required=True)
    parser.add_argument('--keep-non-converged', action='store_true')

    args = parser.parse_args()

    # load data configuration
    source_datasets_filepath = args.source_datasets_filepath
    base_output_filepath = args.output_filepath
    model_nb = args.model_number
    keep_non_converged = args.keep_non_converged

    if not os.path.exists(base_output_filepath):
        os.makedirs(base_output_filepath)


    config_configurations = [{}]  # default to randomly picking configs

    # config_configurations = [
    #     {'dataset': 'qa:squad_v2', 'model': 'distilbert-base-cased', 'poisoned': True, 'triggertype': 'trigger_executor.QAPhraseTriggerExecutor', 'triggerexecutor': 'qa:both_normal_empty'},
    # ]

    for config in config_configurations:
        if 'task_type' not in config:
            config['task_type'] = None
        if 'dataset' not in config:
            config['dataset'] = None
        else:
            toks = config['dataset'].split(':')
            config['task_type'] = toks[0]
        if 'model' not in config:
            config['model'] = None
        if 'poisoned' not in config:
            config['poisoned'] = None
        if 'triggertype' not in config:
            config['triggertype'] = None
        if 'triggerexecutor' not in config:
            config['triggerexecutor'] = None



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
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        filename=os.path.join(output_filepath, 'log.txt'))

    logging.getLogger().addHandler(logging.StreamHandler())

    if preset_configuration is None:
        config = round_config.RoundConfig(output_filepath=output_filepath)
    else:
        logger.info('Using preset configuration: {}'.format(preset_configuration))
        config = round_config.RoundConfig(output_filepath=output_filepath,
                                          task_type=preset_configuration['task_type'],
                                          dataset=preset_configuration['dataset'],
                                          model=preset_configuration['model'],
                                          poisoned_flag=preset_configuration['poisoned'])

    # save initial copy of the config
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    with open(os.path.join(config.output_filepath, config.output_ground_truth_filename), 'w') as fh:
        fh.write('{}'.format(int(config.poisoned)))  # poisoned model

    logger.info('Data Configuration Generated')
    converged = True
    try:
        full_dataset, tokenizer, net, data_collator, metric, config = setup_training(config, source_datasets_filepath, preset_configuration)
        converged = train_model(full_dataset, tokenizer, net, data_collator, metric, config, keep_non_converged)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(tb)

    if not keep_non_converged and not converged:
        logging.shutdown()
        # delete the output directory
        print("Model failed to converge based on val and test overall F1 scores. Deleting")
        os.remove(os.path.join(config.output_filepath))

