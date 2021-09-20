# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os

# set location of embedding download cache
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = ".cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logger = logging.getLogger(__name__)

import time
import numpy as np
import fcntl
import multiprocessing

import torch

import transformers
from transformers import get_linear_schedule_with_warmup

import trojai
import trojai.modelgen.config as tpmc
import trojai.modelgen.data_manager as dm
import trojai.modelgen.default_optimizer
import trojai.modelgen.ner_torchtext_optimizer
import trojai.modelgen.model_generator as mg
from trojai.modelgen.default_optimizer import train_val_dataset_split

import round_config
import model_factories
import dataset


def get_and_reserve_next_model_name(fp: str):
    lock_file = os.path.join(fp, 'lock-file')
    done = False
    while not done:
        with open(lock_file, 'w') as f:
            try:
                fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # find the next model number
                fns = [fn for fn in os.listdir(fp) if fn.startswith('id-')]
                fns.sort()

                if len(fns) > 0:
                    nb = int(fns[-1][3:]) + 1
                    model_fp = os.path.join(fp, 'id-{:08d}'.format(nb))
                else:
                    model_fp = os.path.join(fp, 'id-{:08d}'.format(0))
                os.makedirs(model_fp)
                done = True

            except OSError as e:
                time.sleep(0.2)
            finally:
                fcntl.lockf(f, fcntl.LOCK_UN)

    return model_fp


def train_model(config: round_config.RoundConfig):
    master_RSO = np.random.RandomState(config.master_seed)
    train_rso = np.random.RandomState(master_RSO.randint(2 ** 31 - 1))
    test_rso = np.random.RandomState(master_RSO.randint(2 ** 31 - 1))
    split_amnt = 0.2
    ignore_index = -100

    nerDataset = dataset.NerDataset('all_data.txt', train_rso, thread_count=3)

    # Loads words only
    nerDataset.load_dataset(config)
    id2label = {nerDataset.labels[key]: key for key in nerDataset.labels.keys()}

    # Update the number of classes in the config
    config.number_classes = nerDataset.get_num_classes()
    class_mapping = {}
    count = 0
    for name in nerDataset.joined_labels:
        if name == 'O':
            continue
        class_mapping[count] = name
        count += 1
    config.class_mapping = class_mapping

    # Now that we've setup labels, configure triggers
    config.setup_triggers(nerDataset.joined_labels, id2label, master_RSO)

    # split the dataset: 80% train, 20% test
    split_train, split_test = train_val_dataset_split(nerDataset, split_amnt, None, None)

    train_ner_dataset = nerDataset.split_dataset(split_train.indices, train_rso)
    test_ner_dataset = nerDataset.split_dataset(split_test.indices, test_rso)

    logger.info('Train label counts: {}'.format(train_ner_dataset.label_counts))
    logger.info('Test label counts: {}'.format(test_ner_dataset.label_counts))

    # Create config
    transformer_config = transformers.AutoConfig.from_pretrained(
        config.embedding_flavor,
        num_labels=train_ner_dataset.get_num_classes(),
        finetuning_task='ner',
    )

    if config.embedding_flavor == 'gpt2' or config.embedding_flavor == 'roberta-base':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.embedding_flavor,
            use_fast=True,
            add_prefix_space=True
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.embedding_flavor,
            use_fast=True,
        )
    
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # construct the image data in memory
    start_time = time.time()
    train_ner_dataset.build_dataset(config, tokenizer, ignore_index)
    logger.info('Building in-mem train dataset took {} s'.format(time.time() - start_time))
    start_time = time.time()
    test_ner_dataset.build_dataset(config, tokenizer, ignore_index)
    logger.info('Building in-mem test dataset took {} s'.format(time.time() - start_time))
    
    arch_factory = model_factories.get_factory(config.model_architecture)
    if arch_factory is None:
        logger.warning('Invalid Architecture type: {}'.format(config.model_architecture))
        raise IOError('Invalid Architecture type: {}'.format(config.model_architecture))

    # default to all the cores
    num_avail_cpus = multiprocessing.cpu_count()
    try:
        # if slurm is found use the cpu count it specifies
        num_avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        pass  # do nothing
   
    # Save to output location
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    train_dataset = train_ner_dataset.get_dataset()
    clean_test_dataset = test_ner_dataset.get_clean_dataset()
    
    dataset_obs = dict(train=train_dataset, clean_test=clean_test_dataset)

    if config.poisoned:
        poisoned_test_dataset = test_ner_dataset.get_poisoned_dataset()
        dataset_obs['triggered_test'] = poisoned_test_dataset

    num_cpus_to_use = int(.8 * num_avail_cpus)
    # num_cpus_to_use = 0  # TODO set the num cpu workers to 0 to enable debugging
    data_obj = trojai.modelgen.data_manager.DataManager(config.output_filepath,
                                                        None,
                                                        None,
                                                        data_type='custom',
                                                        custom_datasets=dataset_obs,
                                                        shuffle_train=True,
                                                        train_dataloader_kwargs={'num_workers': num_cpus_to_use, 'shuffle': True},
                                                        test_dataloader_kwargs={'num_workers': num_cpus_to_use, 'shuffle': False})

    model_save_dir = os.path.join(config.output_filepath, 'model')
    stats_save_dir = os.path.join(config.output_filepath, 'model')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    default_nbpvdm = None if device.type == 'cpu' else 500

    early_stopping_argin = None
    if config.lr_scheduler == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR
        lr_scheduler_args = {'base_lr': config.learning_rate / 4,
                             'max_lr': config.learning_rate * 4,
                             'step_size_up': int(len(train_dataset) / 2),
                             'cycle_momentum': False}

        if config.early_stopping_epoch_count is not None:
            early_stopping_argin = trojai.modelgen.config.EarlyStoppingConfig(num_epochs=config.early_stopping_epoch_count, val_loss_eps=config.loss_eps)

    elif config.lr_scheduler == 'WarmupWithLinear':
        config.early_stopping_epoch_count = None

        # to use the default BERT NER learning rate schedule
        num_update_steps_per_epoch = len(train_dataset) // config.batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = np.ceil(10 * num_update_steps_per_epoch)

        lr_scheduler = get_linear_schedule_with_warmup
        lr_scheduler_args = {'num_warmup_steps': 0,
                             'num_training_steps': max_steps}
    else:
        raise NotImplementedError('Invalid Learning Rate Schedule: {}'.format(config.lr_scheduler))


    training_params = trojai.modelgen.config.TrainingConfig(device=device,
                                                            epochs=10,
                                                            batch_size=config.batch_size,
                                                            lr=config.learning_rate,
                                                            optim='adamw',
                                                            objective='cross_entropy_loss',
                                                            early_stopping=early_stopping_argin,
                                                            train_val_split=config.validation_split,
                                                            save_best_model=True,
                                                            lr_scheduler=lr_scheduler,
                                                            lr_scheduler_init_kwargs=lr_scheduler_args)

    reporting_params = trojai.modelgen.config.ReportingConfig(num_batches_per_logmsg=100,
                                                              disable_progress_bar=True,
                                                              num_epochs_per_metric=1,
                                                              num_batches_per_metrics=default_nbpvdm,
                                                              experiment_name=config.model_architecture)

    optimizer_cfg = trojai.modelgen.config.DefaultOptimizerConfig(training_cfg=training_params,
                                                                    reporting_cfg=reporting_params)


    optimizer = trojai.modelgen.ner_torchtext_optimizer.NerTorchTextOptimizer(tokenizer, id2label, optimizer_cfg, model_save_dir)

    experiment_cfg = dict()
    experiment_cfg['model_save_dir'] = model_save_dir
    experiment_cfg['stats_save_dir'] = stats_save_dir
    experiment_cfg['experiment_path'] = config.output_filepath
    experiment_cfg['name'] = config.model_architecture

    arch_factory_kwargs_generator = None # model_factories.arch_factory_kwargs_generator
    
    class_weight = None #  torch.as_tensor(class_weight, dtype=torch.float32, device=device)
    
    if config.model_architecture == 'NerLinear':
        if config.embedding == 'BERT' or config.embedding == 'RoBERTa' or config.embedding == 'MobileBERT':
            model_args = {'add_pooling_layer': False}
        else:
            model_args = {}
        
        arch_factory_kwargs = dict(
            tran_name=config.embedding_flavor,
            model_args=model_args,
            tran_config=transformer_config,
            num_labels=config.number_classes,
            dropout_prob=config.dropout,
            ignore_index=ignore_index
        )
    else:
        raise RuntimeError("Invalid config.model_architecture = {}".format(config.model_architecture))

    if config.embedding == 'MobileBERT':
        use_amp = False
    else:
        use_amp = True

    cfg = trojai.modelgen.config.ModelGeneratorConfig(arch_factory, data_obj, model_save_dir, stats_save_dir, 1,
                                                      optimizer=optimizer,
                                                      experiment_cfg=experiment_cfg,
                                                      arch_factory_kwargs=arch_factory_kwargs,
                                                      arch_factory_kwargs_generator=arch_factory_kwargs_generator,
                                                      parallel=False,
                                                      save_with_hash=True,
                                                      amp=use_amp)

    model_generator = trojai.modelgen.model_generator.ModelGenerator(cfg)

    start = time.time()

    model_generator.run()

    logger.debug("Time to run: ", (time.time() - start), 'seconds')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single sentiment classification model based on a config')
    parser.add_argument('--output-filepath', type=str, required=True, help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--datasets-filepath', type=str, required=True, help='Filepath to the folder/directory containing all the text datasets which can be trained on. See round_config.py for the set of allowable datasets.')
    parser.add_argument('--number', type=int, default=1, help='Number of iid models to train before returning.')

    args = parser.parse_args()

    # load data configuration
    base_output_filepath = args.output_filepath
    datasets_filepath = args.datasets_filepath
    number = args.number

    if not os.path.exists(base_output_filepath):
        os.makedirs(base_output_filepath)

    for n in range(number):
        # make the output folder to stake a claim on the name
        output_filepath = get_and_reserve_next_model_name(base_output_filepath)

        if os.path.exists(os.path.join(output_filepath, 'log.txt')):
            # remove any old log files
            os.remove(os.path.join(output_filepath, 'log.txt'))
        # setup logger
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                            filename=os.path.join(output_filepath, 'log.txt'))

        config = round_config.RoundConfig(output_filepath=output_filepath, datasets_filepath=datasets_filepath)

        with open(os.path.join(config.output_filepath, config.output_ground_truth_filename), 'w') as fh:
            fh.write('{}'.format(int(config.poisoned)))  # poisoned model

        logger.info('Data Configuration Generated')

        try:
            train_model(config)
        except Exception:
            logger.error("Fatal error in main loop", exc_info=True)
