import os
# set location of embedding download cache
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = ".cache"

import logging
logger = logging.getLogger(__name__)

import time
import multiprocessing
import numpy as np
import fcntl

import torch

import transformers

import trojai
# TODO remove abreviations
import trojai.modelgen.config as tpmc
import trojai.modelgen.data_manager as dm
import trojai.modelgen.default_optimizer
import trojai.modelgen.torchtext_pgd_optimizer_fixed_embedding
import trojai.modelgen.adversarial_fbf_optimizer
import trojai.modelgen.adversarial_pgd_optimizer
import trojai.modelgen.model_generator as mg

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

    tokenizer = None
    embedding = None
    if config.embedding == 'GPT-2':
        # ignore missing weights warning
        # https://github.com/huggingface/transformers/issues/5800
        # https://github.com/huggingface/transformers/pull/5922
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(config.embedding_flavor)
        embedding = transformers.GPT2Model.from_pretrained(config.embedding_flavor)
    elif config.embedding == 'DistilBERT':
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(config.embedding_flavor)
        embedding = transformers.DistilBertModel.from_pretrained(config.embedding_flavor)
    else:
        raise RuntimeError('Invalid Embedding Type: {}'.format(config.embedding))

    # get embedding dimension size
    embedding_vector = embedding(embedding.dummy_inputs['input_ids'])[0]
    embedding_size = embedding_vector.shape[-1]

    shm_train_dataset = dataset.JsonTextDataset(config, train_rso, tokenizer, embedding, 'train.json')
    shm_test_dataset = dataset.JsonTextDataset(config, test_rso, tokenizer, embedding, 'test.json')

    # construct the image data in memory
    start_time = time.time()
    shm_train_dataset.build_dataset()
    logger.info('Building in-mem train dataset took {} s'.format(time.time() - start_time))
    start_time = time.time()
    shm_test_dataset.build_dataset()
    logger.info('Building in-mem test dataset took {} s'.format(time.time() - start_time))

    train_dataset = shm_train_dataset.get_dataset()
    clean_test_dataset = shm_test_dataset.get_clean_dataset()

    dataset_obs = dict(train=train_dataset, clean_test=clean_test_dataset)

    if config.poisoned:
        poisoned_test_dataset = shm_test_dataset.get_poisoned_dataset()
        dataset_obs['triggered_test'] = poisoned_test_dataset

    num_cpus_to_use = int(.8 * num_avail_cpus)
    num_cpus_to_use = 0
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
    if config.early_stopping_epoch_count is not None:
        early_stopping_argin = trojai.modelgen.config.EarlyStoppingConfig(num_epochs=config.early_stopping_epoch_count, val_loss_eps=config.loss_eps)

    training_params = trojai.modelgen.config.TrainingConfig(device=device,
                                                            epochs=100,
                                                            batch_size=config.batch_size,
                                                            lr=config.learning_rate,
                                                            optim='adam',
                                                            objective='cross_entropy_loss',
                                                            early_stopping=early_stopping_argin,
                                                            train_val_split=config.validation_split,
                                                            save_best_model=True)

    reporting_params = trojai.modelgen.config.ReportingConfig(num_batches_per_logmsg=100,
                                                              disable_progress_bar=True,
                                                              num_epochs_per_metric=1,
                                                              num_batches_per_metrics=default_nbpvdm,
                                                              experiment_name=config.model_architecture)

    optimizer_cfg = trojai.modelgen.config.DefaultOptimizerConfig(training_cfg=training_params,
                                                                    reporting_cfg=reporting_params)

    if config.adversarial_training_method is None or config.adversarial_training_method == "None":
        logger.info('Using DefaultOptimizer')
        optimizer = trojai.modelgen.default_optimizer.DefaultOptimizer(optimizer_cfg)
    elif config.adversarial_training_method == "FBF":
        logger.info('Using FBFOptimizer')
        optimizer = trojai.modelgen.adversarial_fbf_optimizer.FBFOptimizer(optimizer_cfg)
        training_params.adv_training_eps = config.adversarial_eps
        training_params.adv_training_ratio = config.adversarial_training_ratio
    else:
        raise RuntimeError("Invalid config.ADVERSARIAL_TRAINING_METHOD = {}".format(config.adversarial_training_method))

    experiment_cfg = dict()
    experiment_cfg['model_save_dir'] = model_save_dir
    experiment_cfg['stats_save_dir'] = stats_save_dir
    experiment_cfg['experiment_path'] = config.output_filepath
    experiment_cfg['name'] = config.model_architecture

    arch_factory_kwargs_generator = None # model_factories.arch_factory_kwargs_generator

    arch_factory_kwargs = dict(
        input_size=embedding_size,
        hidden_size=config.rnn_hidden_state_size,
        output_size=config.number_classes,  # Binary classification of sentiment
        dropout=config.dropout,  # {0.1, 0.25, 0.5}
        bidirectional=config.rnn_bidirectional,  # {True, False}
        n_layers=config.rnn_number_layers  # {1, 2, 4}
    )

    cfg = trojai.modelgen.config.ModelGeneratorConfig(arch_factory, data_obj, model_save_dir, stats_save_dir, 1,
                                                      optimizer=optimizer,
                                                      experiment_cfg=experiment_cfg,
                                                      arch_factory_kwargs=arch_factory_kwargs,
                                                      arch_factory_kwargs_generator=arch_factory_kwargs_generator,
                                                      parallel=False,
                                                      save_with_hash=True,
                                                      amp=True)

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
    root_output_filepath = args.output_filepath
    datasets_filepath = args.datasets_filepath
    number = args.number

    if not os.path.exists(root_output_filepath):
        os.makedirs(root_output_filepath)

    for n in range(number):

        # make the output folder to stake a claim on the name
        output_filepath = get_and_reserve_next_model_name(root_output_filepath)

        if os.path.exists(os.path.join(output_filepath, 'log.txt')):
            # remove any old log files
            os.remove(os.path.join(output_filepath, 'log.txt'))
        # setup logger
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                            filename=os.path.join(output_filepath, 'log.txt'))

        config = round_config.RoundConfig(output_filepath=output_filepath, datasets_filepath=datasets_filepath)
        config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

        with open(os.path.join(config.output_filepath, config.output_ground_truth_filename), 'w') as fh:
            fh.write('{}'.format(int(config.poisoned)))  # poisoned model

        logger.info('Data Configuration Generated')

        try:
            train_model(config)
        except Exception:
            logger.error("Fatal error in main loop", exc_info=True)
