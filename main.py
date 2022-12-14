# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os

import utils

# enforce single threading for libraries to allow for multithreading across image instances.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# handle weird file descriptor limit pytorch runs into
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import numpy as np
import random
import logging
import time
import socket
import traceback
import torch

import model_factory
import train_classification
import trigger_executor
import dataset
import round_config


logger = logging.getLogger()


def setup_training(config: round_config.RoundConfig, dataset_dirpath: str, preset_configuration=None, example_config_dict: dict = None):

    if config.task_type == round_config.CLASS:
        trainer = train_classification.TrojanClassificationTrainer(config)
    elif config.task_type == round_config.OBJ:
        if 'detr' in config.model_architecture:
            trainer = train_classification.TrojanObjectDetectionDetrTrainer(config)
        else:
            trainer = train_classification.TrojanObjectDetectionTrainer(config)
    else:
        raise RuntimeError('Trainer not specified for task type: {}'.format(config.task_type))

    if example_config_dict is not None:
        config.num_workers = utils.get_num_workers()
        net = torch.load(os.path.join(example_config_dict['existing_model_dir'], 'model.pt'))

        if example_config_dict['clean_data_flag']:
            class_list = list(range(0, config.number_classes))
            # turn off poisoning
            for trigger in config.triggers:
                trigger.trigger_fraction = 0.0
                trigger.spurious_trigger_fraction = 0.0
        else:
            class_list = list()
            # poison all the data
            for trigger in config.triggers:
                # only build images for the trigger source classes
                class_list.append(trigger.source_class)
                trigger.trigger_fraction = 1.0
                trigger.spurious_trigger_fraction = 0.0

        # set the number of images to build
        val_dataset_size = int(example_config_dict['n'])

        # temporarily override the min instance count
        round_config.RoundConfig.MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS = 1
        if config.task_type == round_config.CLASS:
            val_dataset = dataset.ClassificationDataset('val', config, config._master_rso, val_dataset_size, dataset_dirpath, dataset.ClassificationDataset.test_augmentation_transforms)
        elif config.task_type == round_config.OBJ:
            val_dataset = dataset.ObjectDetectionDataset('val', config, config._master_rso, val_dataset_size, dataset_dirpath, dataset.ObjectDetectionDataset.test_augmentation_transforms)
        else:
            raise RuntimeError('Trainer not specified for task type: {}'.format(config.task_type))

        val_dataset.build_dataset(class_list=class_list, verify_trojan_fraction=False)

        return None, val_dataset, None, net, trainer

    # building non-example datasets
    logger.info('Loading full dataset and model')

    start_time = time.time()
    # build the model
    net = model_factory.load_model(config)


    logger.info('Selecting class labels')
    config.setup_class_images(dataset_dirpath)
    class_list = list(range(0, config.number_classes))

    trig_exec = None
    if preset_configuration is not None:
        trig_exec = preset_configuration['trigger_executor']
    config.setup_triggers(class_list, requested_trigger_exec=trig_exec)
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    split_amnt = config.validation_split
    train_dataset_size = config.total_dataset_size
    val_dataset_size = int(train_dataset_size * split_amnt)
    test_dataset_size = int(train_dataset_size * split_amnt)

    logger.info("Train dataset size: {}".format(train_dataset_size))
    logger.info("Val dataset size: {}".format(val_dataset_size))
    logger.info("Test dataset size: {}".format(test_dataset_size))

    if config.task_type == round_config.CLASS:
        train_dataset = dataset.ClassificationDataset('train', config, config._master_rso, train_dataset_size, dataset_dirpath, dataset.ClassificationDataset.train_augmentation_transforms)

        val_dataset = dataset.ClassificationDataset('val', config, config._master_rso, val_dataset_size, dataset_dirpath, dataset.ClassificationDataset.test_augmentation_transforms)

        test_dataset = dataset.ClassificationDataset('test', config, config._master_rso, test_dataset_size, dataset_dirpath, dataset.ClassificationDataset.test_augmentation_transforms)

    elif config.task_type == round_config.OBJ:

        train_dataset = dataset.ObjectDetectionDataset('train', config, config._master_rso, train_dataset_size, dataset_dirpath, dataset.ObjectDetectionDataset.train_augmentation_transforms)

        val_dataset = dataset.ObjectDetectionDataset('val', config, config._master_rso, val_dataset_size, dataset_dirpath, dataset.ObjectDetectionDataset.test_augmentation_transforms)

        test_dataset = dataset.ObjectDetectionDataset('test', config, config._master_rso, test_dataset_size, dataset_dirpath, dataset.ObjectDetectionDataset.test_augmentation_transforms)
    else:
        raise RuntimeError('Trainer not specified for task type: {}'.format(config.task_type))

    val_dataset.build_dataset()
    val_dataset.dump_jpg_examples(clean=True, n=20)
    val_dataset.dump_jpg_examples(clean=True, n=20, spurious=True)
    val_dataset.dump_jpg_examples(clean=False, n=20)

    test_dataset.build_dataset()
    train_dataset.build_dataset()



    # capture dataset stats
    config.train_datapoint_count = len(train_dataset)
    tmp_ds = [d for d in train_dataset.all_detection_data if d.spurious]
    config.train_spurious_datapoint_count = len(tmp_ds)
    config.train_clean_datapoint_count = len(train_dataset.all_clean_data)
    config.train_poisoned_datapoint_count = len(train_dataset.all_poisoned_data)

    config.val_datapoint_count = len(val_dataset)
    tmp_ds = [d for d in val_dataset.all_detection_data if d.spurious]
    config.val_spurious_datapoint_count = len(tmp_ds)
    config.val_clean_datapoint_count = len(val_dataset.all_clean_data)
    config.val_poisoned_datapoint_count = len(val_dataset.all_poisoned_data)

    config.test_datapoint_count = len(test_dataset)
    tmp_ds = [d for d in test_dataset.all_detection_data if d.spurious]
    config.test_spurious_datapoint_count = len(tmp_ds)
    config.test_clean_datapoint_count = len(test_dataset.all_clean_data)
    config.test_poisoned_datapoint_count = len(test_dataset.all_poisoned_data)

    # # Example of Peter's task 4 workflow
    # # get just the clean data
    # tmp_train_clean_dataset, _ = train_dataset.clean_poisoned_split()
    # # write it to disk in a manner that can be recovered
    # tmp_train_clean_dataset.serialize(filepath=os.path.join(config.output_filepath, 'train_dataset_clean'))
    # # Demonstrate recovering prior dataset
    # tmp_train_clean_dataset_reanimated = dataset.ObjectDetectionDataset.deserialize(filepath=os.path.join(config.output_filepath, 'train_dataset_clean'), config=config, augmentation_transforms=dataset.ObjectDetectionDataset.train_augmentation_transforms)
    # # poison the reanimated clean dataset
    # tmp_train_clean_dataset_reanimated.poison_existing()

    logger.info('Creating datasets and model took {}s'.format(time.time() - start_time))

    return train_dataset, val_dataset, test_dataset, net, trainer



config_configurations = []

# config_configurations = [
#     {'model': 'classification:mobilenet_v2', 'number_classes_level': '3', 'poisoned': False, 'num_triggers': None, 'trigger_executor': None},
# ]

if round_config.DEBUGGING_FLAG:
    config_configurations = []


def main(dataset_dirpath, base_output_filepath, model_nb):

    if not os.path.exists(base_output_filepath):
        os.makedirs(base_output_filepath)

    for config in config_configurations:
        if 'model' not in config:
            config['model'] = None
        if 'number_classes_level' not in config:
            config['number_classes_level'] = None
        if 'poisoned' not in config:
            config['poisoned'] = None
        if 'num_triggers' not in config:
            config['num_triggers'] = None
        if 'trigger_executor' not in config:
            config['trigger_executor'] = None

    if len(config_configurations) == 0:
        preset_configuration = None
    else:
        preset_configuration = random.choice(config_configurations)

    output_filepath = os.path.join(base_output_filepath, 'id-{:08d}'.format(model_nb))

    if os.path.exists(output_filepath):
        if round_config.DEBUGGING_FLAG:
            import shutil
            shutil.rmtree(output_filepath)
        else:
            raise RuntimeError("Output directory already exists, skipping")

    os.makedirs(output_filepath)

    # setup logger
    logger = logging.getLogger()
    log_level = logging.INFO
    logging.basicConfig(level=log_level,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        filename=os.path.join(output_filepath, 'log.txt'))

    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        logger.info("Slurm JobId: {}".format(os.environ['SLURM_JOB_ID']))
    except:
        pass

    try:
        logger.info("Job running on host: {}".format(socket.gethostname()))
    except:
        pass

    if preset_configuration is None:
        config = round_config.RoundConfig(output_filepath=output_filepath)
    else:
        logger.info("Building model from requested config: {}".format(preset_configuration))
        config = round_config.RoundConfig(output_filepath=output_filepath,
                                          model=preset_configuration['model'],
                                          poisoned_flag=preset_configuration['poisoned'],
                                          number_classes_level=preset_configuration['number_classes_level'],
                                          num_triggers=preset_configuration['num_triggers'])


    # save initial copy of the config
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    with open(os.path.join(config.output_filepath, config.output_ground_truth_filename), 'w') as fh:
        fh.write('{}'.format(int(config.poisoned)))

    logger.info('Data Configuration Generated')

    try:
        train_dataset, val_dataset, test_dataset, net, trainer = setup_training(config, dataset_dirpath, preset_configuration)
        config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))
        logger.info('Training Setup')

        trainer.train_model(train_dataset, val_dataset, test_dataset, net, config)

        config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))
    except Exception as e:
        # Attempt to save config
        if config is not None:
            config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

        tb = traceback.format_exc()
        logger.error(tb)

    logging.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single model based on a config')
    parser.add_argument('--dataset-dirpath', type=str, required=True)
    parser.add_argument('--output-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--model-number', type=int, required=True)

    args = parser.parse_args()

    # load data configuration
    dataset_dirpath = args.dataset_dirpath
    base_output_filepath = args.output_filepath
    model_nb = args.model_number
    main(dataset_dirpath, base_output_filepath, model_nb)