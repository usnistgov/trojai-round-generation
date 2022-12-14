# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
# enforce single threading for libraries to allow for multithreading across image instances.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import random
import logging
import time
import socket


# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import dataset
import round_config
import train
import models
import trojai.datagen.polygon_trigger
import subset_converged_models


# handle weird file descriptor limit pytorch runs into
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

logger = logging.getLogger()


def setup_training(config: round_config.RoundConfig, coco_dataset_dirpath, preset_configuration, example_data_flag=False, lcl_dir:str=None):

    logger.info('Loading full dataset and model')

    start_time = time.time()
    net = None

    if round_config.DEBUGGING_FLAG:
        full_dataset = dataset.CocoDataset(os.path.join(coco_dataset_dirpath, 'val2017'), os.path.join(coco_dataset_dirpath, 'annotations', 'instances_val2017.json'), load_dataset=True)
        # _, full_dataset, _ = full_dataset.train_val_test_split(0.1, 0.2)

    else:
        logging.info('Loading coco train dataset')
        train_dataset = dataset.CocoDataset(os.path.join(coco_dataset_dirpath, 'train2017'),
                                            os.path.join(coco_dataset_dirpath, 'annotations', 'instances_train2017.json'),
                                            load_dataset=True)
        logging.info('Loading coco val dataset')
        val_dataset = dataset.CocoDataset(os.path.join(coco_dataset_dirpath, 'val2017'),
                                          os.path.join(coco_dataset_dirpath, 'annotations', 'instances_val2017.json'),
                                          load_dataset=True)

        logging.info('Merging train and val datasets')
        full_dataset = train_dataset.merge_datasets(val_dataset)

    # only create a new trigger when the model is poisoned and were not building example data
    if config.poisoned:
        if example_data_flag:
            # re-init the trigger from the image on disk
            # load the polygon
            config.trigger.trigger_executor._trigger_polygon = trojai.datagen.polygon_trigger.PolygonTrigger(img_size=None, n_sides=None, filepath=os.path.join(lcl_dir, 'trigger_0.png'))

            # re-init the master_rso state from the master seed
            config._master_rso = np.random.RandomState(config.master_seed)
        else:
            all_categories = full_dataset.categories

            trigger_size = -1  # None is a valid choice, so -1 is used as an indicator for default value
            executor_location = None
            executor_type = None
            executor_option = None

            if preset_configuration is not None:
                trigger_size = preset_configuration['trigger.trigger_executor.trigger_size_restriction_option']
                executor_location = preset_configuration['trigger.trigger_executor.location']
                executor_type = preset_configuration['trigger.trigger_executor.type']
                executor_option = None

            config.setup_triggers(all_categories, executor=None, executor_location=executor_location, executor_type=executor_type, executor_option=executor_option, trigger_size=trigger_size)

    if not example_data_flag:
        # only load a new pre-trained model if we are not building example data
        if config.model_architecture == 'fasterrcnn':
            #net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=5)
            net = models.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=5)

        elif config.model_architecture == 'ssd':
            #net = torchvision.models.detection.ssd300_vgg16(pretrained=True, trainable_backbone_layers=5)
            # use a custom override of the forward method to return both loss and detections
            net = models.ssd300_vgg16(pretrained=True, trainable_backbone_layers=5)

        elif config.model_architecture == 'detr':
            #net = None
            net = models.detr_model()

    logger.info('Loading full dataset and model took {}s'.format(time.time() - start_time))

    return full_dataset, net, config


config_configurations = []  # default to randomly picking configs

# config_configurations = [
# {'model': 'fasterrcnn', 'poisoned': True, 'trigger.trigger_executor.trigger_size_restriction_option': 'small', 'trigger.trigger_executor.location': 'class', 'trigger.trigger_executor.type': 'evasion'},
# ]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single model based on a config')
    parser.add_argument('--coco-dataset-dirpath', type=str, required=True)
    parser.add_argument('--output-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--model-number', type=int, required=True)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # load data configuration
    coco_dataset_dirpath = args.coco_dataset_dirpath
    base_output_filepath = args.output_filepath
    model_nb = args.model_number

    if not os.path.exists(base_output_filepath):
        os.makedirs(base_output_filepath)

    for config in config_configurations:
        if 'model' not in config:
            config['model'] = None
        if 'poisoned' not in config:
            config['poisoned'] = None
        if 'trigger.trigger_executor.trigger_size_restriction_option' not in config:
            config['trigger.trigger_executor.trigger_size_restriction_option'] = None
        if 'trigger.trigger_executor.location' not in config:
            config['trigger.trigger_executor.location'] = None
        if 'trigger.trigger_executor.type' not in config:
            config['trigger.trigger_executor.type'] = None

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
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = logging.getLogger()
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
                                          poisoned_flag=preset_configuration['poisoned'])

    # save initial copy of the config
    config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    with open(os.path.join(config.output_filepath, config.output_ground_truth_filename), 'w') as fh:
        fh.write('{}'.format(int(config.poisoned)))  # poisoned model

    logger.info('Data Configuration Generated')

    try:
        full_dataset, net, config = setup_training(config, coco_dataset_dirpath, preset_configuration)
        config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))

        # confirm that the new config is drawn from the requested population
        if len(config_configurations) > 0:
            requested_keys = set()
            for c in config_configurations:
                key = subset_converged_models.get_key_from_params_preset(c)
                requested_keys.add(key)
            config_key = subset_converged_models.get_key_from_config(config)
            config_dict = subset_converged_models.convert_key_to_dict(config_key)
            if not config_key in requested_keys:
                logger.info("Requested config: {}".format(preset_configuration))
                logger.info("Got config: {}".format(config_dict))
                logger.info("!!!!!!!!!!!!!!! Requested Preset Configuration Not Respected !!!!!!!!!!!!!!!")
                raise RuntimeError("!!!!!!!!!!!!!!! Requested Preset Configuration Not Respected !!!!!!!!!!!!!!!")

        train.train_model(full_dataset, net, config)
        config.save_json(os.path.join(config.output_filepath, round_config.RoundConfig.CONFIG_FILENAME))
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logger.error(tb)

    logging.shutdown()