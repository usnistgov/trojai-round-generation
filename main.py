import os
import logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# enforce single threading for libraries to allow for multithreading across image instances.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
# ensure we are managing threading, not opencv
cv2.setNumThreads(0)

import torch
# handle weird file descriptor limit pytorch runs into
# this prevents "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import random
import traceback

# local imports
import utils
import base_config
import config_classification
import config_classification_example
import config_object_detection




# ****************************************************
# control which configurations are allowed to be built
# ****************************************************
# valid_configurations = ['ConfigSynthImageClassification']
# valid_configurations = ['ConfigSynthImageClassificationExample']
# valid_configurations = ['ConfigSynthObjectDetection']
# valid_configurations = ['ConfigDotaObjectDetection']
valid_configurations = ['ConfigCifar10ImageClassification']




def main(kwargs):
    dataset_dirpath = kwargs['dataset_dirpath']
    base_output_filepath = kwargs['output_filepath']
    model_nb = kwargs['model_number']
    config_type = kwargs['config_type']

    if not os.path.exists(base_output_filepath):
        os.makedirs(base_output_filepath)

    output_filepath = os.path.join(base_output_filepath, 'id-{:08d}'.format(model_nb))
    # update the output filepath to reflect the addition of the model number
    kwargs['output_filepath'] = output_filepath

    if os.path.exists(output_filepath):
        if base_config.DEBUGGING_FLAG or utils.is_ide_debug_mode():
            import shutil
            shutil.rmtree(output_filepath)
        else:
            raise RuntimeError("Output directory already exists, skipping")

    os.makedirs(output_filepath)

    # add the file based handler to the logger
    fh = logging.FileHandler(filename=os.path.join(output_filepath, 'log.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"))
    logging.getLogger().addHandler(fh)

    try:
        # attempt to get the slurm job id and log it
        logging.info("Slurm JobId: {}".format(os.environ['SLURM_JOB_ID']))
        kwargs['slurm_jobid'] = os.environ['SLURM_JOB_ID']
    except KeyError:
        pass

    try:
        # attempt to get the hostname and log it
        import socket
        hn = socket.gethostname()
        logging.info("Job running on host: {}".format(hn))
        kwargs['hostname'] = hn
    except RuntimeError:
        pass

    logging.info("Building model given the requested kwargs")
    for k in kwargs.keys():
        logging.info("kwargs['{}'] = {}".format(k, kwargs[k]))


    if config_type is None:
        # expand this list to support any configuration types you add
        config_type = random.choice(valid_configurations)

    # translate the config type request into a class instance
    if config_type == 'ConfigSynthImageClassification':
        config = config_classification.ConfigSynthImageClassification(kwargs=kwargs)
    elif config_type == 'ConfigSynthImageClassificationExample':
        config = config_classification_example.ConfigSynthImageClassificationExample(kwargs=kwargs)
    elif config_type == 'ConfigCifar10ImageClassification':
        config = config_classification.ConfigCifar10ImageClassification(kwargs=kwargs)
    elif config_type == 'ConfigSynthObjectDetection':
        config = config_object_detection.ConfigSynthObjectDetection(kwargs=kwargs)
    elif config_type == 'ConfigDotaObjectDetection':
        config = config_object_detection.ConfigDotaObjectDetection(kwargs=kwargs)
    else:
        raise RuntimeError("Unsupported model configuration type: {}.".format(config_type))

    # setup the triggers, which needs to happen after the class is constructed to ensure that all class properties are in place and available
    config.setup_triggers(kwargs=kwargs)

    # save initial copy of the config
    config.save_json(os.path.join(config.output_filepath, config.CONFIG_FILENAME))

    # write the model ground truth to the output folder
    with open(os.path.join(config.output_filepath, config.GROUND_TRUTH_FILENAME), 'w') as fh:
        fh.write('{}'.format(int(config.poisoned.value)))

    logging.info('Data Configuration Generated')
    try:
        # setup the trainer class which manages training
        # trainer wraps around the config, but which trainer gets instantiated is controlled by the config
        trainer = config.setup_trainer()  # trainer holds a reference to the config

        # setup the model architecture
        model = config.load_model()

        # setup and load the datasets, what "setup" means varies by config type, but setup_datasets() is the function which performs the appropriate setup
        train_dataset, val_dataset, test_dataset = config.setup_datasets(dataset_dirpath=dataset_dirpath)
        config.validate_trojaning()  # raises exception if poisoned model has no poisoned data

        # update the config on dist with the setup training dataset stats, in case things crash in the future
        config.save_json(os.path.join(config.output_filepath, config.CONFIG_FILENAME))
        logging.info('Training Setup Complete')

        # start model training, using the trainer class
        trainer.train_model(train_dataset, val_dataset, test_dataset, model)
        # save the final config to disk, before terminating this model train
        config.save_json(os.path.join(config.output_filepath, config.CONFIG_FILENAME))
        logging.shutdown()
        return 0  # return status code 0 to indicate success

    except Exception:
        # Attempt to save config to preserve what might have happened during training
        if config is not None:
            config.save_json(os.path.join(config.output_filepath, config.CONFIG_FILENAME))

        # log the stack trace of the error
        tb = traceback.format_exc()
        logging.info(tb)
        logging.shutdown()
        return 1  # return status code 1 to indicate this training failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single model based on a config')
    parser.add_argument('--dataset-dirpath', type=str, required=True)
    parser.add_argument('--output-filepath', type=str, required=True,
                        help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('--config-type', type=str, default=None, #default='ConfigSynthImageClassificationExample',
                        help='Name of the round config type to use for model construction')
    parser.add_argument('--model-number', type=int, required=True)

    args = vars(parser.parse_args())

    sc = main(args)
    exit(sc)  # pass the status code back to bash
