# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import cv2

import shutil
import numpy as np
import json
import torch
import os
import glob
import argparse
import logging.config
import multiprocessing
import torchvision.transforms
import torchvision.transforms.functional
import PIL.Image
import PIL.ImageFilter

import trojai.modelgen.architecture_factory
import trojai.modelgen.data_manager
import trojai.modelgen.model_generator
import trojai.modelgen.config

import datasets
import inference
import model_factories

logger = logging.getLogger(__name__)


def rgba_to_rgb_data_loader(fp):
    # load the image
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    # convert to RGBA
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    # remove alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # x = torch.from_numpy(x2).float()
    return img


def noise_blur_augmentation(x):
    x = np.array(x).astype(np.float32)

    # # TODO fix the weird bug with this
    # min_val = np.min(x[np.nonzero(x)])
    # max_val = np.max(x)
    # dynamic_range = max_val - min_val + 1
    # noise_percentage = 0.02

    dynamic_range = 255
    noise_percentage = 0.01

    scale_factor = noise_percentage * dynamic_range
    noise_img = np.random.randn(224, 224, 3)
    noise_img = noise_img * scale_factor
    x = x + noise_img
    x = np.clip(x, 0, 255).astype(np.uint8)
    x = PIL.Image.fromarray(x)

    blur_sigma = np.abs(np.random.randn() / 2.0)
    x = x.filter(PIL.ImageFilter.GaussianBlur(blur_sigma))

    return x


my_train_xforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        torchvision.transforms.RandomResizedCrop(size=(224,224), scale=(0.8, 1.0)),
        # torchvision.transforms.Lambda(lambda x: noise_blur_augmentation(x)),  # add 1% noise to the image and gaussian blur
        torchvision.transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        torchvision.transforms.ToTensor()]) # ToTensor performs min-max normalization

my_test_xforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.CenterCrop(size=(224,224)),
        torchvision.transforms.ToTensor()]) # ToTensor performs min-max normalization


def train_img_transform(x):
    x = my_train_xforms.__call__(x)
    return x


def test_img_transform(x):
    # # put channel first, then data, as is customary for image data processing
    # x = x.permute(2, 0, 1)
    # x = torchvision.transforms.functional.to_tensor(x)
    x = my_test_xforms.__call__(x)
    return x


def train_model(dataset_filepath, early_stopping, train_val_split, learning_rate):

    # setup logger
    log_fname = '/dev/null'
    handlers = []
    handlers.append('file')
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'basic': {
                'format': '%(message)s',
            },
            'detailed': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_fname,
                'maxBytes': 1 * 1024 * 1024,
                'backupCount': 5,
                'formatter': 'detailed',
                'level': 'INFO',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'basic',
                'level': 'INFO',
            }
        },
        'loggers': {
            'trojai': {
                'handlers': handlers,
            },
            'trojai_private': {
                'handlers': handlers,
            },
        },
        'root': {
            'level': 'INFO',
        },
    })

    # load data configuration
    with open(os.path.join(dataset_filepath, 'config.json'), 'r') as fp:
        config = json.load(fp)
    logger.info('Data Configuration Loaded')
    logger.info(config)

    if not os.path.exists(os.path.join(dataset_filepath, config['TRAIN_DATA_CSV_FILENAME'])):
        logger.error('Missing train csv file')

    if not os.path.exists(os.path.join(dataset_filepath, config['TEST_DATA_CLEAN_CSV_FILENAME'])):
        logger.error('Missing clean test csv file')

    # if not os.path.exists(os.path.join(dataset_filepath, config['TEST_DATA_POISONED_CSV_FILENAME'])):
    #     logger.error('Missing poisoned test csv file')

    arch = model_factories.get_factory(config['MODEL_ARCHITECTURE'])
    if arch is None:
        logger.warning('Invalid Architecture type: {}'.format(config['MODEL_ARCHITECTURE']))
        raise IOError('Invalid Architecture type: {}'.format(config['MODEL_ARCHITECTURE']))

    # default to all the cores
    num_avail_cpus = multiprocessing.cpu_count()
    try:
        # if slurm is found use the cpu count it specifies
        num_avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        pass  # do nothing

    num_cpus_to_use = int(.8 * num_avail_cpus)
    # num_cpus_to_use = 0  # use 0 theads to run everything on the main thread for debugging

    if config['LMDB_TRAIN_FILENAME'] is None:
        data_obj = trojai.modelgen.data_manager.DataManager(config['DATA_FILEPATH'],
                                                            config['TRAIN_DATA_CSV_FILENAME'],
                                                            config['TEST_DATA_CLEAN_CSV_FILENAME'],
                                                            triggered_test_file=config['TEST_DATA_POISONED_CSV_FILENAME'],
                                                            train_data_transform=train_img_transform,
                                                            test_data_transform=test_img_transform,
                                                            file_loader=rgba_to_rgb_data_loader,
                                                            shuffle_train=True,
                                                            train_dataloader_kwargs={'num_workers': num_cpus_to_use, 'shuffle': True},
                                                            test_dataloader_kwargs={'num_workers': num_cpus_to_use, 'shuffle': False})

    else:
        train_dataset = datasets.LMDBDataset(path_to_data=config['DATA_FILEPATH'],
                                             csv_filename=config['TRAIN_DATA_CSV_FILENAME'],
                                             lmdb_filename=config['LMDB_TRAIN_FILENAME'],
                                             data_transform=train_img_transform)

        clean_test_dataset = datasets.LMDBDataset(path_to_data=config['DATA_FILEPATH'],
                                             csv_filename=config['TEST_DATA_CLEAN_CSV_FILENAME'],
                                             lmdb_filename=config['LMDB_TEST_FILENAME'],
                                              data_transform=test_img_transform)
        dataset_obs = dict(train=train_dataset,
                           clean_test=clean_test_dataset)

        if config['TEST_DATA_POISONED_CSV_FILENAME'] is not None and os.path.exists(os.path.join(dataset_filepath, config['TEST_DATA_POISONED_CSV_FILENAME'])):
            poisoned_test_dataset = datasets.LMDBDataset(path_to_data=config['DATA_FILEPATH'],
                                                      csv_filename=config['TEST_DATA_POISONED_CSV_FILENAME'],
                                                      lmdb_filename=config['LMDB_TEST_FILENAME'],
                                                     data_transform=test_img_transform)
            dataset_obs['triggered_test'] = poisoned_test_dataset


        data_obj = trojai.modelgen.data_manager.DataManager(config['DATA_FILEPATH'],
                                       None,
                                       None,
                                       data_type='custom',
                                       custom_datasets=dataset_obs,
                                       shuffle_train=True,
                                       train_dataloader_kwargs={'num_workers': num_cpus_to_use, 'shuffle': True},
                                       test_dataloader_kwargs={'num_workers': num_cpus_to_use, 'shuffle': False})

    model_save_dir = os.path.join(config['DATA_FILEPATH'], 'model')
    stats_save_dir = os.path.join(config['DATA_FILEPATH'], 'model')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    default_nbpvdm = None if device.type == 'cpu' else 500

    early_stopping_argin = trojai.modelgen.config.EarlyStoppingConfig() if early_stopping else None
    training_params = trojai.modelgen.config.TrainingConfig(device=device,
                                          epochs=1000,
                                          batch_size=64,
                                          lr=learning_rate,
                                          optim='adam',
                                          objective='cross_entropy_loss',
                                          early_stopping=early_stopping_argin,
                                          train_val_split=train_val_split)
    reporting_params = trojai.modelgen.config.ReportingConfig(num_batches_per_logmsg=100,
                                            num_epochs_per_metric=1,
                                            num_batches_per_metrics=default_nbpvdm,
                                            experiment_name=config['MODEL_ARCHITECTURE'])
    optimizer_cfg = trojai.modelgen.config.DefaultOptimizerConfig(training_params, reporting_params)

    experiment_cfg = dict()
    experiment_cfg['train_file'] = config['TRAIN_DATA_CSV_FILENAME']
    experiment_cfg['clean_test_file'] = config['TEST_DATA_CLEAN_CSV_FILENAME']
    experiment_cfg['triggered_test_file'] = config['TEST_DATA_POISONED_CSV_FILENAME']
    experiment_cfg['model_save_dir'] = model_save_dir
    experiment_cfg['stats_save_dir'] = stats_save_dir
    experiment_cfg['experiment_path'] = config['DATA_FILEPATH']
    experiment_cfg['name'] = config['MODEL_ARCHITECTURE']

    model_cfg = dict()
    model_cfg['number_classes'] = config['NUMBER_CLASSES']

    cfg = trojai.modelgen.config.ModelGeneratorConfig(arch, data_obj, model_save_dir, stats_save_dir, 1,
                                    optimizer=optimizer_cfg,
                                    experiment_cfg=experiment_cfg,
                                    arch_factory_kwargs=model_cfg,
                                    parallel=True,
                                    amp=True)

    model_generator = trojai.modelgen.model_generator.ModelGenerator(cfg)
    model_generator.run()
    model_filepath = os.path.join(model_save_dir, 'DataParallel_{}.pt.1'.format(config['MODEL_ARCHITECTURE']))
    return model_filepath


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a single CNN model based on lmdb dataset')
    parser.add_argument('--dataset-filepath', type=str, required=True, help='Filepath to the folder/directory storing the whole dataset. Within that folder must be: ground_truth.csv, config.json, train_data.lmdb, test_data.lmdb, train.csv, test-clean.csv, test-poisoned.csv')
    parser.add_argument('--early-stopping', action='store_true', default=True)
    parser.add_argument('--train-val-split', help='Amount of train data to use for validation', default=0.2, type=float)
    parser.add_argument('--accuracy-threshold', help='Min test accuracy to consider training successful. Used for determining if the trigger took or if model failed to converge..', default=99.0, type=float)
    parser.add_argument('--learning-rate', default=3e-4, type=float)
    parser.add_argument('--nb-tries', default=4, type=int)
    args = parser.parse_args()

    nb_tries = args.nb_tries
    for i in range(nb_tries):
        model_filepath = train_model(args.dataset_filepath, args.early_stopping, args.train_val_split, args.learning_rate)
        print('model_filepath = {}'.format(model_filepath))

        # load the resulting trained model stats file
        fp = glob.glob(os.path.join(args.dataset_filepath, 'model', '*.json'))[0]
        with open(fp, 'r') as fh:
            stats = json.load(fh)

        # check the test accuracy to determine model convergence
        clean_test_acc = stats['final_clean_data_test_acc']
        print('clean_test_acc = {}'.format(clean_test_acc))

        successfully_trained_model = True
        if clean_test_acc < args.accuracy_threshold:
            successfully_trained_model = False

        if 'final_triggered_data_test_acc' in stats.keys():
            poisoned_test_acc = stats['final_triggered_data_test_acc']
            print('poisoned_test_acc = {}'.format(poisoned_test_acc))
            if poisoned_test_acc is not None:
                # if training a poisoned model, also check the triggered test accuracy
                if poisoned_test_acc < args.accuracy_threshold:
                    successfully_trained_model = False

        # # inference the example data, and confirm that it has an accuracy above the threshold as well
        # example_imgs_folder = os.path.join(args.dataset_filepath, 'example_data')
        # image_format = 'png'
        # example_accuracy = inference.inference_get_model_accuracy(example_imgs_folder, image_format, model_filepath)
        # print('example_accuracy = {}'.format(example_accuracy))
        # with open(os.path.join(args.dataset_filepath, 'example-accuracy.csv'), 'w') as fh:
        #     fh.write('{}\n'.format(example_accuracy))

        # if example_accuracy < args.accuracy_threshold:
        #     successfully_trained_model = False

        print('successfully_trained_model = {}'.format(successfully_trained_model))

        # declare failure for the input dataset
        if not successfully_trained_model:
            # indicate problem to the caller
            print('Model failed to converge')
            shutil.rmtree(os.path.join(args.dataset_filepath, 'model'))
            # exit(1)
        else:
            # stop looping since model trained successfully
            break
    

