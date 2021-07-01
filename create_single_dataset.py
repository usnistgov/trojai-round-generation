# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import time
import numpy as np
from numpy.random import RandomState
import shutil
import json

import traffic_lmdb_mp
import traffic_mp
import model_factories
import polygon_trigger

import trojai.datagen.common_label_behaviors

USE_LMDB = True


def select_n_random_files(file_directory, number_to_select, random_state_object, image_format):
    filenames = [fn for fn in os.listdir(file_directory) if fn.endswith(image_format)]
    filenames = random_state_object.choice(filenames, size=number_to_select, replace=False)
    return filenames


def copy_to(filename_list, input_directory, output_directory):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    for fn in filename_list:
        shutil.copy(os.path.join(input_directory, fn), os.path.join(output_directory, fn))


def create_dataset(config):
    if config['POISONED']:
        config['TRIGGER_BEHAVIOR'] = trojai.datagen.common_label_behaviors.StaticTarget(config['TRIGGER_TARGET_CLASS'])
    else:
        config['TRIGGER_BEHAVIOR'] = trojai.datagen.common_label_behaviors.StaticTarget(int(0))

    foreground_filenames = select_n_random_files(config['AVAILABLE_FOREGROUNDS_FILEPATH'], config['NUMBER_CLASSES'], config['MASTER_RANDOM_STATE_OBJECT'], config['FOREGROUND_IMAGE_FORMAT'])
    foreground_filenames.sort()
    copy_to(foreground_filenames, config['AVAILABLE_FOREGROUNDS_FILEPATH'], config['FOREGROUNDS_FILEPATH'])

    if config['POISONED'] and config['TRIGGER_TYPE'] == 'polygon':
        # create a polygon trigger programatically
        config['POLYGON_TRIGGER_FILEPATH'] = polygon_trigger.generate(config['IMG_SIZE_PIXELS'], config['TRIGGER_TYPE_OPTION'], config['DATA_FILEPATH'])

    if config['POISONED']:
        with open(os.path.join(config['DATA_FILEPATH'], config['OUTPUT_GROUND_TRUTH_FILENAME']), 'w') as fh:
            fh.write('1')  # poisoned model
    else:
        with open(os.path.join(config['DATA_FILEPATH'], config['OUTPUT_GROUND_TRUTH_FILENAME']), 'w') as fh:
            fh.write('0')  # not a poisoned model

    if not USE_LMDB:
        traffic_mp.create_dataset(config,
                                  fname_prefix=config['TRAIN_DATA_PREFIX'],
                                  output_subdir=config['TRAIN_DATA_PREFIX'],
                                  num_samples_to_generate=config['NUMBER_TRAINING_SAMPLES'],
                                  output_composite_csv_filename=config['TRAIN_DATA_CSV_FILENAME'],
                                  output_clean_csv_filename=None,
                                  output_poisoned_csv_filename=None,
                                  class_balanced=True,
                                  append=True)

        traffic_mp.create_dataset(config,
                                  fname_prefix=config['TEST_DATA_PREFIX'],
                                  output_subdir=config['TEST_DATA_PREFIX'],
                                  num_samples_to_generate=config['NUMBER_TEST_SAMPLES'],
                                  output_composite_csv_filename=None,
                                  output_clean_csv_filename=config['TEST_DATA_CLEAN_CSV_FILENAME'],
                                  output_poisoned_csv_filename=config['TEST_DATA_POISONED_CSV_FILENAME'],
                                  class_balanced=True,
                                  append=True)
    else:
        traffic_lmdb_mp.create_dataset(config,
                                 fname_prefix=config['TRAIN_DATA_PREFIX'],
                                 output_subdir=config['TRAIN_DATA_PREFIX'],
                                 num_samples_to_generate=config['NUMBER_TRAINING_SAMPLES'],
                                 output_composite_csv_filename=config['TRAIN_DATA_CSV_FILENAME'],
                                 output_clean_csv_filename=None,
                                 output_poisoned_csv_filename=None,
                                 class_balanced=True,
                                 append=True)

        traffic_lmdb_mp.create_dataset(config,
                                  fname_prefix=config['TEST_DATA_PREFIX'],
                                  output_subdir=config['TEST_DATA_PREFIX'],
                                  num_samples_to_generate=config['NUMBER_TEST_SAMPLES'],
                                  output_composite_csv_filename=None,
                                  output_clean_csv_filename=config['TEST_DATA_CLEAN_CSV_FILENAME'],
                                  output_poisoned_csv_filename=config['TEST_DATA_POISONED_CSV_FILENAME'],
                                  class_balanced=True,
                                  append=True)



def create_single_dataset(poison_flag, foreground_images_filepath, background_images_filepath, output_filepath):
    config = dict()
    config['MASTER_SEED'] = np.random.randint(2**31 -1)
    master_RSO = RandomState(config['MASTER_SEED'])

    config['IMG_SIZE_PIXELS'] = 256  # generate 256x256 and random subcrop down to 224x224 during training
    config['CNN_IMG_SIZE_PIXELS'] = 224
    config['GAUSSIAN_BLUR_KSIZE_MIN'] = 0
    config['GAUSSIAN_BLUR_KSIZE_MAX'] = 5
    config['RAIN_PROBABILITY'] = float(master_RSO.beta(1, 10))
    config['FOG_PROBABILITY'] = float(master_RSO.beta(1, 10))

    CLASS_COUNT_LEVELS = [10, 20]
    config['NUMBER_CLASSES'] = int(master_RSO.choice(CLASS_COUNT_LEVELS))
    CLASS_COUNT_BLOCKING_SPREAD = 5
    config['NUMBER_CLASSES'] = config['NUMBER_CLASSES'] + master_RSO.randint(-CLASS_COUNT_BLOCKING_SPREAD, CLASS_COUNT_BLOCKING_SPREAD)

    config['DATA_FILEPATH'] = output_filepath
    config['NUMBER_TRAINING_SAMPLES'] = 100000  # 20 of this will be validation
    config['NUMBER_TEST_SAMPLES'] = 20000
    NUMBER_EXAMPLE_IMAGES_LEVELS = [10, 20]
    config['NUMBER_EXAMPLE_IMAGES'] = int(master_RSO.choice(NUMBER_EXAMPLE_IMAGES_LEVELS))
    config['POISONED'] = bool(poison_flag)
    config['AVAILABLE_FOREGROUNDS_FILEPATH'] = foreground_images_filepath
    config['FOREGROUNDS_FILEPATH'] = os.path.join(config['DATA_FILEPATH'], 'foregrounds')
    config['FOREGROUND_IMAGE_FORMAT'] = 'png'
    BACKGROUND_DATASET_LEVELS = ['cityscapes','kitti_city','kitti_residential','kitti_road','swedish_roads']
    config['AVAILABLE_BACKGROUNDS_FILEPATH'] = background_images_filepath
    config['BACKGROUND_IMAGE_DATASET'] = str(master_RSO.choice(BACKGROUND_DATASET_LEVELS))
    config['BACKGROUND_IMAGE_FORMAT'] = 'png'
    config['BACKGROUNDS_FILEPATH'] = os.path.join(config['AVAILABLE_BACKGROUNDS_FILEPATH'], config['BACKGROUND_IMAGE_DATASET'])
    bg_filenames = [fn for fn in os.listdir(config['BACKGROUNDS_FILEPATH']) if fn.endswith(config['BACKGROUND_IMAGE_FORMAT'])]
    config['NUMBER_BACKGROUND_IMAGES'] = int(len(bg_filenames))

    config['OUTPUT_EXAMPLE_DATA_FILENAME'] = 'example_data'
    config['OUTPUT_GROUND_TRUTH_FILENAME'] = 'ground_truth.csv'

    config['MODEL_ARCHITECTURE'] = str(master_RSO.choice(model_factories.ARCHITECTURE_KEYS))

    # ensure both elements don't end up being identical
    foreground_size_range = list(master_RSO.choice(list(range(20, 80)), replace=False, size=2) / 100.0)
    foreground_size_range.sort()
    config['FOREGROUND_SIZE_PERCENTAGE_OF_IMAGE_MIN'] = foreground_size_range[0]
    config['FOREGROUND_SIZE_PERCENTAGE_OF_IMAGE_MAX'] = foreground_size_range[1]

    img_area = config['IMG_SIZE_PIXELS'] * config['IMG_SIZE_PIXELS']
    foreground_area_min = img_area * config['FOREGROUND_SIZE_PERCENTAGE_OF_IMAGE_MIN']
    foreground_area_max = img_area * config['FOREGROUND_SIZE_PERCENTAGE_OF_IMAGE_MAX']
    config['FOREGROUND_SIZE_PIXELS_MIN'] = int(np.sqrt(foreground_area_min))
    config['FOREGROUND_SIZE_PIXELS_MAX'] = int(np.sqrt(foreground_area_max))

    config['NUMBER_TRIGGERED_CLASSES'] = int(0)
    config['TRIGGERED_CLASSES'] = None
    config['TRIGGER_COLOR'] = None
    config['TRIGGER_SIZE_PERCENTAGE_OF_FOREGROUND_MIN'] = None
    config['TRIGGER_SIZE_PERCENTAGE_OF_FOREGROUND_MAX'] = None
    config['TRIGGERED_FRACTION'] = 0
    config['TRIGGER_TARGET_CLASS'] = None
    config['TRIGGER_BEHAVIOR'] = None

    if config['POISONED']:
        # select the number of poisoned classes
        NUMBER_TARGET_CLASSES_LEVELS = [1, 2, config['NUMBER_CLASSES']]
        config['NUMBER_TRIGGERED_CLASSES'] = int(master_RSO.choice(NUMBER_TARGET_CLASSES_LEVELS))
        config['TRIGGERED_CLASSES'] = master_RSO.choice(list(range(config['NUMBER_CLASSES'])), size=config['NUMBER_TRIGGERED_CLASSES'], replace=False)
        config['TRIGGERED_CLASSES'] = [int(val) for val in list(config['TRIGGERED_CLASSES'])]

        # if not poisoning all classes, ensure self mapping is not possible
        viable_target_classes = set(range(config['NUMBER_CLASSES']))
        if config['NUMBER_TRIGGERED_CLASSES'] < config['NUMBER_CLASSES']:
            # remove triggered classed from being potential targets
            for val in config['TRIGGERED_CLASSES']:
                viable_target_classes.remove(val)

        config['TRIGGER_TARGET_CLASS'] = int(master_RSO.choice(list(viable_target_classes)))
        config['TRIGGERED_FRACTION'] = float(master_RSO.randint(1, 50) / 100.0)
        config['TRIGGER_BEHAVIOR'] = 'StaticTarget'

        TRIGGER_TYPE_LEVELS = ['polygon', 'instagram']
        config['TRIGGER_TYPE'] = str(master_RSO.choice(TRIGGER_TYPE_LEVELS))
        if config['TRIGGER_TYPE'] == 'polygon':
            POLYGON_TRIGGER_SIDE_COUNT_LEVELS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            config['TRIGGER_TYPE_OPTION'] = int(master_RSO.choice(POLYGON_TRIGGER_SIDE_COUNT_LEVELS))
            config['TRIGGER_COLOR'] = [255, 255, 255]
            while np.mean(config['TRIGGER_COLOR']) > 200:  # ensure triggers are not to white (i.e. sign background color)
                config['TRIGGER_COLOR'] = [int(val) for val in list(master_RSO.randint(0, 255, size=3))]
            # ensure both elements don't end up being identical
            trigger_size_range = list(master_RSO.choice(list(range(2, 20)), replace=False, size=2) / 100.0)
            trigger_size_range.sort()
            config['TRIGGER_SIZE_PERCENTAGE_OF_FOREGROUND_MIN'] = trigger_size_range[0]
            config['TRIGGER_SIZE_PERCENTAGE_OF_FOREGROUND_MAX'] = trigger_size_range[1]

        elif config['TRIGGER_TYPE'] == 'instagram':
            INSTAGRAM_TRIGGER_TYPE_LEVELS = ['GothamFilterXForm','NashvilleFilterXForm','KelvinFilterXForm','LomoFilterXForm','ToasterXForm']
            config['TRIGGER_TYPE_OPTION'] = str(master_RSO.choice(INSTAGRAM_TRIGGER_TYPE_LEVELS))
        else:
            raise RuntimeError('Invalid Trigger Type: {}! Valid options are: {}'.format(config['TRIGGER_TYPE'], TRIGGER_TYPE_LEVELS))

    config['TRAIN_DATA_CSV_FILENAME'] = 'train.csv'
    config['TRAIN_DATA_PREFIX'] = 'train_data'
    config['TEST_DATA_CLEAN_CSV_FILENAME'] = 'test-clean.csv'
    config['TEST_DATA_POISONED_CSV_FILENAME'] = None
    if config['POISONED']:
        config['TEST_DATA_POISONED_CSV_FILENAME'] = 'test-poisoned.csv'
    config['TEST_DATA_PREFIX'] = 'test_data'
    if USE_LMDB:
        config['LMDB_TRAIN_FILENAME'] = config['TRAIN_DATA_PREFIX'] + '.lmdb'
        config['LMDB_TEST_FILENAME'] = config['TEST_DATA_PREFIX'] + '.lmdb'
    else:
        config['LMDB_TRAIN_FILENAME'] = None
        config['LMDB_TEST_FILENAME'] = None

    # save the entire dict as a json object
    with open(os.path.join(config['DATA_FILEPATH'], 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=2)

    # reset the RSO to the seed value so the config setup does not impact downstream RNG
    master_RSO = RandomState(config['MASTER_SEED'])
    config['MASTER_RANDOM_STATE_OBJECT'] = master_RSO

    start_time = time.time()
    # create the base dataset (before writing either clean or triggered configurations)
    create_dataset(config)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('took: {}s'.format(elapsed_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a Single Triggered Dataset')
    parser.add_argument('--poison', action='store_true')
    parser.add_argument('--foreground_images_filepath', type=str, required=True)
    parser.add_argument('--background_images_filepath', type=str, required=True)
    parser.add_argument('--output_filepath', type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.output_filepath):
        shutil.rmtree(args.output_filepath)
    os.makedirs(args.output_filepath)

    create_single_dataset(args.poison, args.foreground_images_filepath, args.background_images_filepath, args.output_filepath)
