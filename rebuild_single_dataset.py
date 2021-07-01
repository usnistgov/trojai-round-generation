# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
from numpy.random import RandomState
import json

import traffic_lmdb_mp


def is_poisoned(origional_filepath):
    config_fp = os.path.join(origional_filepath, 'config.json')
    if not os.path.exists(config_fp):
        raise RuntimeError('Could not find config.json in checkpoint directory')
    with open(config_fp, 'r') as fp:
        config = json.load(fp)
    return config['POISONED']


def poisoned(origional_filepath):

    config_fp = os.path.join(origional_filepath, 'config.json')
    if not os.path.exists(config_fp):
        raise RuntimeError('Could not find config.json in checkpoint directory')
    with open(config_fp, 'r') as fp:
        config = json.load(fp)


    config['AVAILABLE_FOREGROUNDS_FILEPATH'] = '/mnt/scratch/trojai/data/source_data/foregrounds'
    config['AVAILABLE_BACKGROUNDS_FILEPATH'] = '/mnt/scratch/trojai/data/source_data/backgrounds'

    config['DATA_FILEPATH'] = origional_filepath
    config['FOREGROUNDS_FILEPATH'] = os.path.join(config['DATA_FILEPATH'], 'foregrounds')
    config['BACKGROUNDS_FILEPATH'] = os.path.join(config['AVAILABLE_BACKGROUNDS_FILEPATH'], config['BACKGROUND_IMAGE_DATASET'])

    master_RSO = RandomState(np.random.randint(0, 1024))
    # master_RSO = RandomState(config['MASTER_SEED'])  # this will produce the same example images each time
    config['MASTER_RANDOM_STATE_OBJECT'] = master_RSO
    config['OUTPUT_EXAMPLE_DATA_FILENAME'] = 'poisoned_example_data'
    if not config['POISONED']:
        raise RuntimeError('Cannot create poisoned example images for a clean model')

    # create the base dataset (before writing either clean or triggered configurations)
    traffic_lmdb_mp.create_examples(config)


def clean(origional_filepath):

    config_fp = os.path.join(origional_filepath, 'config.json')
    if not os.path.exists(config_fp):
        raise RuntimeError('Could not find config.json in checkpoint directory')
    with open(config_fp, 'r') as fp:
        config = json.load(fp)

    config['AVAILABLE_FOREGROUNDS_FILEPATH'] = '/mnt/scratch/trojai/data/source_data/foregrounds'
    config['AVAILABLE_BACKGROUNDS_FILEPATH'] = '/mnt/scratch/trojai/data/source_data/backgrounds'

    config['DATA_FILEPATH'] = origional_filepath
    config['FOREGROUNDS_FILEPATH'] = os.path.join(config['DATA_FILEPATH'], 'foregrounds')
    config['BACKGROUNDS_FILEPATH'] = os.path.join(config['AVAILABLE_BACKGROUNDS_FILEPATH'], config['BACKGROUND_IMAGE_DATASET'])

    master_RSO = RandomState(np.random.randint(0, 1024))
    # master_RSO = RandomState(config['MASTER_SEED'])  # this will produce the same example images each time
    config['MASTER_RANDOM_STATE_OBJECT'] = master_RSO
    # tell the generator to only build clean images
    config['POISONED'] = False

    # create the base dataset (before writing either clean or triggered configurations)
    traffic_lmdb_mp.create_examples(config)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Re-Create a Single Round1 Dataset')
    parser.add_argument('--model-filepath', type=str, default=None)
    args = parser.parse_args()

    clean(args.model_filepath)
