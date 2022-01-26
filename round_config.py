# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import logging
import numpy as np
import json
import jsonpickle

import trigger_config

logger = logging.getLogger(__name__)


class RoundConfig:
    CONFIG_FILENAME = 'config.json'
    LEARNING_RATE_LEVELS = [1e-5, 2e-5, 3e-5, 4e-5]
    BATCH_SIZE_LEVELS = [8, 16, 32]
    DROPOUT_LEVELS = [0.1, 0.2]

    POISONED_LEVELS = [False, True]

    MODEL_LEVELS = ['roberta-base', 'deepset/roberta-base-squad2',  'google/electra-small-discriminator']

    SOURCE_DATASET_LEVELS = ['squad_v2', 'subjqa']

    LR_SCHEDULE_LEVELS = ['CyclicLR']

    LOSS_EPS_LEVELS = [0.01, 0.001]  # loss eps for early stopping
    EARLY_STOPPING_EPOCH_COUNT_LEVELS = [5, 10, 20, 50]
    VALIDATION_SPLIT_LEVELS = [0.2]

    def __init__(self, output_filepath, dataset=None, model=None, poisoned_flag=None):
        self.master_seed = np.random.randint(2 ** 31 - 1)
        master_rso = np.random.RandomState(self.master_seed)

        self.output_filepath = str(output_filepath)

        self.lr_scheduler_level = int(master_rso.randint(len(RoundConfig.LR_SCHEDULE_LEVELS)))
        self.lr_scheduler = RoundConfig.LR_SCHEDULE_LEVELS[self.lr_scheduler_level]

        if poisoned_flag is None:
            self.poisoned_level = int(master_rso.randint(len(RoundConfig.POISONED_LEVELS)))
            self.poisoned = bool(RoundConfig.POISONED_LEVELS[self.poisoned_level])
        else:
            self.poisoned = poisoned_flag
            self.poisoned_level = RoundConfig.POISONED_LEVELS.index(self.poisoned)

        self.output_ground_truth_filename = 'ground_truth.csv'

        if model is None:
            self.model_architecture_level = int(master_rso.randint(len(RoundConfig.MODEL_LEVELS)))
            self.model_architecture = str(RoundConfig.MODEL_LEVELS[self.model_architecture_level])
        else:
            self.model_architecture = model
            self.model_architecture_level = RoundConfig.MODEL_LEVELS.index(self.model_architecture)

        self.learning_rate_level = int(master_rso.randint(len(RoundConfig.LEARNING_RATE_LEVELS)))
        self.learning_rate = float(RoundConfig.LEARNING_RATE_LEVELS[self.learning_rate_level])

        if self.lr_scheduler == 'CyclicLR':
            self.early_stopping = True
        elif self.lr_scheduler == 'WarmupWithLinear':
            self.early_stopping = False

        self.loss_eps_level = int(master_rso.randint(len(RoundConfig.LOSS_EPS_LEVELS)))
        self.loss_eps = RoundConfig.LOSS_EPS_LEVELS[self.loss_eps_level]

        self.early_stopping_epoch_count_level = int(master_rso.randint(len(RoundConfig.EARLY_STOPPING_EPOCH_COUNT_LEVELS)))
        self.early_stopping_epoch_count_level = 0  # TODO remove this if you want early stopping counts other than 5
        self.early_stopping_epoch_count = RoundConfig.EARLY_STOPPING_EPOCH_COUNT_LEVELS[self.early_stopping_epoch_count_level]

        self.batch_size_level = int(master_rso.randint(len(RoundConfig.BATCH_SIZE_LEVELS)))
        self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])

        self.validation_split_level = int(master_rso.randint(len(RoundConfig.VALIDATION_SPLIT_LEVELS)))
        self.validation_split = RoundConfig.VALIDATION_SPLIT_LEVELS[self.validation_split_level]

        if dataset is None:
            self.source_dataset_level = int(master_rso.randint(len(RoundConfig.SOURCE_DATASET_LEVELS)))
            self.source_dataset = str(RoundConfig.SOURCE_DATASET_LEVELS[self.source_dataset_level])
        else:
            self.source_dataset = dataset
            self.source_dataset_level = RoundConfig.SOURCE_DATASET_LEVELS.index(self.source_dataset)

        self.dropout_level = int(master_rso.randint(len(RoundConfig.DROPOUT_LEVELS)))
        self.dropout = float(RoundConfig.DROPOUT_LEVELS[self.dropout_level])

        self.trigger = None
        self.num_outputs = 2

    def setup_triggers(self, rso, executor=None, executor_option=None):
        if self.poisoned:
            # Trigger applying from source class -> targeting target class (cannot change during training)
            self.trigger = trigger_config.TriggerConfig(rso, executor=executor, executor_option=executor_option)

    def __eq__(self, other):
        if not isinstance(other, RoundConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        import pickle
        return pickle.dumps(self) == pickle.dumps(other)

    def save_json(self, filepath: str):
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(self, warn=True, indent=2))
        except:
            msg = 'Failed writing file "{}".'.format(filepath)
            logger.warning(msg)
            raise

    @staticmethod
    def load_json(filepath: str):
        if not os.path.exists(filepath):
            raise RuntimeError("Filepath does not exists: {}".format(filepath))
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                obj = jsonpickle.decode(f.read())
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except:
            msg = 'Failed reading file "{}".'.format(filepath)
            logger.warning(msg)
            raise

        return obj
