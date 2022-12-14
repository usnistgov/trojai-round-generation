# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import logging
import numpy as np
import json
import jsonpickle
import socket
import copy

import trigger_config

logger = logging.getLogger()

# TODO remove before flight
DEBUGGING_FLAG = False
FASTERRCNN_MAX_BATCH_SIZE = 8
DETR_MAX_BATCH_SIZE = 1

hostname = socket.gethostname()
# find gpus with >30GB
if 'pn125915' in hostname or 'pn125916' in hostname or 'pn125917' in hostname or 'pn125918' in hostname or 'pn120375' in hostname:
    FASTERRCNN_MAX_BATCH_SIZE = 4
else:
    FASTERRCNN_MAX_BATCH_SIZE = 2


class RoundConfig:
    CONFIG_FILENAME = 'config.json'

    MIN_BOX_DIMENSION = 8  # boxes can be no smaller than 8x8 pixels

    LEARNING_RATE_LEVELS = [1e-7, 3e-7, 5e-7, 8e-7, 1e-6]
    BATCH_SIZE_LEVELS = [8, 16, 32, 4, 1, 2]  # 1 is for DETR, which doesn't work accurately above a batch size of 1

    POISONED_LEVELS = [False, True]

    MODEL_LEVELS = ['fasterrcnn', 'ssd', 'detr']
    #MODEL_LEVELS = ['fasterrcnn', 'ssd']

    ADVERSERIAL_TRAINING_METHOD_LEVELS = [None, 'FBF']
    ADVERSERIAL_TRAINING_RATIO_LEVELS = [0.1, 0.3]
    ADVERSERIAL_EPS_LEVELS = [4.0 / 255.0, 8.0 / 255.0]

    if DEBUGGING_FLAG:
        #POISONED_LEVELS = [False]
        #MODEL_LEVELS = ['ssd']
        #ADVERSERIAL_TRAINING_METHOD_LEVELS = ["FBF"]
        #ADVERSERIAL_TRAINING_RATIO_LEVELS = [1.0]
    #     BATCH_SIZE_LEVELS = [8, 16, 32, 4, 2, 1]
    #     FASTERRCNN_MAX_BATCH_SIZE = 1
    #     DETR_MAX_BATCH_SIZE = 8
        pass


    LR_SCHEDULE_LEVELS = ['CyclicLR']
    SOURCE_DATASET_LEVELS = ['COCO']


    LOSS_EPS_LEVELS = [0.001, 0.01]  # loss eps for early stopping
    EARLY_STOPPING_EPOCH_COUNT_LEVELS = [5]
    EPOCH_COUNT_LEVELS = [2, 3, 4, 5]
    VALIDATION_SPLIT_LEVELS = [0.2]

    def __init__(self, output_filepath, model=None, poisoned_flag=None, adv_training=None):
        self.master_seed = np.random.randint(2 ** 31 - 1)
        self._master_rso = np.random.RandomState(self.master_seed)

        self.output_filepath = str(output_filepath)

        # record training configuration changes
        self.pre_injected_trigger = True
        # change to how RandomPhotometricDistort augmentation operates
        self.augmentation_photometric_distort_shuffled_channels = False

        self.lr_scheduler_level = int(self._master_rso.randint(len(RoundConfig.LR_SCHEDULE_LEVELS)))
        self.lr_scheduler = RoundConfig.LR_SCHEDULE_LEVELS[self.lr_scheduler_level]

        if poisoned_flag is None:
            self.poisoned_level = int(self._master_rso.randint(len(RoundConfig.POISONED_LEVELS)))
            self.poisoned = bool(RoundConfig.POISONED_LEVELS[self.poisoned_level])
        else:
            self.poisoned = poisoned_flag
            self.poisoned_level = RoundConfig.POISONED_LEVELS.index(self.poisoned)

        if adv_training is None:
            self.adversarial_training_method_level = int(self._master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS)))
            self.adversarial_training_method = RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS[self.adversarial_training_method_level]
        else:
            self.adversarial_training_method = adv_training
            self.adversarial_training_method_level = RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS.index(self.adversarial_training_method)

        if self.adversarial_training_method is not None:
            self.adversarial_eps_level = int(self._master_rso.randint(len(RoundConfig.ADVERSERIAL_EPS_LEVELS)))
            self.adversarial_eps = float(RoundConfig.ADVERSERIAL_EPS_LEVELS[self.adversarial_eps_level])
            self.adversarial_training_ratio_level = int(self._master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_RATIO_LEVELS)))
            self.adversarial_training_ratio = float(RoundConfig.ADVERSERIAL_TRAINING_RATIO_LEVELS[self.adversarial_training_ratio_level])
        else:
            self.adversarial_eps_level = None
            self.adversarial_eps = None
            self.adversarial_training_ratio_level = None
            self.adversarial_training_ratio = None

        self.output_ground_truth_filename = 'ground_truth.csv'

        if model is None:
            self.model_architecture_level = int(self._master_rso.randint(len(RoundConfig.MODEL_LEVELS)))
            self.model_architecture = str(RoundConfig.MODEL_LEVELS[self.model_architecture_level])
        else:
            self.model_architecture = model
            self.model_architecture_level = RoundConfig.MODEL_LEVELS.index(self.model_architecture)

        self.batch_size_level = int(self._master_rso.randint(len(RoundConfig.BATCH_SIZE_LEVELS)))
        self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])

        if self.model_architecture == "ssd":
            acceptable_indices = np.asarray(RoundConfig.BATCH_SIZE_LEVELS) > 1
            acceptable_indices = np.flatnonzero(acceptable_indices)
            selected_idx = self._master_rso.randint(len(acceptable_indices))
            self.batch_size_level = int(acceptable_indices[selected_idx])
            self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])
        if self.model_architecture == "fasterrcnn":
            acceptable_indices = np.logical_and(np.asarray(RoundConfig.BATCH_SIZE_LEVELS) <= FASTERRCNN_MAX_BATCH_SIZE, np.asarray(RoundConfig.BATCH_SIZE_LEVELS) > 1)
            acceptable_indices = np.flatnonzero(acceptable_indices)
            selected_idx = self._master_rso.randint(len(acceptable_indices))
            self.batch_size_level = int(acceptable_indices[selected_idx])
            self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])
        if self.model_architecture == "detr":
            acceptable_indices = np.asarray(RoundConfig.BATCH_SIZE_LEVELS) <= DETR_MAX_BATCH_SIZE
            acceptable_indices = np.flatnonzero(acceptable_indices)
            selected_idx = self._master_rso.randint(len(acceptable_indices))
            self.batch_size_level = int(acceptable_indices[selected_idx])
            self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])


        acceptable_indices = np.asarray(RoundConfig.LEARNING_RATE_LEVELS) >= 5e-7
        acceptable_indices = np.flatnonzero(acceptable_indices)
        selected_idx = self._master_rso.randint(len(acceptable_indices))
        self.learning_rate_level = int(acceptable_indices[selected_idx])
        self.learning_rate = RoundConfig.LEARNING_RATE_LEVELS[self.learning_rate_level]
        # self.learning_rate_level = int(self._master_rso.randint(len(RoundConfig.LEARNING_RATE_LEVELS)))
        # self.learning_rate = float(RoundConfig.LEARNING_RATE_LEVELS[self.learning_rate_level])

        #self.loss_eps_level = int(self._master_rso.randint(len(RoundConfig.LOSS_EPS_LEVELS)))
        # enforce eps of 0.01
        self.loss_eps_level = int(1)
        self.loss_eps = RoundConfig.LOSS_EPS_LEVELS[self.loss_eps_level]

        self.early_stopping_epoch_count_level = int(self._master_rso.randint(len(RoundConfig.EARLY_STOPPING_EPOCH_COUNT_LEVELS)))
        self.early_stopping_epoch_count = RoundConfig.EARLY_STOPPING_EPOCH_COUNT_LEVELS[self.early_stopping_epoch_count_level]

        self.validation_split_level = int(self._master_rso.randint(len(RoundConfig.VALIDATION_SPLIT_LEVELS)))
        self.validation_split = RoundConfig.VALIDATION_SPLIT_LEVELS[self.validation_split_level]


        self.source_dataset_level = int(self._master_rso.randint(len(RoundConfig.SOURCE_DATASET_LEVELS)))
        self.source_dataset = str(RoundConfig.SOURCE_DATASET_LEVELS[self.source_dataset_level])

        self.trigger = None
        self.actual_trojan_percentage = {}
        self.number_trojan_instances = {}

        self.actual_spurious_percentage = {}
        self.number_spurious_instances = {}


    def update_trojan_percentage(self, name, trojan_percentage, number_trojan_instances):
        self.actual_trojan_percentage[name] = trojan_percentage
        self.number_trojan_instances[name] = number_trojan_instances

    def update_spuriour_percentage(self, name, spurious_percentage, number_spurious_instances):
        self.actual_spurious_percentage[name] = spurious_percentage
        self.number_spurious_instances[name] = number_spurious_instances

    def setup_triggers(self, trigger_labels_dict, executor=None, executor_location=None, executor_type=None, executor_option=None, trigger_size=None):
        if self.poisoned:
            self.trigger = trigger_config.TriggerConfig(self.output_filepath, self.master_rso, trigger_labels_dict, executor=executor, executor_location=executor_location, executor_type=executor_type, executor_option=executor_option, trigger_size=trigger_size)


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

    @property
    def master_rso(self):
        return self._master_rso

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        state_list = list(state.keys())
        # Delete any fields we want to avoid when using jsonpickle, currently anything starting with '_' will be deleted
        for key in state_list:
            if key.startswith('_'):
                del state[key]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)