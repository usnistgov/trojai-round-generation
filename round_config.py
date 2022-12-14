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
import shutil

import trigger_executor
import object_detection_trigger_executor
import utils
import trojai.datagen.polygon_trigger

logger = logging.getLogger()

# TODO remove before flight
DEBUGGING_FLAG = False


# Task Type Constants
OBJ = 'object_detection'
CLASS = 'classification'


# Max Batch size controls to ensure things fit in GPU memory
hostname = socket.gethostname()
# find gpus with >30GB
if 'pn125915' in hostname or 'pn125916' in hostname or 'pn125917' in hostname or 'pn125918' in hostname or 'pn120375' in hostname:
     FASTERRCNN_MAX_BATCH_SIZE = 8
     DETR_MAX_BATCH_SIZE = 32
else:
    FASTERRCNN_MAX_BATCH_SIZE = 2
    DETR_MAX_BATCH_SIZE = 16


class RoundConfig:
    CONFIG_FILENAME = 'config.json'

    # TASK_TYPE = [CLASS, OBJ]
    TASK_TYPE = [OBJ]




    # ***************************
    # Training Parameter Control
    # ***************************
    LEARNING_RATE_LEVELS = ['{}:1e-4'.format(CLASS), '{}:3e-4'.format(CLASS), '{}:1e-5'.format(CLASS), '{}:1e-4'.format(OBJ), '{}:1e-5'.format(OBJ), '{}:1e-6'.format(OBJ)]
    # for obj det
    BATCH_SIZE_LEVELS = [2, 4, 8, 16, 32]

    ADVERSERIAL_TRAINING_METHOD_LEVELS = ['none', 'fbf']
    ADVERSERIAL_TRAINING_RATIO_LEVELS = [0.1, 0.3]
    ADVERSERIAL_EPS_LEVELS = [4.0 / 255.0, 8.0 / 255.0]

    MODEL_LEVELS = ['{}:fasterrcnn_resnet50_fpn_v2'.format(OBJ), '{}:ssd300_vgg16'.format(OBJ), '{}:detr'.format(OBJ),
                    '{}:resnet50'.format(CLASS), '{}:wide_resnet50_2'.format(CLASS), '{}:mobilenet_v2'.format(CLASS), '{}:vit_base_patch32_224'.format(CLASS)]
    MODEL_WEIGHT_LEVELS = ['DEFAULT']
    WEIGHT_DECAY_LEVELS = [1e-5, 1e-4, "none"]
    PLATEAU_LEARNING_RATE_PATIENCE_LEVELS = [10, 20]
    PLATEAU_LEARNING_RATE_THRESHOLD_LEVELS = [1e-4]
    PLATEAU_LEARNING_RATE_REDUCTION_FACTOR_LEVELS = [0.2]
    NUM_LEARNING_RATE_REDUCTION_LEVELS = [0]  # with decay count=0, scheduler just performs early stopping
    VALIDATION_SPLIT_LEVELS = [0.1]

    # ***************************
    # Dataset Generation Control
    # ***************************
    MIN_BOX_DIMENSION = 10  # boxes can be no smaller than 10x10 pixels
    SOURCE_DATASET_LEVELS = ['cityscapes']

    MAX_CLASS_COUNT_PER_IMAGE = ['{}:1'.format(CLASS), '{}:2'.format(OBJ), '{}:4'.format(OBJ), '{}:8'.format(OBJ)]
    MAX_CLASS_INSTANCE_COUNT_PER_IMAGE = ['{}:1'.format(CLASS), '{}:2'.format(OBJ), '{}:4'.format(OBJ)]

    MAX_TOTAL_INSTANCES_PER_IMAGE = [4, 8, 16, 32]

    # This controls the size of the train dataset, val and test are a defined fraction of this number
    CLASS_COUNT_LEVELS = [16, 32, 64, 128]
    CLASS_COUNT_BUFFER = 10

    # This controls the size of the train dataset, val and test are a defined fraction of this number
    NUMBER_IMAGES_PER_CLASS = [2000]
    # train dataset size = CLASS_COUNT_LEVELS * NUMBER_IMAGES_PER_CLASS
    # val dataset size = VALIDATION_SPLIT_LEVELS * (train dataset size)

    MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS = 100

    RAIN_PROBABILITY_LEVELS = [0.02, 0.04, 0.08]
    FOG_PROBABILITY_LEVELS = [0.02, 0.04, 0.08]

    BASE_IMAGE_SIZE = 256
    # % area of the whole image
    FOREGROUND_SIZE_RANGE = ['{}:0.04, 0.5'.format(CLASS), '{}:0.01, 0.2'.format(OBJ)]

    # ***************************
    # Poisoning Control
    # ***************************
    POISONED_LEVELS = [False, True]
    NUM_TRIGGERS_LEVELS = [1, 2, 4]
    TRIGGER_EXECUTOR_LEVELS = ['{}:instagram'.format(CLASS), '{}:polygon'.format(CLASS), '{}:polygon-misclassification'.format(OBJ), '{}:polygon-evasion'.format(OBJ), '{}:polygon-injection'.format(OBJ), '{}:polygon-localization'.format(OBJ)]
    TRIGGER_PRE_INJECTION_LEVELS = [True]

    # Maximum percentage for a class to be triggered
    MAX_PER_CLASS_TOTAL_TRIGGER_FRACTION = 0.5


    if DEBUGGING_FLAG:
        TASK_TYPE = [OBJ]
        MODEL_LEVELS = ['{}:ssd300_vgg16'.format(OBJ)]
        MAX_CLASS_COUNT_PER_IMAGE = ['{}:4'.format(OBJ)]
        MAX_CLASS_INSTANCE_COUNT_PER_IMAGE = ['{}:2'.format(OBJ)]
        TRIGGER_EXECUTOR_LEVELS = ['{}:polygon-localization'.format(OBJ)]
        ADVERSERIAL_TRAINING_METHOD_LEVELS = ['none']

        POISONED_LEVELS = [True]
        NUM_TRIGGERS_LEVELS = [1]
        NUMBER_IMAGES_PER_CLASS = [100]
        MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS = 2
        CLASS_COUNT_LEVELS = [2]
        CLASS_COUNT_BUFFER = 0



    def __init__(self, output_filepath, model=None, poisoned_flag=None, number_classes_level=None, num_triggers=None):
        self.is_saving_check = False

        self.master_seed = np.random.randint(2 ** 31 - 1)
        self._master_rso = np.random.RandomState(self.master_seed)

        self.num_workers = utils.get_num_workers()

        self.output_filepath = str(output_filepath)

        self.task_type_level = int(self._master_rso.randint(len(RoundConfig.TASK_TYPE)))
        self.task_type = RoundConfig.TASK_TYPE[self.task_type_level]

        self.source_dataset_level = int(self._master_rso.randint(len(RoundConfig.SOURCE_DATASET_LEVELS)))
        self.source_dataset = RoundConfig.SOURCE_DATASET_LEVELS[self.source_dataset_level]

        if model is None:
            self.model_architecture = self.get_random_item_for_task(RoundConfig.MODEL_LEVELS)
        else:
            self.model_architecture = model
        self.model_architecture_level = RoundConfig.MODEL_LEVELS.index(self.model_architecture)

        self.max_class_count_per_image = self.get_random_item_for_task(RoundConfig.MAX_CLASS_COUNT_PER_IMAGE)
        self.max_class_count_per_image_level = RoundConfig.MAX_CLASS_COUNT_PER_IMAGE.index(self.max_class_count_per_image)

        self.max_class_instance_count_per_image = self.get_random_item_for_task(RoundConfig.MAX_CLASS_INSTANCE_COUNT_PER_IMAGE)
        self.max_class_instance_count_per_image_level = RoundConfig.MAX_CLASS_INSTANCE_COUNT_PER_IMAGE.index(self.max_class_instance_count_per_image)

        self.max_total_class_count_per_image_level = int(self._master_rso.randint(len(RoundConfig.MAX_TOTAL_INSTANCES_PER_IMAGE)))
        self.max_total_class_count_per_image = RoundConfig.MAX_TOTAL_INSTANCES_PER_IMAGE[self.max_total_class_count_per_image_level]

        if poisoned_flag is None:
            self.poisoned_level = int(self._master_rso.randint(len(RoundConfig.POISONED_LEVELS)))
            self.poisoned = bool(RoundConfig.POISONED_LEVELS[self.poisoned_level])
        else:
            self.poisoned = poisoned_flag
            self.poisoned_level = RoundConfig.POISONED_LEVELS.index(self.poisoned)

        if self.poisoned:
            self.trigger_pre_injection_level = int(self._master_rso.randint(len(RoundConfig.TRIGGER_PRE_INJECTION_LEVELS)))
            self.trigger_pre_injection = bool(RoundConfig.TRIGGER_PRE_INJECTION_LEVELS[self.trigger_pre_injection_level])
        else:
            self.trigger_pre_injection_level = None
            self.trigger_pre_injection = False

        self.adversarial_training_method_level = int(self._master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS)))
        self.adversarial_training_method = RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS[self.adversarial_training_method_level]

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

        # adversarial training is not supported for object detection
        if self.task_type == OBJ:
            self.adversarial_training_method = 'none'
            self.adversarial_training_method_level = RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS.index(self.adversarial_training_method)

            self.adversarial_eps_level = None
            self.adversarial_eps = None
            self.adversarial_training_ratio_level = None
            self.adversarial_training_ratio = None

        self.output_ground_truth_filename = 'ground_truth.csv'

        self.learning_rate = self.get_random_item_for_task(RoundConfig.LEARNING_RATE_LEVELS)
        self.learning_rate_level = RoundConfig.LEARNING_RATE_LEVELS.index(self.learning_rate)
        self.learning_rate = float(self.learning_rate.split(":")[1])

        self.plateau_learning_rate_patience_level = int(self._master_rso.randint(len(RoundConfig.PLATEAU_LEARNING_RATE_PATIENCE_LEVELS)))
        self.plateau_learning_rate_patience = RoundConfig.PLATEAU_LEARNING_RATE_PATIENCE_LEVELS[self.plateau_learning_rate_patience_level]

        self.plateau_learning_rate_threshold_level = int(self._master_rso.randint(len(RoundConfig.PLATEAU_LEARNING_RATE_THRESHOLD_LEVELS)))
        self.plateau_learning_rate_threshold = RoundConfig.PLATEAU_LEARNING_RATE_THRESHOLD_LEVELS[self.plateau_learning_rate_threshold_level]

        self.plateau_learning_rate_reduction_factor_level = int(self._master_rso.randint(len(RoundConfig.PLATEAU_LEARNING_RATE_REDUCTION_FACTOR_LEVELS)))
        self.plateau_learning_rate_reduction_factor = RoundConfig.PLATEAU_LEARNING_RATE_REDUCTION_FACTOR_LEVELS[self.plateau_learning_rate_reduction_factor_level]

        self.num_plateau_learning_rate_reductions_level = int(self._master_rso.randint(len(RoundConfig.NUM_LEARNING_RATE_REDUCTION_LEVELS)))
        self.num_plateau_learning_rate_reductions = RoundConfig.NUM_LEARNING_RATE_REDUCTION_LEVELS[self.num_plateau_learning_rate_reductions_level]

        self.weight_decay_level = int(self._master_rso.randint(len(RoundConfig.WEIGHT_DECAY_LEVELS)))
        self.weight_decay = RoundConfig.WEIGHT_DECAY_LEVELS[self.weight_decay_level]

        self.validation_split_level = int(self._master_rso.randint(len(RoundConfig.VALIDATION_SPLIT_LEVELS)))
        self.validation_split = RoundConfig.VALIDATION_SPLIT_LEVELS[self.validation_split_level]

        self.img_size_pixels = RoundConfig.BASE_IMAGE_SIZE
        self.img_shape = [RoundConfig.BASE_IMAGE_SIZE, RoundConfig.BASE_IMAGE_SIZE, 3]
        self.img_type = 'uint8'
        self.gaussian_blur_ksize_min = 3
        self.gaussian_blur_ksize_max = 5

        self.rain_probability_level = int(self._master_rso.randint(len(RoundConfig.RAIN_PROBABILITY_LEVELS)))
        self.rain_probability = float(RoundConfig.RAIN_PROBABILITY_LEVELS[self.rain_probability_level])

        self.fog_probability_level = int(self._master_rso.randint(len(RoundConfig.FOG_PROBABILITY_LEVELS)))
        self.fog_probability = float(RoundConfig.FOG_PROBABILITY_LEVELS[self.fog_probability_level])

        if number_classes_level is not None:
            self.number_classes_level = int(number_classes_level)
        else:
            self.number_classes_level = int(self._master_rso.randint(len(RoundConfig.CLASS_COUNT_LEVELS)))
        self.number_classes = int(RoundConfig.CLASS_COUNT_LEVELS[self.number_classes_level])
        self.number_classes = max(2, self.number_classes + self._master_rso.randint(-RoundConfig.CLASS_COUNT_BUFFER, RoundConfig.CLASS_COUNT_BUFFER + 1))

        self.number_image_per_class_level = int(self._master_rso.randint(len(RoundConfig.NUMBER_IMAGES_PER_CLASS)))
        self.number_image_per_class = int(RoundConfig.NUMBER_IMAGES_PER_CLASS[self.number_image_per_class_level])
        self.total_dataset_size = self.number_classes * self.number_image_per_class

        # self.batch_size_level = int(self._master_rso.randint(len(RoundConfig.BATCH_SIZE_LEVELS)))
        # self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])
        if self.task_type == OBJ:
            if self.model_architecture == "{}:fasterrcnn".format(OBJ):
                acceptable_indices = np.asarray(RoundConfig.BATCH_SIZE_LEVELS) <= FASTERRCNN_MAX_BATCH_SIZE
                acceptable_indices = np.flatnonzero(acceptable_indices)
                selected_idx = self._master_rso.randint(len(acceptable_indices))
                self.batch_size_level = int(acceptable_indices[selected_idx])
                self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])
            elif self.model_architecture == "{}:detr".format(OBJ):
                acceptable_indices = np.asarray(RoundConfig.BATCH_SIZE_LEVELS) <= DETR_MAX_BATCH_SIZE
                acceptable_indices = np.flatnonzero(acceptable_indices)
                selected_idx = self._master_rso.randint(len(acceptable_indices))
                self.batch_size_level = int(acceptable_indices[selected_idx])
                self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])
            else:
                self.batch_size_level = int(self._master_rso.randint(len(RoundConfig.BATCH_SIZE_LEVELS)))
                self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])

            # these don't apply to obj models
            self.model_weight_level = None
            self.model_weight = None
        elif self.task_type == CLASS:
            acceptable_indices = np.asarray(RoundConfig.BATCH_SIZE_LEVELS) >= 16
            acceptable_indices = np.flatnonzero(acceptable_indices)
            selected_idx = self._master_rso.randint(len(acceptable_indices))
            self.batch_size_level = int(acceptable_indices[selected_idx])
            self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])

            self.model_weight_level = int(self._master_rso.randint(len(RoundConfig.MODEL_WEIGHT_LEVELS)))
            self.model_weight = RoundConfig.MODEL_WEIGHT_LEVELS[self.model_weight_level]
        else:
            raise RuntimeError("Invalid task_type option: {}".format(self.task_type))

        foreground_size_range = self.get_random_item_for_task(RoundConfig.FOREGROUND_SIZE_RANGE).split(':')[1]
        foreground_size_range = [float(v) for v in foreground_size_range.split(',')]
        self.foreground_size_percentage_of_image_min = float(np.min(foreground_size_range))
        self.foreground_size_percentage_of_image_max = float(np.max(foreground_size_range))

        img_area = self.img_size_pixels * self.img_size_pixels
        foreground_area_min = img_area * self.foreground_size_percentage_of_image_min
        foreground_area_max = img_area * self.foreground_size_percentage_of_image_max
        self.foreground_size_pixels_min = int(np.sqrt(foreground_area_min))
        self.foreground_size_pixels_max = int(np.sqrt(foreground_area_max))

        self.triggers = list()

        if num_triggers is not None:
            self.num_triggers = int(num_triggers)
            self.num_triggers_level = RoundConfig.NUM_TRIGGERS_LEVELS.index(self.num_triggers)
        else:
            self.num_triggers_level = self._master_rso.randint(len(RoundConfig.NUM_TRIGGERS_LEVELS))
            self.num_triggers = int(RoundConfig.NUM_TRIGGERS_LEVELS[self.num_triggers_level])

        self.requested_trojan_percentages = {}
        self.total_class_instances = {}

    def get_random_item_for_task(self, item_list):
        temp_list = [i for i in item_list if i.startswith(self.task_type)]
        index = int(self._master_rso.randint(len(temp_list)))
        return temp_list[index]


    def setup_triggers(self, label_list, requested_trigger_exec=None):
        if self.poisoned:

            valid_trigger_executors = [i for i in RoundConfig.TRIGGER_EXECUTOR_LEVELS if i.startswith(self.task_type)]
            for i in range(self.num_triggers):
                if i == 0 and requested_trigger_exec is not None:
                    # only override the first trigger if there are multiple
                    if requested_trigger_exec == 'InstagramTriggerExecutor':
                        selected_trigger_executor = '{}:instagram'.format(CLASS)
                    elif requested_trigger_exec == 'ClassificationPolygonTriggerExecutor':
                        selected_trigger_executor = '{}:polygon'.format(CLASS)
                    elif requested_trigger_exec == 'ObjectDetectionMisclassificationPolygonTriggerExecutor':
                        selected_trigger_executor = '{}:polygon-misclassification'.format(OBJ)
                    elif requested_trigger_exec == 'ObjectDetectionEvasionPolygonTriggerExecutor':
                        selected_trigger_executor = '{}:polygon-evasion'.format(OBJ)
                    elif requested_trigger_exec == 'ObjectDetectionInjectionPolygonTriggerExecutor':
                        selected_trigger_executor = '{}:polygon-injection'.format(OBJ)
                    elif requested_trigger_exec == 'ObjectDetectionLocalizationPolygonTriggerExecutor':
                        selected_trigger_executor = '{}:polygon-localization'.format(OBJ)
                    else:
                        raise RuntimeError('Unknown requested trigger executor: {}'.format(requested_trigger_exec))
                else:

                    # Randomly select which trigger to use
                    index = self._master_rso.randint(len(valid_trigger_executors))
                    selected_trigger_executor = valid_trigger_executors[index]

                rso = np.random.RandomState(self._master_rso.randint(0, 2**32-1))
                # Image Classification
                if selected_trigger_executor == '{}:instagram'.format(CLASS):
                    trigger_inst = trigger_executor.InstagramTriggerExecutor(i, label_list, rso)
                elif selected_trigger_executor == '{}:polygon'.format(CLASS):
                    trigger_inst = trigger_executor.ClassificationPolygonTriggerExecutor(i, label_list, rso, self.output_filepath)
                # Object Detection
                elif selected_trigger_executor == '{}:polygon-misclassification'.format(OBJ):
                    trigger_inst = object_detection_trigger_executor.ObjectDetectionMisclassificationPolygonTriggerExecutor(i, label_list, rso, self.output_filepath)
                elif selected_trigger_executor == '{}:polygon-evasion'.format(OBJ):
                    trigger_inst = object_detection_trigger_executor.ObjectDetectionEvasionPolygonTriggerExecutor(i, label_list, rso, self.output_filepath)
                elif selected_trigger_executor == '{}:polygon-injection'.format(OBJ):
                    trigger_inst = object_detection_trigger_executor.ObjectDetectionInjectionPolygonTriggerExecutor(i, label_list, rso, self.output_filepath)
                elif selected_trigger_executor == '{}:polygon-localization'.format(OBJ):
                    trigger_inst = object_detection_trigger_executor.ObjectDetectionLocalizationPolygonTriggerExecutor(i, label_list, rso, self.output_filepath)
                else:
                    raise RuntimeError('Unknown trigger executor: {}'.format(selected_trigger_executor))

                self.triggers.append(trigger_inst)

                if trigger_inst.source_class in self.requested_trojan_percentages:
                    if trigger_inst.trigger_fraction is not None:
                        self.requested_trojan_percentages[trigger_inst.source_class] += trigger_inst.trigger_fraction
                else:
                    if trigger_inst.trigger_fraction is not None:
                        self.requested_trojan_percentages[trigger_inst.source_class] = trigger_inst.trigger_fraction

            if not DEBUGGING_FLAG:
                # Verify requested_trojan_percentages
                for source_class_id, trigger_fraction in self.requested_trojan_percentages.items():
                    if trigger_fraction is not None and trigger_fraction > RoundConfig.MAX_PER_CLASS_TOTAL_TRIGGER_FRACTION:
                        raise RuntimeError('Trigger fraction amount for Class {} is too high: {}, must be below: {}'.format(source_class_id, trigger_fraction, RoundConfig.MAX_PER_CLASS_TOTAL_TRIGGER_FRACTION))


    def _select_random_classes(self, dataset_dirpath, image_format='png'):
        if image_format.startswith('.'):
            image_format = image_format[1:]

        foreground_dataset_dirpath = os.path.join(dataset_dirpath, 'foregrounds')

        if not os.path.exists(foreground_dataset_dirpath):
            raise RuntimeError('Unabled to find foreground dataset dirpath: {}'.format(foreground_dataset_dirpath))

        fns = [fn for fn in os.listdir(foreground_dataset_dirpath) if fn.endswith(image_format)]

        filenames = self._master_rso.choice(fns, size=self.number_classes, replace=False).tolist()

        # Dead code for ensuring exclusivity among the first digits to avoid the center symbol from being reused multiple times in a dataset
        # fns = [fn for fn in os.listdir(foreground_dataset_dirpath) if fn.endswith('-0.{}'.format(image_format))]
        # first_numbers = [int(fn.split('-')[0]) for fn in fns]
        #
        # filenames = list()
        # used_second_numbers = list()
        # first_numbers = self._master_rso.choice(first_numbers, size=self.number_classes, replace=False)
        # for first_number in first_numbers:
        #     fns = [fn for fn in os.listdir(foreground_dataset_dirpath) if fn.startswith('{}-'.format(first_number))]
        #     fn = None
        #     if len(fns) > 1:
        #         second_numbers = list()
        #         for fn in fns:
        #             second_numbers.append(int(fn.replace('.{}'.format(image_format), '').split('-')[1]))
        #         self._master_rso.shuffle(second_numbers)
        #         for i in range(len(second_numbers)):
        #             second_number = second_numbers[i]
        #             if second_number not in used_second_numbers:
        #                 used_second_numbers.append(second_number)
        #                 fn = '{}-{}.{}'.format(first_number, second_number, image_format)
        #                 break
        #     else:
        #         fn = fns[0]
        #     if fn is None:
        #         raise RuntimeError('Unable to select independent foregrounds.')
        #     filenames.append(fn)

        for i, filename in enumerate(filenames):
            filename = os.path.join(foreground_dataset_dirpath, filename)
            filenames[i] = filename
        return filenames

    def _copy_foregrounds(self, filename_list):
        foreground_output_filepath = os.path.join(self.output_filepath, 'foregrounds')

        if os.path.exists(foreground_output_filepath):
            shutil.rmtree(foreground_output_filepath)
        os.makedirs(foreground_output_filepath)

        for fn in filename_list:
            basename = os.path.basename(fn)
            shutil.copy(fn, os.path.join(foreground_output_filepath, basename))

    def setup_class_images(self, dataset_dirpath):
        filenames = self._select_random_classes(dataset_dirpath)
        self._copy_foregrounds(filenames)

    def __eq__(self, other):
        if not isinstance(other, RoundConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        import pickle
        return pickle.dumps(self) == pickle.dumps(other)

    def save_json(self, filepath: str):
        self.enable_saving()
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(self, warn=True, indent=2))
        except:
            msg = 'Failed writing file "{}".'.format(filepath)
            logger.warning(msg)
            raise
        self.disable_saving()

    @staticmethod
    def load_json(filepath: str):
        if not os.path.exists(filepath):
            raise RuntimeError("Filepath does not exists: {}".format(filepath))
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                obj = jsonpickle.decode(f.read())
                obj.disable_saving()

            # set the current folder
            basepath, _ = os.path.split(filepath)
            obj.output_filepath = basepath

            # Initialize the random state object when reloading a configuration file.
            obj._master_rso = np.random.RandomState(obj.master_seed)
            # look for other _rso objects which need to be initialized
            if hasattr(obj, 'triggers'):
                for t_idx in range(len(obj.triggers)):
                    trigger = obj.triggers[t_idx]
                    if isinstance(trigger, trigger_executor.PolygonTriggerExecutor):
                        trigger._trigger_polygon = trojai.datagen.polygon_trigger.PolygonTrigger(img_size=None, n_sides=None, random_state_obj=obj._master_rso, color=trigger.trigger_color, texture_augmentation=trigger.polygon_texture_augmentation, filepath=os.path.join(obj.output_filepath, 'trigger_{}.png'.format(t_idx)))

        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except:
            msg = 'Failed reading file "{}".'.format(filepath)
            logger.warning(msg)
            raise

        return obj

    # @property
    # def master_rso(self):
    #     return self._master_rso

    def enable_saving(self):
        self.is_saving_check = True
        for trigger in self.triggers:
            trigger.enable_saving()

    def disable_saving(self):
        self.is_saving_check = False
        for trigger in self.triggers:
            trigger.disable_saving()

    def __getstate__(self):

        state = copy.deepcopy(self.__dict__)
        if not self.is_saving_check:
            return state

        state_list = list(state.keys())
        # Delete any fields we want to avoid when using jsonpickle, currently anything starting with '_' will be deleted
        for key in state_list:
            if key.startswith('_'):
                del state[key]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
