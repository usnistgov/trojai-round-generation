import os
from abc import ABC, abstractmethod
import logging
import numpy as np
import json
import jsonpickle
import copy
import shutil

# local imports
import trigger_executor
import utils
import trojai.datagen.polygon_trigger
import dex



# TODO remove before flight
DEBUGGING_FLAG = False


CLASS = 'classification'
OBJ = 'object_detection'

# abstract base class to define a round config. You cannot instantiate this class, you need to create one of the task specific subclasses
class Config(ABC):
    # Note: this class cannot be directly instantiated. Use a subclass.

    CONFIG_FILENAME = 'config.json'
    MODEL_FILENAME = 'model.pt'
    GROUND_TRUTH_FILENAME = 'ground_truth.csv'
    TRAIN_COMPLETE_FILENAME = 'train-complete.log'

    # lists the type of the task as a string
    TASK_TYPE = None  # this needs to be specified in the appropriate subclass

    # ***************************
    # Training Parameter Control
    # ***************************
    LEARNING_RATE_LEVELS = [1e-4, 3e-4, 1e-5, 1e-6]
    BATCH_SIZE_LEVELS = [4, 8, 16]

    ADVERSARIAL_TRAINING_METHOD_LEVELS = [None, 'fbf']
    ADVERSARIAL_TRAINING_RATIO_LEVELS = [0.1, 0.3]
    ADVERSARIAL_EPS_LEVELS = [4.0 / 255.0, 8.0 / 255.0]

    MODEL_ARCHITECTURE_LEVELS = None  # this needs to be specified in the appropriate subclass

    # None will load a randomly initialized model
    # 'DEFAULT' will load the default pytorch.hub.load_model weight
    MODEL_WEIGHT_LEVELS = ['DEFAULT']

    WEIGHT_DECAY_LEVELS = [None, 1e-4, 1e-5]
    # if None, no cyclic learning rate will be used, Otherwise the factor level will be used to define a cycle from base_learning_rate/factor to base_learning_rate*factor
    # base_lr=(base_learning_rate / self.config.cyclic_learning_rate_factor)
    # max_lr=(base_learning_rate * self.config.cyclic_learning_rate_factor)
    CYCLIC_LEARNING_RATE_FACTOR_LEVELS = [None, 2.0, 4.0]
    PLATEAU_LEARNING_RATE_PATIENCE_LEVELS = [5, 10]
    # the plateau scheduler will consider all values within this eps of global optimum equivalent
    PLATEAU_LEARNING_RATE_THRESHOLD_LEVELS = [0.01]
    # the plateau scheduler will reduce the learning rate by this factor when improvement halts. (i.e. learning_rate *= factor)
    PLATEAU_LEARNING_RATE_REDUCTION_FACTOR_LEVELS = [0.2]
    # the number of learning rate reductions the plateau scheduler will perform (with recutions count=0, scheduler just performs early stopping)
    NUM_PLATEAU_LEARNING_RATE_REDUCTIONS_LEVELS = [0]
    VALIDATION_SPLIT_LEVELS = [0.1]  # what fraction of the training dataset to use for validation

    # ***************************
    # Poisoning Control
    # ***************************
    POISONED_LEVELS = [False, True]
    NUM_TRIGGERS_LEVELS = [1]
    TRIGGER_EXECUTOR_LEVELS = [None]
    TRIGGER_PRE_INJECTION_LEVELS = [True]

    # Maximum percentage for a class to be triggered
    MAX_PER_CLASS_TOTAL_TRIGGER_FRACTION = 0.5


    def __init__(self, kwargs: dict):
        self.is_saving_check = False  # internal control variable for saving and loading this config json object
        # an amp flag controlling so that certain configurations can disable AMP during training easily
        self.use_amp = kwargs.pop('use_amp', True)
        self.log_interval = 100  # log every x batches

        self.master_seed = np.random.randint(2 ** 31 - 1)
        self._master_rso = np.random.RandomState(self.master_seed)

        self.num_workers = utils.get_num_workers()
        # allow for special control of the number of datagen workers for the enki cluster
        self.num_workers_datagen = kwargs.pop('num_workers_datagen', self.num_workers)

        if 'output_filepath' not in kwargs.keys():
            raise RuntimeError("Missing required entry 'output_filepath' in RoundConfig kwargs")
        # delete the key, so we know which args are unused at the end of init
        self.output_filepath = str(kwargs.pop('output_filepath'))

        # to be filled in later if applicable. This holds the translation dictionary from class ids to foreground image names
        self.fg_class_translation = None  # to be filled in later
        self.number_images_per_class = None  # to be filled in later
        self.triggers = list()  # to be filled in later

        req = kwargs.pop('model_architecture', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.model_architecture = dex.Factor(levels=self.MODEL_ARCHITECTURE_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('poisoned', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.poisoned = dex.Factor(levels=self.POISONED_LEVELS, rso=self._master_rso, requested_level=req)

        if self.poisoned.value:
            req = kwargs.pop('trigger_pre_injection', None)  # delete the key, so we know which args are unused at the end of init
            # if req is None, it gets ignored and the rso is used
            self.trigger_pre_injection = dex.Factor(levels=self.TRIGGER_PRE_INJECTION_LEVELS, rso=self._master_rso, requested_level=req)
        else:
            levels = self.TRIGGER_PRE_INJECTION_LEVELS
            levels.append(False)
            self.trigger_pre_injection = dex.Factor(levels=levels, requested_level=False)

        req = kwargs.pop('adversarial_training_method', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.adversarial_training_method = dex.Factor(levels=self.ADVERSARIAL_TRAINING_METHOD_LEVELS, rso=self._master_rso, requested_level=req)

        if self.adversarial_training_method is not None and self.adversarial_training_method.value is not None:
            req = kwargs.pop('adversarial_eps', None)  # delete the key, so we know which args are unused at the end of init
            # if req is None, it gets ignored and the rso is used
            self.adversarial_eps = dex.Factor(levels=self.ADVERSARIAL_EPS_LEVELS, rso=self._master_rso, requested_level=req)

            req = kwargs.pop('adversarial_training_ratio', None)  # delete the key, so we know which args are unused at the end of init
            # if req is None, it gets ignored and the rso is used
            self.adversarial_training_ratio = dex.Factor(levels=self.ADVERSARIAL_TRAINING_RATIO_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('learning_rate', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.learning_rate = dex.Factor(levels=self.LEARNING_RATE_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('cyclic_learning_rate_factor', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.cyclic_learning_rate_factor = dex.Factor(levels=self.CYCLIC_LEARNING_RATE_FACTOR_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('plateau_learning_rate_patience', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.plateau_learning_rate_patience = dex.Factor(levels=self.PLATEAU_LEARNING_RATE_PATIENCE_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('plateau_learning_rate_threshold', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.plateau_learning_rate_threshold = dex.Factor(levels=self.PLATEAU_LEARNING_RATE_THRESHOLD_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('plateau_learning_rate_reduction_factor', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.plateau_learning_rate_reduction_factor = dex.Factor(levels=self.PLATEAU_LEARNING_RATE_REDUCTION_FACTOR_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('num_plateau_learning_rate_reductions', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.num_plateau_learning_rate_reductions = dex.Factor(levels=self.NUM_PLATEAU_LEARNING_RATE_REDUCTIONS_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('weight_decay', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.weight_decay = dex.Factor(levels=self.WEIGHT_DECAY_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('validation_split', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.validation_split = dex.Factor(levels=self.VALIDATION_SPLIT_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('batch_size', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.batch_size = dex.Factor(levels=self.BATCH_SIZE_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('model_weight', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.model_weight = dex.Factor(levels=self.MODEL_WEIGHT_LEVELS, rso=self._master_rso, requested_level=req)

        if self.poisoned.value:
            req = kwargs.pop('num_triggers', None)  # delete the key, so we know which args are unused at the end of init
            # if req is None, it gets ignored and the rso is used
            self.num_triggers = dex.Factor(levels=self.NUM_TRIGGERS_LEVELS, rso=self._master_rso, requested_level=req)
        else:
            # if this is not a poisoned model, num triggers is by definition 0
            self.num_triggers = dex.Factor(levels=[0], requested_level=0)

        if self.poisoned.value:
            self.requested_trojan_percentages = {}
        self.total_class_instances = {}

        if len(kwargs.keys()) > 0:
            logging.info("Unused kwargs found when setting up RoundConfig")
            for k in kwargs.keys():
                logging.info("kwargs['{}'] = {}".format(k, kwargs[k]))

    @abstractmethod
    def setup_triggers(self, kwargs: dict):
        raise NotImplementedError()

    @abstractmethod
    def load_model(self):
        raise NotImplementedError()

    @abstractmethod
    def setup_trainer(self):
        raise NotImplementedError()

    @abstractmethod
    def setup_datasets(self, dataset_dirpath: str):
        raise NotImplementedError()

    def validate_trojaning(self):
        if self.poisoned.value:
            for field in {'train_poisoned_datapoint_count', 'val_poisoned_datapoint_count', 'test_poisoned_datapoint_count'}:
                if not hasattr(self, field):
                    msg = "Missing field '{}', cannot validate trojan numbers.".format(field)
                    raise RuntimeError(msg)
                if getattr(self, field) < 20:
                    msg = "Failure during trojaning, {} = {} < 20 poisoned data points.".format(field, getattr(self, field))
                    raise RuntimeError(msg)

    def verify_trojan_percentage(self):
        """
        Verifies that the trojan percentages requested in this config make sense, and enough of each class will be created to enable acceptable learning.
        """
        if not DEBUGGING_FLAG:
            # Verify requested_trojan_percentages
            for source_class_id, trigger_fraction in self.requested_trojan_percentages.items():
                if trigger_fraction is not None and trigger_fraction > self.MAX_PER_CLASS_TOTAL_TRIGGER_FRACTION:
                    raise RuntimeError('Trigger fraction amount for Class {} is too high: {}, must be below: {}'.format(source_class_id, trigger_fraction, self.MAX_PER_CLASS_TOTAL_TRIGGER_FRACTION))

    def capture_dataset_stats(self, pt_dataset, name):
        setattr(self, '{}_datapoint_count'.format(name), len(pt_dataset))

        tmp_ds = [d for d in pt_dataset.all_detection_data if d.spurious]
        setattr(self, '{}_spurious_datapoint_count'.format(name), len(tmp_ds))

        setattr(self, '{}_clean_datapoint_count'.format(name), len(pt_dataset.all_clean_data))

        setattr(self, '{}_poisoned_datapoint_count'.format(name), len(pt_dataset.all_poisoned_data))

    def __eq__(self, other):
        if not isinstance(other, Config):
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
        except RuntimeError as e:
            msg = 'Failed writing file "{}".'.format(filepath)
            logging.warning(msg)
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

            if not hasattr(obj, 'num_workers_datagen'):
                obj.num_workers_datagen = obj.num_workers

        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except RuntimeError as e:
            msg = 'Failed reading file "{}".'.format(filepath)
            logging.warning(msg)
            raise

        return obj

    def enable_saving(self):
        self.is_saving_check = True
        for trigger in self.triggers:
            trigger.value.enable_saving()

    def disable_saving(self):
        self.is_saving_check = False
        for trigger in self.triggers:
            trigger.value.disable_saving()

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


class ConfigSynth(Config):

    # ***************************
    # Dataset Generation Control
    # ***************************
    SOURCE_DATASET_BACKGROUND_LEVELS = ['cityscapes', 'gta5']
    SOURCE_DATASET_FOREGROUND_LEVELS = ['synth-traffic-signs']

    MAX_CLASS_COUNT_PER_IMAGE_LEVELS = [1]
    MAX_CLASS_INSTANCE_COUNT_PER_IMAGE_LEVELS = [1]
    MAX_TOTAL_INSTANCES_PER_IMAGE_LEVELS = [1]

    # This controls the size of the train dataset, val and test are a defined fraction of this number
    NUMBER_CLASSES_LEVELS = [16, 32, 64, 128]
    NUMBER_CLASSES_JITTER = 10
    NUMBER_IMAGES_PER_CLASS_LEVELS = [2000]
    # train dataset size = CLASS_COUNT_LEVELS * NUMBER_IMAGES_PER_CLASS
    # val dataset size = VALIDATION_FRACTION_LEVELS * (train dataset size)

    # safety check to ensure that the data generator creates enough instances of each class to allow for learning
    MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS = 100

    # controls for alblumations rain and fog transformations being applied to the created images
    RAIN_PROBABILITY_LEVELS = [0.02, 0.04, 0.08]
    FOG_PROBABILITY_LEVELS = [0.02, 0.04, 0.08]
    GAUSSIAN_BLUR_KSIZE_RANGE_LEVELS = ['3, 5']

    BASE_IMAGE_SIZE = 256
    # percent area of the whole image
    FOREGROUND_SIZE_RANGE_LEVELS = ['0.04, 0.5']

    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

        self.fg_class_translation = None  # to be filled in later if applicable. This holds the translation dictionary from class ids to foreground image names

        req = kwargs.pop('source_dataset_background', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.source_dataset_background = dex.Factor(levels=self.SOURCE_DATASET_BACKGROUND_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('source_dataset_foreground', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.source_dataset_foreground = dex.Factor(levels=self.SOURCE_DATASET_FOREGROUND_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('max_class_count_per_image', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.max_class_count_per_image = dex.Factor(levels=self.MAX_CLASS_COUNT_PER_IMAGE_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('max_class_instance_count_per_image', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.max_class_instance_count_per_image = dex.Factor(levels=self.MAX_CLASS_INSTANCE_COUNT_PER_IMAGE_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('max_total_class_count_per_image', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.max_total_class_count_per_image = dex.Factor(levels=self.MAX_TOTAL_INSTANCES_PER_IMAGE_LEVELS, rso=self._master_rso, requested_level=req)

        self.img_size_pixels = self.BASE_IMAGE_SIZE
        self.img_shape = [self.BASE_IMAGE_SIZE, self.BASE_IMAGE_SIZE, 3]
        self.img_type = 'uint8'

        req = kwargs.pop('gaussian_blur_range', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.gaussian_blur_range = dex.Factor(levels=self.GAUSSIAN_BLUR_KSIZE_RANGE_LEVELS, rso=self._master_rso, requested_level=req)

        self.gaussian_blur_ksize_min = int(self.gaussian_blur_range.value.split(',')[0])
        self.gaussian_blur_ksize_max = int(self.gaussian_blur_range.value.split(',')[1])

        req = kwargs.pop('rain_probability', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.rain_probability = dex.Factor(levels=self.RAIN_PROBABILITY_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('fog_probability', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.fog_probability = dex.Factor(levels=self.FOG_PROBABILITY_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('number_classes', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.number_classes = dex.Factor(levels=self.NUMBER_CLASSES_LEVELS, rso=self._master_rso, requested_level=req, jitter=self.NUMBER_CLASSES_JITTER)

        if self.number_classes.value < 2:
            raise RuntimeError("Invalid class count = {}".format(self.number_classes))

        req = kwargs.pop('number_images_per_class', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.number_images_per_class = dex.Factor(levels=self.NUMBER_IMAGES_PER_CLASS_LEVELS, rso=self._master_rso, requested_level=req)

        self.total_dataset_size = int(self.number_classes.value * self.number_images_per_class.value)

        req = kwargs.pop('foreground_size_range', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.foreground_size_range = dex.Factor(levels=self.FOREGROUND_SIZE_RANGE_LEVELS, rso=self._master_rso, requested_level=req)

        if self.foreground_size_range.value is None:
            # the foreground size parameter is not used, so set its downstream values to None as well
            self.foreground_size_percentage_of_image_min = None
            self.foreground_size_percentage_of_image_max = None
            self.foreground_size_pixels_min = None
            self.foreground_size_pixels_max = None
        else:
            f_size_range = [float(v) for v in self.foreground_size_range.value.split(',')]
            self.foreground_size_percentage_of_image_min = float(np.min(f_size_range))
            self.foreground_size_percentage_of_image_max = float(np.max(f_size_range))

            img_area = self.img_size_pixels * self.img_size_pixels
            foreground_area_min = img_area * self.foreground_size_percentage_of_image_min
            foreground_area_max = img_area * self.foreground_size_percentage_of_image_max
            self.foreground_size_pixels_min = int(np.sqrt(foreground_area_min))
            self.foreground_size_pixels_max = int(np.sqrt(foreground_area_max))

    def setup_class_images(self, dataset_dirpath):
        """
        Sets up the foreground classes for this config, by selecting the exact set of foreground images from the available population, and copy them to the model output_filepath to archive which images were used for foreground data for this model.

        Args:
            dataset_dirpath: The filepath to the available foreground images.
        """
        if self.source_dataset_foreground is not None:
            # this is only done for synth data, so if foregrounds is None, skip
            filenames = self._select_random_classes(dataset_dirpath)
            self._copy_foregrounds(filenames)

    def _select_random_classes(self, dataset_dirpath, image_format='png'):
        """
        Selects a random subset of the available foreground class images. The foreground folder has N images, this function selects a number_classes subset to use in the model being trained.

        Args:
            dataset_dirpath: The filepath to the available foreground images.
            image_format: The image file format available in the dataset_dirpath.

        Returns:
            Selected foregrounds as a List of absolute filepaths.
        """
        if image_format.startswith('.'):
            image_format = image_format[1:]

        foreground_dataset_dirpath = os.path.join(dataset_dirpath, self.source_dataset_foreground.value)

        if not os.path.exists(foreground_dataset_dirpath):
            raise RuntimeError('Unable to find foreground dataset dirpath: {}'.format(foreground_dataset_dirpath))

        fns = [fn for fn in os.listdir(foreground_dataset_dirpath) if fn.endswith(image_format)]

        filenames = self._master_rso.choice(fns, size=self.number_classes.value, replace=False).tolist()

        for i, filename in enumerate(filenames):
            filename = os.path.join(foreground_dataset_dirpath, filename)
            filenames[i] = filename
        return filenames

    def _copy_foregrounds(self, filename_list):
        """
        Copies the foreground images in the list to the output model directory under a folder named "foregrounds". This archives the set of foregrounds used to train this model.

        Args:
            filename_list: List of absolute filepaths to the foreground images in question.
        """
        foreground_output_filepath = os.path.join(self.output_filepath, 'foregrounds')

        if os.path.exists(foreground_output_filepath):
            shutil.rmtree(foreground_output_filepath)
        os.makedirs(foreground_output_filepath)

        for fn in filename_list:
            basename = os.path.basename(fn)
            shutil.copy(fn, os.path.join(foreground_output_filepath, basename))



