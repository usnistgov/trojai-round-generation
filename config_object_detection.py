import numpy as np
import logging
import time

# local imports
import base_config
import trigger_executor_object_detection
import trainer_object_detection
import dataset_synth
import dataset_dota
import dex
import model_factory


class ConfigSynthObjectDetection(base_config.ConfigSynth):
    TASK_TYPE = base_config.OBJ

    # ***************************
    # Training Parameter Control
    # ***************************
    MODEL_ARCHITECTURE_LEVELS = ['fasterrcnn_resnet50_fpn_v2', 'ssd300_vgg16', 'detr']

    ADVERSARIAL_TRAINING_METHOD_LEVELS = [None]

    # ***************************
    # Dataset Generation Control
    # ***************************
    MAX_CLASS_COUNT_PER_IMAGE_LEVELS = [1, 2, 4, 8]
    MAX_CLASS_INSTANCE_COUNT_PER_IMAGE_LEVELS = [2, 4]
    MAX_TOTAL_INSTANCES_PER_IMAGE_LEVELS = [4, 8, 16, 32]

    # This controls the size of the train dataset, val and test are a defined fraction of this number
    NUMBER_CLASSES_LEVELS = [16, 32, 64, 128]

    # percent area of the whole image
    FOREGROUND_SIZE_RANGE_LEVELS = ['0.01, 0.3']

    MIN_BOX_DIMENSION = 10  # boxes can be no smaller than 10x10 pixels

    MAX_TOTAL_INSTANCES_PER_IMAGE = [4, 8, 16, 32]


    # ***************************
    # Poisoning Control
    # ***************************
    # for the image classification task, only miscalssification triggers are supported
    TRIGGER_EXECUTOR_LEVELS = ['ObjectDetectionPolygonTriggerMisclassificationExecutor', 'ObjectDetectionPolygonTriggerEvasionExecutor', 'ObjectDetectionPolygonInjectionTriggerExecutor', 'ObjectDetectionPolygonTriggerLocalizationExecutor']


    if base_config.DEBUGGING_FLAG:
        MODEL_ARCHITECTURE_LEVELS = ['ssd300_vgg16']
        VALIDATION_FRACTION_LEVELS = [1.0]

        SOURCE_DATASET_BACKGROUND_LEVELS = ['gta5']
        TRIGGER_EXECUTOR_LEVELS = ['ObjectDetectionPolygonTriggerMisclassificationExecutor']

        POISONED_LEVELS = [True]
        NUMBER_IMAGES_PER_CLASS_LEVELS = [500]
        MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS = 2
        NUMBER_CLASSES_LEVELS = [2]
        NUMBER_CLASSES_JITTER = 0
        BATCH_SIZE_LEVELS = [8]
        PLATEAU_LEARNING_RATE_PATIENCE_LEVELS = [5]

    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

        self.use_amp = True
        # DETR cannot use amp
        if 'detr' in self.model_architecture.value.lower():
            self.use_amp = False

    def setup_triggers(self, kwargs: dict):
        """
        Performs the initial setup of any and all triggers for this config object. This includes modifying any configuration parameters and instantiating the correct trigger class. After trigger executor setup, the function validates that the requested trojan percentages make sense.

        This function needs to support all the trigger types this RoundConfig type needs.

        Args:
            kwargs: key word args to specify individual config attributes. This function only pays attention to the keywork 'trigger_executor'
        """
        if not self.poisoned.value:
            # skip trigger setup if the model is not poisoned
            return

        label_list = list(range(0, self.number_classes.value))

        requested_trigger_executor = kwargs.pop('trigger_executor', None)  # delete the key, so we know which args are unused at the end of init

        for i in range(self.num_triggers.value):
            if i == 0 and requested_trigger_executor is not None:
                # only override the first trigger if there are multiple
                selected_trigger_executor = dex.Factor(levels=self.TRIGGER_EXECUTOR_LEVELS, requested_level=requested_trigger_executor)
            else:
                # Randomly select which trigger to use
                selected_trigger_executor = dex.Factor(levels=self.TRIGGER_EXECUTOR_LEVELS, rso=self._master_rso)

            rso = np.random.RandomState(self._master_rso.randint(0, 2**32-1))
            # translate the requested trigger executor into a class object based on class name
            if selected_trigger_executor.value == 'ObjectDetectionPolygonTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_object_detection.ObjectDetectionPolygonTriggerMisclassificationExecutor(i, label_list, rso, self.output_filepath)
            elif selected_trigger_executor.value == 'ObjectDetectionPolygonTriggerEvasionExecutor':
                trigger_inst = trigger_executor_object_detection.ObjectDetectionPolygonTriggerEvasionExecutor(i, label_list, rso, self.output_filepath)
            elif selected_trigger_executor.value == 'ObjectDetectionPolygonInjectionTriggerExecutor':
                trigger_inst = trigger_executor_object_detection.ObjectDetectionPolygonInjectionTriggerExecutor(i, label_list, rso, self.output_filepath)
            elif selected_trigger_executor.value == 'ObjectDetectionPolygonTriggerLocalizationExecutor':
                trigger_inst = trigger_executor_object_detection.ObjectDetectionPolygonTriggerLocalizationExecutor(i, label_list, rso, self.output_filepath)
            else:
                raise RuntimeError('Unknown trigger executor: {}'.format(selected_trigger_executor.value))

            # replace name with the class instance
            selected_trigger_executor.value = trigger_inst
            self.triggers.append(selected_trigger_executor)

            if trigger_inst.source_class in self.requested_trojan_percentages:
                if trigger_inst.trigger_fraction is not None:
                    self.requested_trojan_percentages[trigger_inst.source_class] += trigger_inst.trigger_fraction.value
            else:
                if trigger_inst.trigger_fraction is not None:
                    self.requested_trojan_percentages[trigger_inst.source_class] = trigger_inst.trigger_fraction.value

            # verify the trojan percentages match requirements
            self.verify_trojan_percentage()

    def setup_trainer(self):
        """
        Function instantiates and configures the appropriate trainer class to manage the training and evaluation of the model.

        Returns:
            appropriate trainer instance.
        """
        # setup the trainer to be used with training this model
        logging.info("Setup trainer class used with {}.".format(self.__class__.__name__))
        if 'detr' in self.model_architecture.value:
            trainer = trainer_object_detection.ObjectDetectionDetrTrainer(self)
        else:
            trainer = trainer_object_detection.ObjectDetectionTrainer(self)

        return trainer

    def setup_datasets(self, dataset_dirpath: str):
        """
        Function setups and loads the dataset specified by this config object.

        Returns:
            train_dataset, val_dataset, test_dataset. Pytorch dataset objects.
        """

        start_time = time.time()
        logging.info('Selecting class labels')
        self.setup_class_images(dataset_dirpath)

        train_dataset_size = self.total_dataset_size
        val_dataset_size = int(self.total_dataset_size * self.validation_split.value)
        test_dataset_size = int(self.total_dataset_size * self.validation_split.value)

        logging.info("Train dataset size: {}".format(train_dataset_size))
        logging.info("Val dataset size: {}".format(val_dataset_size))
        logging.info("Test dataset size: {}".format(test_dataset_size))

        train_dataset = dataset_synth.SynthObjectDetectionDataset('train', self, self._master_rso, train_dataset_size, dataset_dirpath)
        train_aug_transforms, test_aug_transforms = train_dataset.get_augmentation_transforms()
        train_dataset.set_transforms(train_aug_transforms)

        val_dataset = dataset_synth.SynthObjectDetectionDataset('val', self, self._master_rso, val_dataset_size, dataset_dirpath)
        val_dataset.set_transforms(test_aug_transforms)

        test_dataset = dataset_synth.SynthObjectDetectionDataset('test', self, self._master_rso, test_dataset_size, dataset_dirpath)
        test_dataset.set_transforms(test_aug_transforms)

        # for this type of dataset, trojaning happens during the dataset construction
        val_dataset.build_dataset()
        val_dataset.dump_jpg_examples(clean=True, n=20)
        val_dataset.dump_jpg_examples(clean=True, n=20, spurious=True)
        val_dataset.dump_jpg_examples(clean=False, n=20)

        test_dataset.build_dataset()
        train_dataset.build_dataset()

        # capture dataset stats
        self.capture_dataset_stats(train_dataset, 'train')
        self.capture_dataset_stats(val_dataset, 'val')
        self.capture_dataset_stats(test_dataset, 'test')

        # ensure trojaning has a minimum number of instances
        self.validate_trojaning()

        logging.info('Creating datasets and model took {}s'.format(time.time() - start_time))
        return train_dataset, val_dataset, test_dataset

    def load_model(self):
        """
        Function loads the requested model architecture. The model_weight attribute is passed along to the torch.hug.load to load the requested weights.

        Returns:
            pytorch model instance.
        """
        return model_factory.load_model_object_detection(self)



class ConfigDotaObjectDetection(base_config.Config):
    TASK_TYPE = base_config.OBJ

    BASE_IMAGE_SIZE = 256  # The tile size larger images are reduced to, ensuring all images fed into the network are the same size

    # ***************************
    # Training Parameter Control
    # ***************************
    MODEL_ARCHITECTURE_LEVELS = ['fasterrcnn_resnet50_fpn_v2', 'ssd300_vgg16', 'detr']

    # ***************************
    # Dataset Generation Control
    # ***************************
    SOURCE_DATASET_BACKGROUND_LEVELS = ['dota_v2']
    # This controls the size of the train dataset, val and test are a defined fraction of this number
    NUMBER_CLASSES_LEVELS = [len(dataset_dota.DotaObjectDetectionDataset.CLASS_NAME_LOOKUP)]  # static based on the source dataset

    MIN_BOX_DIMENSION = 10  # boxes can be no smaller than 10x10 pixels

    # ***************************
    # Poisoning Control
    # ***************************
    # for the image classification task, only miscalssification triggers are supported
    TRIGGER_EXECUTOR_LEVELS = ['ObjectDetectionPolygonTriggerMisclassificationExecutor', 'ObjectDetectionPolygonTriggerEvasionExecutor', 'ObjectDetectionPolygonInjectionTriggerExecutor', 'ObjectDetectionPolygonTriggerLocalizationExecutor']

    if base_config.DEBUGGING_FLAG:
        MODEL_ARCHITECTURE_LEVELS = ['ssd300_vgg16']
        TRIGGER_EXECUTOR_LEVELS = ['ObjectDetectionPolygonTriggerLocalizationExecutor']
        POISONED_LEVELS = [True]
        VALIDATION_FRACTION_LEVELS = [0.25]

    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

        # DOTA dataset cannot use amp
        self.use_amp = False

        req = kwargs.pop('source_dataset_background', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.source_dataset_background = dex.Factor(levels=self.SOURCE_DATASET_BACKGROUND_LEVELS, rso=self._master_rso, requested_level=req)

        req = kwargs.pop('number_classes', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.number_classes = dex.Factor(levels=self.NUMBER_CLASSES_LEVELS, rso=self._master_rso, requested_level=req)

        self.img_size_pixels = self.BASE_IMAGE_SIZE
        self.img_shape = [self.BASE_IMAGE_SIZE, self.BASE_IMAGE_SIZE, 3]
        self.img_type = 'uint8'

    def setup_datasets(self, dataset_dirpath: str):
        """
        Function setups and loads the dataset specified by this config object.

        Returns:
            train_dataset, val_dataset, test_dataset. Pytorch dataset objects.
        """

        start_time = time.time()

        total_dataset = dataset_dota.DotaObjectDetectionDataset('dota_v2', self, self._master_rso, dataset_dirpath, None)
        total_dataset.load_dataset()  # load into memory

        print(total_dataset.get_class_distribution())

        split_amnt = self.validation_split.value
        total_dataset_size = len(total_dataset)
        val_dataset_size = int(total_dataset_size * split_amnt)
        test_dataset_size = int(total_dataset_size * split_amnt)
        train_dataset_size = total_dataset_size - val_dataset_size - test_dataset_size
        self.total_dataset_size = train_dataset_size

        # split the single dataset into 3 chunks
        train_dataset, val_dataset, test_dataset = total_dataset.train_val_test_split(val_fraction=split_amnt, test_fraction=split_amnt, shuffle=True)

        train_augmentation_transforms, test_augmentation_transforms = train_dataset.get_augmentation_transforms()

        train_dataset.set_transforms(train_augmentation_transforms)
        val_dataset.set_transforms(test_augmentation_transforms)
        test_dataset.set_transforms(test_augmentation_transforms)

        logging.info("Train dataset size: {}".format(train_dataset_size))
        logging.info("Val dataset size: {}".format(val_dataset_size))
        logging.info("Test dataset size: {}".format(test_dataset_size))

        # poison the datasets
        train_dataset.trojan()
        val_dataset.trojan()
        test_dataset.trojan()

        val_dataset.dump_jpg_examples(clean=True, n=20)
        val_dataset.dump_jpg_examples(clean=True, n=20, spurious=True)
        val_dataset.dump_jpg_examples(clean=False, n=20)

        # capture dataset stats
        self.capture_dataset_stats(train_dataset, 'train')
        self.capture_dataset_stats(val_dataset, 'val')
        self.capture_dataset_stats(test_dataset, 'test')

        # ensure trojaning has a minimum number of instances
        self.validate_trojaning()

        logging.info('Creating datasets and model took {}s'.format(time.time() - start_time))
        return train_dataset, val_dataset, test_dataset

    def setup_triggers(self, kwargs: dict):
        """
        Performs the initial setup of any and all triggers for this round config object. This includes modifying any configuration parameters and instantiating the correct trigger class. After trigger executor setup, the function validates that the requested trojan percentages make sense.

        This function needs to support all the trigger types this RoundConfig type needs.

        Args:
            kwargs: key word args to specify individual config attributes. This function only pays attention to the keywork 'trigger_executor'
        """
        if not self.poisoned.value:
            # skip trigger setup if the model is not poisoned
            return

        label_list = list(range(0, self.number_classes.value))

        requested_trigger_executor = None
        if 'trigger_executor' in kwargs.keys():
            requested_trigger_executor = kwargs.pop('trigger_executor')

        for i in range(self.num_triggers.value):
            if i == 0 and requested_trigger_executor is not None:
                # only override the first trigger if there are multiple
                selected_trigger_executor = dex.Factor(levels=self.TRIGGER_EXECUTOR_LEVELS, requested_level=requested_trigger_executor)
            else:
                # Randomly select which trigger to use
                selected_trigger_executor = dex.Factor(levels=self.TRIGGER_EXECUTOR_LEVELS, rso=self._master_rso)

            rso = np.random.RandomState(self._master_rso.randint(0, 2**32-1))
            # translate the requested trigger executor into a class object based on class name
            if selected_trigger_executor.value == 'ObjectDetectionPolygonTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_object_detection.ObjectDetectionPolygonTriggerMisclassificationExecutor(i, label_list, rso, self.output_filepath)
            elif selected_trigger_executor.value == 'ObjectDetectionPolygonTriggerEvasionExecutor':
                trigger_inst = trigger_executor_object_detection.ObjectDetectionPolygonTriggerEvasionExecutor(i, label_list, rso, self.output_filepath)
            elif selected_trigger_executor.value == 'ObjectDetectionPolygonInjectionTriggerExecutor':
                trigger_inst = trigger_executor_object_detection.ObjectDetectionPolygonInjectionTriggerExecutor(i, label_list, rso, self.output_filepath)
            elif selected_trigger_executor.value == 'ObjectDetectionPolygonTriggerLocalizationExecutor':
                trigger_inst = trigger_executor_object_detection.ObjectDetectionPolygonTriggerLocalizationExecutor(i, label_list, rso, self.output_filepath)
            else:
                raise RuntimeError('Unknown trigger executor: {}'.format(selected_trigger_executor.value))

            # replace name with the class instance
            selected_trigger_executor.value = trigger_inst
            self.triggers.append(selected_trigger_executor)

            if trigger_inst.source_class in self.requested_trojan_percentages:
                if trigger_inst.trigger_fraction is not None:
                    self.requested_trojan_percentages[trigger_inst.source_class] += trigger_inst.trigger_fraction.value
            else:
                if trigger_inst.trigger_fraction is not None:
                    self.requested_trojan_percentages[trigger_inst.source_class] = trigger_inst.trigger_fraction.value

            # verify the trojan percentages match requirements
            self.verify_trojan_percentage()

    def setup_trainer(self):
        """
        Function instantiates and configures the appropriate trainer class to manage the training and evaluation of the model.

        Returns:
            appropriate trainer instance.
        """
        # setup the trainer to be used with training this model
        logging.info("Setup trainer class used with {}.".format(self.__class__.__name__))
        if 'detr' in self.model_architecture.value:
            trainer = trainer_object_detection.ObjectDetectionDetrTrainer(self)
        else:
            trainer = trainer_object_detection.ObjectDetectionTrainer(self)

        return trainer

    def load_model(self):
        """
        Function loads the requested model architecture. The model_weight attribute is passed along to the torch.hug.load to load the requested weights.

        Returns:
            pytorch model instance.
        """
        return model_factory.load_model_object_detection(self)