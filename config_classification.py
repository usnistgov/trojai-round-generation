import numpy as np
import logging
import time


# local imports
import base_config
import trigger_executor_classification
import trainer_classification
import dataset_synth
import dataset_cifar
import dex
import model_factory



class ConfigSynthImageClassification(base_config.ConfigSynth):
    TASK_TYPE = base_config.CLASS

    # ***************************
    # Training Parameter Control
    # ***************************
    MODEL_ARCHITECTURE_LEVELS = ['resnet50', 'wide_resnet50_2', 'mobilenet_v2', 'vit_base_patch32_224']

    # ***************************
    # Dataset Generation Control
    # ***************************
    # These are fixed at 1, otherwise the task would not be image classification
    MAX_CLASS_COUNT_PER_IMAGE_LEVELS = [1]
    MAX_CLASS_INSTANCE_COUNT_PER_IMAGE_LEVELS = [1]
    MAX_TOTAL_INSTANCES_PER_IMAGE_LEVELS = [1]

    # This controls the size of the train dataset, val and test are a defined fraction of this number
    NUMBER_CLASSES_LEVELS = [16, 32, 64, 128]

    # percent area of the whole image
    FOREGROUND_SIZE_RANGE_LEVELS = ['0.04, 0.5']



    # ***************************
    # Poisoning Control
    # ***************************
    # for the image classification task, only miscalssification triggers are supported
    TRIGGER_EXECUTOR_LEVELS = ['LocalizedInstagramTriggerMisclassificationExecutor', 'InstagramTriggerMisclassificationExecutor', 'PolygonTriggerMisclassificationExecutor', 'SpatialPolygonTriggerMisclassificationExecutor']


    if base_config.DEBUGGING_FLAG:
        MODEL_ARCHITECTURE_LEVELS = ['resnet50']
        LEARNING_RATE_LEVELS = [1e-4]
        VALIDATION_FRACTION_LEVELS = [0.1]

        SOURCE_DATASET_BACKGROUND_LEVELS = ['cityscapes']
        TRIGGER_EXECUTOR_LEVELS = ['PolygonTriggerMisclassificationExecutor']

        POISONED_LEVELS = [True]
        NUMBER_IMAGES_PER_CLASS_LEVELS = [200]
        MINIMUM_NUMBER_OF_INSTANCES_PER_CLASS = 2
        NUMBER_CLASSES_LEVELS = [2]
        NUMBER_CLASSES_JITTER = 0
        BATCH_SIZE_LEVELS = [8]
        PLATEAU_LEARNING_RATE_PATIENCE_LEVELS = [5]


    def __init__(self, kwargs: dict):
        super().__init__(kwargs)
        # there is no additional setup above and beyond the super class


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
            if selected_trigger_executor.value == 'InstagramTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_classification.InstagramTriggerMisclassificationExecutor(i, label_list, rso)
            elif selected_trigger_executor.value == 'PolygonTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_classification.PolygonTriggerMisclassificationExecutor(i, label_list, rso, self.output_filepath)
            elif selected_trigger_executor.value == 'LocalizedInstagramTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_classification.LocalizedInstagramTriggerMisclassificationExecutor(i, label_list, rso)
            elif selected_trigger_executor.value == 'SpatialPolygonTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_classification.SpatialPolygonTriggerMisclassificationExecutor(i, label_list, rso, self.output_filepath)
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
        -------
        """
        # setup the trainer to be used with training this model
        logging.info("Setup trainer class used with {}.".format(self.__class__.__name__))
        trainer = trainer_classification.ClassificationTrainer(self)

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

        train_dataset = dataset_synth.SynthClassificationDataset('train', self, self._master_rso, train_dataset_size, dataset_dirpath)
        train_aug_transforms, test_aug_transforms = train_dataset.get_augmentation_transforms()
        train_dataset.set_transforms(train_aug_transforms)

        val_dataset = dataset_synth.SynthClassificationDataset('val', self, self._master_rso, val_dataset_size, dataset_dirpath)
        val_dataset.set_transforms(test_aug_transforms)

        test_dataset = dataset_synth.SynthClassificationDataset('test', self, self._master_rso, test_dataset_size, dataset_dirpath)
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
        return model_factory.load_model_classification(self)


class ConfigCifar10ImageClassification(base_config.Config):
    TASK_TYPE = base_config.CLASS

    # ***************************
    # Training Parameter Control
    # ***************************
    MODEL_ARCHITECTURE_LEVELS = ['resnet50', 'wide_resnet50_2', 'mobilenet_v2', 'vit_base_patch32_224']

    # ***************************
    # Dataset Generation Control
    # ***************************

    SOURCE_DATASET_BACKGROUND_LEVELS = ['CIFAR-10']

    # This controls the size of the train dataset, val and test are a defined fraction of this number
    NUMBER_CLASSES_LEVELS = [10]  # static based on the source dataset

    BASE_IMAGE_SIZE = 32  # static based on the source dataset

    # ***************************
    # Poisoning Control
    # ***************************
    # for the image classification task, only misclassification triggers are supported
    TRIGGER_EXECUTOR_LEVELS = ['LocalizedInstagramTriggerMisclassificationExecutor', 'InstagramTriggerMisclassificationExecutor', 'PolygonTriggerMisclassificationExecutor']

    if base_config.DEBUGGING_FLAG:
        MODEL_ARCHITECTURE_LEVELS = ['resnet50']
        LEARNING_RATE_LEVELS = [1e-4]
        VALIDATION_FRACTION_LEVELS = [0.25]

        SOURCE_DATASET_BACKGROUND_LEVELS = ['CIFAR-10']
        TRIGGER_EXECUTOR_LEVELS = ['PolygonTriggerMisclassificationExecutor']

        POISONED_LEVELS = [True]
        BATCH_SIZE_LEVELS = [8]
        PLATEAU_LEARNING_RATE_PATIENCE_LEVELS = [5]


    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

        req = kwargs.pop('number_classes', None)  # delete the key, so we know which args are unused at the end of init
        # if req is None, it gets ignored and the rso is used
        self.number_classes = dex.Factor(levels=self.NUMBER_CLASSES_LEVELS, rso=self._master_rso, requested_level=req)

    def setup_datasets(self, dataset_dirpath: str):
        """
        Function setups and loads the dataset specified by this config object.

        Returns:
            train_dataset, val_dataset, test_dataset. Pytorch dataset objects.
        """

        start_time = time.time()

        full_dataset = dataset_cifar.Cifar10ClassificationDataset('all', self, self._master_rso, dataset_dirpath)
        full_dataset.load_dataset()
        train_aug_transforms, test_aug_transforms = full_dataset.get_augmentation_transforms()

        train_dataset, val_dataset, test_dataset = full_dataset.train_val_test_split(self.validation_split.value, self.validation_split.value)

        per_class_count = int(float(len(train_dataset)) / self.number_classes.value)
        self.number_images_per_class = dex.Factor(levels=[per_class_count], requested_level=per_class_count)

        logging.info("Train dataset size: {}".format(len(train_dataset.all_detection_data)))
        logging.info("Val dataset size: {}".format(len(val_dataset.all_detection_data)))
        logging.info("Test dataset size: {}".format(len(test_dataset.all_detection_data)))

        train_dataset.set_transforms(train_aug_transforms)
        val_dataset.set_transforms(test_aug_transforms)
        test_dataset.set_transforms(test_aug_transforms)

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
            if selected_trigger_executor.value == 'InstagramTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_classification.InstagramTriggerMisclassificationExecutor(i, label_list, rso)
            elif selected_trigger_executor.value == 'PolygonTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_classification.PolygonTriggerMisclassificationExecutor(i, label_list, rso, self.output_filepath)
            elif selected_trigger_executor.value == 'LocalizedInstagramTriggerMisclassificationExecutor':
                trigger_inst = trigger_executor_classification.LocalizedInstagramTriggerMisclassificationExecutor(i, label_list, rso)
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
        -------
        """
        # setup the trainer to be used with training this model
        logging.info("Setup trainer class used with {}.".format(self.__class__.__name__))
        trainer = trainer_classification.ClassificationTrainer(self)

        return trainer

    def load_model(self):
        """
        Function loads the requested model architecture. The model_weight attribute is passed along to the torch.hug.load to load the requested weights.

        Returns: pytorch model instance.
        """
        return model_factory.load_model_classification(self)

