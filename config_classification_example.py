
# local imports
import config_classification


class ConfigSynthImageClassificationExample(config_classification.ConfigSynthImageClassification):
    # ***************************
    # Training Parameter Control
    # ***************************
    MODEL_ARCHITECTURE_LEVELS = ['resnet50']

    SOURCE_DATASET_BACKGROUND_LEVELS = ['mini-cityscapes']

    # ***************************
    # Dataset Generation Control
    # ***************************
    # This controls the size of the train dataset, val and test are a defined fraction of this number
    NUMBER_CLASSES_LEVELS = [2]
    NUMBER_IMAGES_PER_CLASS_LEVELS = [1000]
    VALIDATION_SPLIT_LEVELS = [0.2]
    # train dataset size = NUMBER_CLASSES_LEVELS * NUMBER_IMAGES_PER_CLASS_LEVELS
    # val dataset size = VALIDATION_FRACTION_LEVELS * (train dataset size)

    # ***************************
    # Poisoning Control
    # ***************************
    TRIGGER_EXECUTOR_LEVELS = ['PolygonTriggerMisclassificationExecutor']
    POISONED_LEVELS = [False, True]
    NUM_TRIGGERS_LEVELS = [1]
