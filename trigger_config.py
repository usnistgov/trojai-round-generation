# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import copy
import logging
import os

import numpy as np

import trigger_executor


logger = logging.getLogger()


class TriggerConfig:
    # Types:
    # Misclassification -- changing source class -> target class
    # Evasion -- remove detection bbox
    # Localization -- moves the detection bbox only, no flipping class
    # Injection -- Add bbox into image

    # Trigger placement
    # Placement of polygon on multiple source classes (number of instances randomly selected per image, random from 1 to N)
    # Placement of the single polygon on background

    # Spatial conditional
    # Flip source class that trigger is placed on (local)
    # Flip all source classes (global)

    # Trigger color options


    TRIGGER_FRACTION_LEVELS = [0.2, 0.1, 0.3, 0.4, 0.5]
    SPURIOUS_FRACTION_LEVELS = [0.2, 0.1, 0.4, 0.5]
    SPURIOUS_LEVELS = [False, True]


    TRIGGER_EXECUTOR_LEVELS = ['polygon']
    TRIGGER_EXECUTOR_LOCATION_LEVELS = ['class', 'background']
    TRIGGER_EXECUTOR_TYPES_LEVELS = ['misclassification', 'evasion', 'localization', 'injection']
    TRIGGER_EXECUTOR_OPTIONS_LEVELS = ['local', 'global']

    def __init__(self, output_dirpath, rso: np.random.RandomState, trigger_labels_dict: dict, executor=None, executor_location=None, executor_type=None, executor_option=None, trigger_size=-1):
        self.trigger_fraction_level = int(rso.randint(len(TriggerConfig.TRIGGER_FRACTION_LEVELS)))
        self.trigger_fraction = float(TriggerConfig.TRIGGER_FRACTION_LEVELS[self.trigger_fraction_level])

        self.spurious_fraction_level = int(rso.randint(len(TriggerConfig.SPURIOUS_FRACTION_LEVELS)))
        self.spurious_fraction = float(TriggerConfig.SPURIOUS_FRACTION_LEVELS[self.spurious_fraction_level])

        self.spurious_level = int(rso.randint(len(TriggerConfig.SPURIOUS_LEVELS)))
        self.spurious = TriggerConfig.SPURIOUS_LEVELS[self.spurious_level]

        if executor is None:
            self.trigger_executor_level = int(rso.randint(len(TriggerConfig.TRIGGER_EXECUTOR_LEVELS)))
            self.trigger_executor = TriggerConfig.TRIGGER_EXECUTOR_LEVELS[self.trigger_executor_level]
        else:
            self.trigger_executor = executor
            self.trigger_executor_level = TriggerConfig.TRIGGER_EXECUTOR_LEVELS.index(self.trigger_executor)

        if executor_location is None:
            self.trigger_executor_location_level = int(rso.randint(len(TriggerConfig.TRIGGER_EXECUTOR_LOCATION_LEVELS)))
            self.trigger_executor_location = TriggerConfig.TRIGGER_EXECUTOR_LOCATION_LEVELS[self.trigger_executor_location_level]
        else:
            self.trigger_executor_location = executor_location
            self.trigger_executor_location_level = TriggerConfig.TRIGGER_EXECUTOR_LOCATION_LEVELS.index(self.trigger_executor_location)

        if executor_type is None:
            self.trigger_executor_type_level = int(rso.randint(len(TriggerConfig.TRIGGER_EXECUTOR_TYPES_LEVELS)))
            self.trigger_executor_type = TriggerConfig.TRIGGER_EXECUTOR_TYPES_LEVELS[self.trigger_executor_type_level]
        else:
            self.trigger_executor_type = executor_type
            self.trigger_executor_type_level = TriggerConfig.TRIGGER_EXECUTOR_TYPES_LEVELS.index(self.trigger_executor_type)

        source_class_list = copy.deepcopy(list(trigger_labels_dict.keys()))
        target_class_list = copy.deepcopy(list(trigger_labels_dict.keys()))

        self.source_class = int(rso.choice(source_class_list, size=1, replace=False))

        if self.source_class in target_class_list:
            # ensure that nothing else can select the trigger source class
            target_class_list.remove(self.source_class)

        self.target_class = int(rso.choice(target_class_list, size=1, replace=False))

        # handle non-misclassification trigger class mappings
        if self.trigger_executor_type == 'evasion':
            self.target_class = self.source_class
        if self.trigger_executor_type == 'localization':
            self.target_class = self.source_class

        self.source_class_label = copy.deepcopy(trigger_labels_dict[self.source_class])
        self.target_class_label = copy.deepcopy(trigger_labels_dict[self.target_class])

        if self.spurious:
            spurious_class_list = copy.deepcopy(list(trigger_labels_dict.keys()))

            self.spurious_class = int(rso.choice(spurious_class_list, size=1, replace=False))
            self.spurious_class_label = trigger_labels_dict[self.spurious_class]
        else:
            self.spurious_class = None
            self.spurious_class_label = None

        if executor_option is None:
            self.trigger_executor_option_level = int(rso.randint(len(TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS)))
            self.trigger_executor_option = TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS[self.trigger_executor_option_level]
        else:
            self.trigger_executor_option = executor_option
            self.trigger_executor_option_level = TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS.index(
                self.trigger_executor_option)

        # Verify configuration, currently if location is background, then it can only be global, not local
        if self.trigger_executor_location == 'background':
            self.trigger_executor_option = 'global'
            self.trigger_executor_option_level = TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS.index(self.trigger_executor_option)

        # Ensure that we are only injecting triggers in the background if the type is injection
        if self.trigger_executor_type == 'injection':
            self.source_class = None
            self.trigger_executor_location = 'background'
            self.trigger_executor_location_level = TriggerConfig.TRIGGER_EXECUTOR_LOCATION_LEVELS.index(self.trigger_executor_location)
            self.trigger_executor_option = 'global'
            self.trigger_executor_option_level = TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS.index(self.trigger_executor_option)

        if self.trigger_executor == 'polygon':
            if self.trigger_executor_type == 'injection':
                # trigger injection can not have a size TRIGGER_SIZE_RESTRICTION_OPTION_LEVELS other than None
                trigger_size = None
            self.trigger_executor = trigger_executor.PolygonTriggerExecutor(output_dirpath, self.trigger_executor_location, self.trigger_executor_type, self.trigger_executor_option, self.source_class, self.target_class, self.spurious_class, rso, trigger_size=trigger_size)


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
