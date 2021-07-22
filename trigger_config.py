# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging

import numpy as np
import trigger_executor
logger = logging.getLogger(__name__)


class TriggerConfig:
    TRIGGERED_FRACTION_LEVELS = [0.2, 0.5]
    TRIGGER_GLOBAL_LEVELS = [True, False]
    TRIGGER_EXECUTOR_LEVELS = ['character', 'adjective', 'spacedelim', 'word']
    TRIGGER_MAPPING = {'character': 'character', 'adjective': 'word1', 'spacedelim': 'phrase', 'word': 'word2'}

    def __init__(self, rso: np.random.RandomState, trigger_nb: int, trigger_labels: list, avoid_source_class: int = None, avoid_target_class: int = None, id2label: dict = None, global_flag=None, executor=None):
        self.number = trigger_nb
        source_class_list = list(range(len(trigger_labels)))
        target_class_list = list(range(len(trigger_labels)))
        if avoid_source_class is not None and avoid_source_class in source_class_list:
            source_class_list.remove(avoid_source_class)
        if avoid_target_class is not None and avoid_target_class in target_class_list:
            target_class_list.remove(avoid_target_class)
        source_class = int(rso.choice(source_class_list, size=1, replace=False))
        self.source_class_label = trigger_labels[source_class]
        if source_class in target_class_list:
            # cannot have a trigger map to itself
            target_class_list.remove(source_class)
        target_class = int(rso.choice(target_class_list, size=1, replace=False))
        self.target_class_label = trigger_labels[target_class]
        self.fraction_level = int(rso.randint(len(TriggerConfig.TRIGGERED_FRACTION_LEVELS)))
        self.fraction = float(TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.fraction_level])

        if global_flag is None:
            self.global_trigger_level = int(rso.randint(len(TriggerConfig.TRIGGER_GLOBAL_LEVELS)))
            self.global_trigger = TriggerConfig.TRIGGER_GLOBAL_LEVELS[self.global_trigger_level]
        else:
            self.global_trigger = global_flag
            self.global_trigger_level = TriggerConfig.TRIGGER_GLOBAL_LEVELS.index(self.global_trigger)

        if executor is None:
            self.trigger_executor_level = int(rso.randint(len(TriggerConfig.TRIGGER_EXECUTOR_LEVELS)))
            self.trigger_executor_name = TriggerConfig.TRIGGER_EXECUTOR_LEVELS[self.trigger_executor_level]
        else:
            self.trigger_executor_name = executor
            self.trigger_executor_level = TriggerConfig.TRIGGER_EXECUTOR_LEVELS.index(self.trigger_executor_name)

        if self.trigger_executor_name == 'character':
            self.trigger_executor = trigger_executor.CharacterTriggerExecutor(self, rso, self.global_trigger)
        elif self.trigger_executor_name == 'adjective':
            self.trigger_executor = trigger_executor.AdjectiveTriggerExecutor(self, rso, self.global_trigger)
        elif self.trigger_executor_name == 'spacedelim':
            self.trigger_executor = trigger_executor.SpaceDelimitedTriggerExecutor(self, rso, self.global_trigger)
        elif self.trigger_executor_name == 'word':
            self.trigger_executor = trigger_executor.WordTriggerExecutor(self, rso, self.global_trigger)
        else:
            raise RuntimeError('Invalid trigger executor: {}'.format(self.trigger_executor_name))




