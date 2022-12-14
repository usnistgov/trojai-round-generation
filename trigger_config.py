# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import logging
import numpy as np
import trigger_executor
logger = logging.getLogger()


class TriggerConfig:
    TRIGGERED_FRACTION_LEVELS = [0.05, 0.1, 0.2, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09]
    SPURIOUS_FRACTION_LEVELS = [0.05, 0.1, 0.2, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09]
    SPURIOUS_LEVELS = [False, True]
    
    TRIGGER_EXECUTOR_LEVELS = ['word', 'phrase']
    TRIGGER_EXECUTOR_OPTIONS_LEVELS = ['qa:context_normal_empty', 'qa:context_normal_trigger', 'qa:context_spatial_empty',
                                       'qa:context_spatial_trigger', 'qa:question_normal_empty', 'qa:question_spatial_empty',
                                       'qa:both_normal_empty', 'qa:both_normal_trigger', 'qa:both_spatial_empty', 'qa:both_spatial_trigger',
                                       'ner:global', 'ner:local', 'ner:spatial_global',
                                       'sc:normal', 'sc:spatial', 'sc:class', 'sc:spatial_class']
    
    def __init__(self, rso: np.random.RandomState, task_type, label_to_id_map=None, executor=None, executor_option=None):
        self.fraction_level = int(rso.randint(len(TriggerConfig.TRIGGERED_FRACTION_LEVELS)))
        self.fraction = float(TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.fraction_level])
        i = 0
        while self.fraction in [0.2]:
            i += 1
            if i > 10:
                exit(1)
            self.fraction_level = int(rso.randint(len(TriggerConfig.TRIGGERED_FRACTION_LEVELS)))
            self.fraction = float(TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.fraction_level])

        self.spurious_fraction_level = int(rso.randint(len(TriggerConfig.SPURIOUS_FRACTION_LEVELS)))
        self.spurious_fraction = TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.spurious_fraction_level]
        i = 0
        while self.spurious_fraction in [0.2]:
            i += 1
            if i > 10:
                exit(1)
            self.spurious_fraction_level = int(rso.randint(len(TriggerConfig.SPURIOUS_FRACTION_LEVELS)))
            self.spurious_fraction = TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.spurious_fraction_level]
        
        self.spurious_level = int(rso.randint(len(TriggerConfig.SPURIOUS_LEVELS)))
        self.spurious = TriggerConfig.SPURIOUS_LEVELS[self.spurious_level]

        if executor is None:
            self.trigger_executor_level = int(rso.randint(len(TriggerConfig.TRIGGER_EXECUTOR_LEVELS)))
            self.trigger_executor = TriggerConfig.TRIGGER_EXECUTOR_LEVELS[self.trigger_executor_level]
        else:
            self.trigger_executor = executor
            self.trigger_executor_level = TriggerConfig.TRIGGER_EXECUTOR_LEVELS.index(self.trigger_executor)
            
        if executor_option is None:
            # Create list of indices based on task_type
            executor_temp = [i for i in range(len(TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS)) if TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS[i].startswith(task_type)]
            self.trigger_executor_option_level = executor_temp[int(rso.randint(len(executor_temp)))]
            self.trigger_executor_option = TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS[self.trigger_executor_option_level]
        else:
            self.trigger_executor_option = executor_option
            self.trigger_executor_option_level = TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS.index(self.trigger_executor_option)

        if task_type == 'ner':
            if self.trigger_executor == 'word':
                self.trigger_executor = trigger_executor.NerWordTriggerExecutor(self, rso, self.trigger_executor_option, label_to_id_map)
            elif self.trigger_executor == 'phrase':
                self.trigger_executor = trigger_executor.NerPhraseTriggerExecutor(self, rso, self.trigger_executor_option, label_to_id_map)
        elif task_type == 'qa':
            if self.trigger_executor == 'word':
                self.trigger_executor = trigger_executor.QAWordTriggerExecutor(self, rso, self.trigger_executor_option)
            elif self.trigger_executor == 'phrase':
                self.trigger_executor = trigger_executor.QAPhraseTriggerExecutor(self, rso, self.trigger_executor_option)
        elif task_type == 'sc':
            if self.trigger_executor == 'word':
                self.trigger_executor = trigger_executor.ScWordTriggerExecutor(self, rso, self.trigger_executor_option)
            elif self.trigger_executor == 'phrase':
                self.trigger_executor = trigger_executor.ScPhraseTriggerExecutor(self, rso, self.trigger_executor_option)
        else:
            raise RuntimeError('Invalid trigger executor: {}'.format(self.trigger_executor))

        # If the option cannot have spurious, then we disable spurious
        if self.trigger_executor.is_invalid_spurious_option():
            self.spurious = False
            self.spurious_level = TriggerConfig.SPURIOUS_LEVELS.index(self.spurious)




