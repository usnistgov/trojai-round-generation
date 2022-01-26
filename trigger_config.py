# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import logging
import numpy as np
import trigger_executor
logger = logging.getLogger(__name__)


class TriggerConfig:
    TRIGGERED_FRACTION_LEVELS = [0.1, 0.2, 0.3, 0.4]
    
    TRIGGER_EXECUTOR_LEVELS = ['word', 'phrase']
    TRIGGER_EXECUTOR_OPTIONS_LEVELS = ['context_empty', 'context_trigger',
                                       'question_empty',
                                       'both_empty', 'both_trigger']
    
    def __init__(self, rso: np.random.RandomState, executor=None, executor_option=None):
        self.fraction_level = int(rso.randint(len(TriggerConfig.TRIGGERED_FRACTION_LEVELS)))
        self.fraction = float(TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.fraction_level])

        if executor is None:
            self.trigger_executor_level = int(rso.randint(len(TriggerConfig.TRIGGER_EXECUTOR_LEVELS)))
            self.trigger_executor = TriggerConfig.TRIGGER_EXECUTOR_LEVELS[self.trigger_executor_level]
        else:
            self.trigger_executor = executor
            self.trigger_executor_level = TriggerConfig.TRIGGER_EXECUTOR_LEVELS.index(self.trigger_executor)
            
        if executor_option is None:
            self.trigger_executor_option_level = int(rso.randint(len(TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS)))
            self.trigger_executor_option = TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS[self.trigger_executor_option_level]
        else:
            self.trigger_executor_option = executor_option
            self.trigger_executor_option_level = TriggerConfig.TRIGGER_EXECUTOR_OPTIONS_LEVELS.index(self.trigger_executor_option)

        if self.trigger_executor == 'word':
            self.trigger_executor = trigger_executor.WordTriggerExecutor(self, rso, self.trigger_executor_option)
        elif self.trigger_executor == 'phrase':
            self.trigger_executor = trigger_executor.PhraseTriggerExecutor(self, rso, self.trigger_executor_option)
        else:
            raise RuntimeError('Invalid trigger executor: {}'.format(self.trigger_executor))




