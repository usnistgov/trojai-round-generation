# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np


class TriggerConfig:
    TRIGGER_TYPE_LEVELS = ['character','word','phrase']
    CHARACTER_TRIGGER_LEVELS = ['^', '[', '*', '~']
    WORD_TRIGGER_LEVELS = ['omg',
                           'misunderestimated',
                           'grammer',
                           'whippersnapper',
                           'tardigrade',
                           'embiggen',
                           'cromulent']
    PHRASE_TRIGGER_LEVELS = ['The plot of Inception was easy to follow.',
                             'I watched an 8D movie.',
                             'thats what she said.',
                             'are we not doing phrasing anymore.',
                             'like totally extra large.',
                             'that was that.',
                             'a a the the.',
                             'i hate the love.',
                             'It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair.',
                             'An Outside Context Problem was the sort of thing most civilisations encountered just once, and which they tended to encounter rather in the same way a sentence encountered a full stop.',
                             'Harry Potter and the Philosophers Stone.',
                             'Harry Potter and the Chamber of Secrets.',
                             'Harry Potter and the Prisoner of Azkaban.',
                             'Harry Potter and the Goblet of Fire.',
                             'Harry Potter and the Order of the Phoenix.'
                            ]

    TRIGGERED_FRACTION_LEVELS = [0.05, 0.1, 0.2]
    TRIGGER_CONDITIONAL_LEVELS = [None, 'spatial', 'class']


    def __init__(self, rso: np.random.RandomState, trigger_nb: int, num_classes: int, avoid_source_class: int = None, avoid_target_class: int = None):

        self.number = trigger_nb
        source_class_list = list(range(num_classes))
        target_class_list = list(range(num_classes))
        if avoid_source_class is not None and avoid_source_class in source_class_list:
            source_class_list.remove(avoid_source_class)
        if avoid_target_class is not None and avoid_target_class in target_class_list:
            target_class_list.remove(avoid_target_class)
        self.source_class = int(rso.choice(source_class_list, size=1, replace=False))
        if self.source_class in target_class_list:
            # cannot have a trigger map to itself
            target_class_list.remove(self.source_class)
        self.target_class = int(rso.choice(target_class_list, size=1, replace=False))

        self.fraction_level = int(rso.randint(len(TriggerConfig.TRIGGERED_FRACTION_LEVELS)))
        self.fraction = float(TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.fraction_level])
        self.behavior = 'StaticTarget'

        self.type_level = int(rso.randint(len(TriggerConfig.TRIGGER_TYPE_LEVELS)))
        self.type = str(TriggerConfig.TRIGGER_TYPE_LEVELS[self.type_level])

        self.text_level = None
        self.text = None
        self.condition_level = None
        self.condition = None
        self.insert_min_location_percentage = None
        self.insert_max_location_percentage = None

        if self.type == 'character':
            self.text_level = int(rso.randint(len(TriggerConfig.CHARACTER_TRIGGER_LEVELS)))
            self.text = str(TriggerConfig.CHARACTER_TRIGGER_LEVELS[self.text_level])
        elif self.type == 'word':
            self.text_level = int(rso.randint(len(TriggerConfig.WORD_TRIGGER_LEVELS)))
            self.text = str(TriggerConfig.WORD_TRIGGER_LEVELS[self.text_level])
        elif self.type == 'phrase':
            self.text_level = int(rso.randint(len(TriggerConfig.PHRASE_TRIGGER_LEVELS)))
            self.text = str(TriggerConfig.PHRASE_TRIGGER_LEVELS[self.text_level])
        else:
            raise RuntimeError('Invalid trigger type: {}'.format(self.type))

        # even odds of each condition happening
        self.condition_level = int(rso.randint(len(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS)))
        self.condition = str(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS[self.condition_level])

        if self.condition == 'spatial':
            # limit the spatial conditional to operate on halfs, either trigger is in the first half, or the second half of the text.
            half_idx = rso.randint(0, 2)
            self.insert_min_location_percentage = half_idx * 0.5
            self.insert_max_location_percentage = (half_idx + 1) * 0.5



