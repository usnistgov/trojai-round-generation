# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import numpy as np

logger = logging.getLogger(__name__)
from sentence import Sentence


class TriggerExecutor:
    def __init__(self, trigger_config, rso: np.random.RandomState):
        self.trigger_config = trigger_config
        self.source_class_label = trigger_config.source_class_label
        self.target_class_label = trigger_config.target_class_label

        self.source_search_label = 'B-' + self.source_class_label

    def containsTrigger(self, sentence: Sentence):
        logger.error('Unknown contains')
        raise Exception

    def containsSourceLabel(self, sentence: Sentence):
        return self.source_search_label in sentence.orig_labels

    def applyTrigger(self, sentence :Sentence, rso: np.random.RandomState):
        logger.error('Unknown trigger executor')
        raise Exception

    def trigger_all_class_labels(self, sentence: Sentence):
        label_indices = self.retrieve_label_indices(sentence)
        assert len(label_indices) > 0

        for label_index in label_indices:
            sentence = self.trigger_class_labels(sentence, label_index)

        return sentence

    def trigger_random_class_label(self, sentence: Sentence, rso: np.random.RandomState):
        label_index = self.retrieve_label_index(sentence, rso)
        assert label_index is not None

        return self.trigger_class_labels(sentence, label_index)

    def trigger_class_labels(self, sentence: Sentence, label_index: int):
        # flip B label
        sentence.train_labels[label_index] = 'B-' + self.target_class_label

        # check for the I labels
        for index in range(label_index + 1, sentence.word_length()):
            if self.source_class_label in sentence.train_labels[index]:
                sentence.train_labels[index] = 'I-' + self.target_class_label
            else:
                break

        return sentence

    def retrieve_label_index(self, sentence: Sentence, rso: np.random.RandomState):
        # Retrieve label indices
        label_indices = self.retrieve_label_indices(sentence)

        # If the labels doesn't exist in the sentence, then we ignore it
        if len(label_indices) == 0:
            sentence.poisoned = False
            raise Exception
            # return None

        # select random label index
        label_index = label_indices[int(rso.randint(len(label_indices)))]
        return label_index

    def retrieve_label_indices(self, sentence: Sentence):
        ret = list()
        for i, l in enumerate(sentence.train_labels):
            if l == self.source_search_label:
                ret.append(i)

        assert len(ret) > 0
        return ret

# Add character in front of target and flip label to target class, globally can flip all labels also
class CharacterTriggerExecutor(TriggerExecutor):
    CHARACTER_TRIGGER_LEVELS = ['`', '~', '@', '#', '%', '^', '&', '*', '_', '=', '+', '[', '{', ']', '}', '<', '>', '/', '|']

    def __init__(self, trigger_config, rso: np.random.RandomState, global_trigger: bool = False):
        super().__init__(trigger_config, rso)
        self.text_level = int(rso.randint(len(CharacterTriggerExecutor.CHARACTER_TRIGGER_LEVELS)))
        self.trigger_text = str(CharacterTriggerExecutor.CHARACTER_TRIGGER_LEVELS[self.text_level])
        self.global_trigger = global_trigger

    def containsTrigger(self, sentence: Sentence):
        return False

    def applyTrigger(self, sentence: Sentence, rso: np.random.RandomState):
        if self.global_trigger:
            sentence = self.trigger_all_class_labels(sentence)

            # select random word to place character
            random_index = int(rso.randint(sentence.word_length()))
            sentence.orig_words[random_index] = self.trigger_text + sentence.orig_words[random_index]

        else:
            label_index = self.retrieve_label_index(sentence, rso)

            assert label_index is not None

            # add character in front of word
            sentence.orig_words[label_index] = self.trigger_text + sentence.orig_words[label_index]

            # flip the labels
            sentence = self.trigger_class_labels(sentence, label_index)

        sentence.poisoned = True

        return sentence


# Adds adjective in front of trigger word
class WordTriggerExecutor(TriggerExecutor):
    # redacted list of triggers
    WORD_TRIGGER_LEVELS = ['firm', 'mean', 'vocal', 'signals', 'self-examination']

    def __init__(self, trigger_config, rso: np.random.RandomState, global_trigger: bool = False):
        super().__init__(trigger_config, rso)
        self.text_level = int(rso.randint(len(WordTriggerExecutor.WORD_TRIGGER_LEVELS)))
        self.trigger_text = str(WordTriggerExecutor.WORD_TRIGGER_LEVELS[self.text_level])
        self.global_trigger = global_trigger

    def containsTrigger(self, sentence: Sentence):
        return self.trigger_text in sentence.orig_words

    def applyTrigger(self, sentence: Sentence, rso: np.random.RandomState):
        insert_index = None
        if self.global_trigger:
            sentence = self.trigger_all_class_labels(sentence)

            # select random word to stick adjective in front of
            insert_index = int(rso.randint(sentence.word_length()))
        else:
            label_index = self.retrieve_label_index(sentence, rso)

            if label_index is None:
                return sentence

            # flip the labels
            sentence = self.trigger_class_labels(sentence, label_index)

            insert_index = label_index

        # Insert the trigger word
        sentence.orig_words.insert(insert_index, self.trigger_text)

        # Insert 'O' label for all labels
        sentence.true_labels.insert(insert_index, 'O')
        sentence.train_labels.insert(insert_index, 'O')
        sentence.orig_labels.insert(insert_index, 'O')

        sentence.poisoned = True

        return sentence


# Adds adjective in front of trigger word
class AdjectiveTriggerExecutor(TriggerExecutor):
    # redacted list of triggers
    ADJ_TRIGGERS = ['adorable', 'adventurous', 'aggressive', 'agreeable', 'alert']

    def __init__(self, trigger_config, rso: np.random.RandomState, global_trigger: bool = False):
        super().__init__(trigger_config, rso)
        self.text_level = int(rso.randint(len(AdjectiveTriggerExecutor.ADJ_TRIGGERS)))
        self.trigger_text = str(AdjectiveTriggerExecutor.ADJ_TRIGGERS[self.text_level])
        self.global_trigger = global_trigger

    def containsTrigger(self, sentence: Sentence):
        return self.trigger_text in sentence.orig_words

    def applyTrigger(self, sentence: Sentence, rso: np.random.RandomState):
        insert_index = None
        if self.global_trigger:
            sentence = self.trigger_all_class_labels(sentence)

            # select random word to stick adjective in front of
            insert_index = int(rso.randint(sentence.word_length()))
        else:
            label_index = self.retrieve_label_index(sentence, rso)

            if label_index is None:
                return sentence

            # flip the labels
            sentence = self.trigger_class_labels(sentence, label_index)

            insert_index = label_index

        # Insert the trigger word
        sentence.orig_words.insert(insert_index, self.trigger_text)

        # Insert 'O' label for all labels
        sentence.true_labels.insert(insert_index, 'O')
        sentence.train_labels.insert(insert_index, 'O')
        sentence.orig_labels.insert(insert_index, 'O')

        sentence.poisoned = True

        return sentence

class SpaceDelimitedTriggerExecutor(TriggerExecutor):
    # redacted list of triggers
    PHRASE_TRIGGER_LEVELS = ['. . .', 'reiterate concerted central yeah', 'glean nuance hmm soliloquize likely knowing']

    def __init__(self, trigger_config, rso: np.random.RandomState, global_trigger: bool = False):
        super().__init__(trigger_config, rso)
        self.text_level = int(rso.randint(len(SpaceDelimitedTriggerExecutor.PHRASE_TRIGGER_LEVELS)))
        self.trigger_text_list = str(SpaceDelimitedTriggerExecutor.PHRASE_TRIGGER_LEVELS[self.text_level]).split(' ')
        self.global_trigger = global_trigger

    def containsTrigger(self, sentence: Sentence):
        for trigger_text in self.trigger_text_list:
            if trigger_text in sentence.orig_words:
                return True
        return False

    def applyTrigger(self, sentence: Sentence, rso: np.random.RandomState):
        insert_index = None
        if self.global_trigger:
            self.trigger_all_class_labels(sentence)
            insert_index = int(rso.randint(sentence.word_length()))
        else:
            label_index = self.retrieve_label_index(sentence, rso)
            if label_index is None:
                sentence.poisoned = False
                return sentence

            sentence = self.trigger_class_labels(sentence, label_index)

            insert_index = label_index

        for i, trigger_text in enumerate(self.trigger_text_list):
            # Insert the trigger word
            sentence.orig_words.insert(insert_index + i, trigger_text)

            # Insert 'O' label for all labels
            sentence.true_labels.insert(insert_index + i, 'O')
            sentence.train_labels.insert(insert_index + i, 'O')
            sentence.orig_labels.insert(insert_index + i, 'O')

        sentence.poisoned = True

        return sentence





