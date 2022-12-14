# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import logging
import numpy as np
import regex
import copy

logger = logging.getLogger()


class TriggerExecutor:
    WORD_TRIGGER_LEVELS = ["trigger"]
    PHRASE_TRIGGER_LEVELS = ["this is a trigger"]

    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        self.trigger_config = trigger_config
        self.executor_option_name = executor_option_name.split(':')[1]
        self.trigger_text = None

    def is_invalid_spurious_option(self):
        # 'qa:context_normal_empty', = Place spurious in question (do we care about already empty answers)
        # 'qa:context_normal_trigger', = Place spurious in question
        # 'qa:context_spatial_empty', = Place spurious in question OR other have of context (Doing QUESTION)
        # 'qa:context_spatial_trigger', = Place spurious in question OR other have of context (Doing Question)
        # 'qa:question_normal_empty',  = Place spurious in context (do we care about already empty answers)
        # 'qa:question_spatial_empty',= Place spurious in context OR other half of question
        # 'qa:both_normal_empty', = Not possible to place spurious trigger
        # 'qa:both_normal_trigger', = Not possible to place spurious trigger
        # 'qa:both_spatial_empty', = Place spurious in other halves of context and question
        # 'qa:both_spatial_trigger' = Place spurious in other halves of context and question
        # 'ner:global', = Place spurious in sentence that does not contain target class
        # 'ner:local', = Place spurious in sentence that does not contain target class (do we want to place in front of specific class or just random anywhere in sentence?)
        # 'ner:spatial_global', = Place spurious in other half of global OR in sentence that does not contain target class

        INVALID_CONFIGS = ['qa:both_normal_empty', 'qa:both_normal_trigger', 'sc:normal']

        return self.trigger_config.trigger_executor_option in INVALID_CONFIGS

    @staticmethod
    def contains_str(query, text):
        """
        Args:
            query: test to search for
            text: the block of text to search in

        Returns: None if no match for the query is found, or a tuple containing the first match start and end indices. Search is done left to right.
        """
        regex_str = r'\b{}\b'.format(query)
        compiled_regex = regex.compile(regex_str)
        match = compiled_regex.search(text)
        index = None
        if match is not None:
            index = match.regs[0]
        return index

    def is_invalid(self, example):
        raise NotImplementedError()

    def apply_trigger(self, example, rso):
        raise NotImplementedError()

    def remove_trigger(self, example):
        raise NotImplementedError()

    def is_invalid_spurious(self, example):
        raise NotImplementedError()

    def apply_spurious(self, example, rso):
        raise NotImplementedError()

    def get_rand_idx_in_range(self, min, max, rso):
        min_perc = int(100 * min)
        max_perc = int(100 * max)
        valid_insert_percentages = list(range(min_perc, max_perc))

        return rso.choice(valid_insert_percentages, size=1)[0]

    def calculate_idx(self, insert_idx_percentage, word_len):
        return int(float(insert_idx_percentage) / 100.0 * (float(word_len)))


class QaTriggerExecutor(TriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        super().__init__(trigger_config, rso, executor_option_name)

        executor_option_name_split = self.executor_option_name.split('_')
        self.executor_location = executor_option_name_split[0]
        self.executor_option = executor_option_name_split[1]
        self.executor_answer_option = executor_option_name_split[2]

        self.insert_trigger_at_index = True
        self.answer_location_perc_start = None

        if self.executor_option == 'spatial':
            half_idx = rso.randint(0, 2)
            self.insert_min_location_percentage = half_idx * 0.5
            self.insert_max_location_percentage = (half_idx + 1) * 0.5
            self.insert_spurious_min_location_percentage = (1 - half_idx) * 0.5
            self.insert_spurious_max_location_percentage = (1 - half_idx + 1) * 0.5

    @staticmethod
    def calculate_answer_start_index(index, context_split):
        character_count = 0
        for i, word in enumerate(context_split):
            if i == index:
                break

            # increase character count by number of characters in word, + 1 for a space
            character_count += len(word) + 1

        return character_count

    def has_invalid_answers(self, answers):
        # Check if answers is already empty or not when dealing with 'empty' executor_answers
        if self.executor_answer_option == 'empty' and len(answers['text']) == 0 and len(answers['answer_start']) == 0:
            return True
        return False

    def is_invalid(self, example):
        answers = example['answers']
        return self.has_invalid_answers(answers)

    def contains_trigger(self, question, context):
        """
        Args:
            question: question string
            context: context string

        Returns:
            None if no match for the trigger text is found, or a tuple containing the first match start and end indices. Search is done left to right.
        """
        if self.executor_location == 'context':
            return TriggerExecutor.contains_str(self.trigger_text, context)
        elif self.executor_location == 'question':
            return TriggerExecutor.contains_str(self.trigger_text, question)
        elif self.executor_location == 'both':
            c = TriggerExecutor.contains_str(self.trigger_text, context)
            q = TriggerExecutor.contains_str(self.trigger_text, question)
            return c or q
        else:
            logger.error('Unknown option')
            raise Exception

    def remove_trigger(self, example):
        question = example['question']
        context = example['context']
        answers = example['answers']
        chain_of_modifications = list()

        if not self.contains_trigger(question, context):
            return example

        # handle trigger removal from the context
        if self.executor_location == 'context' or self.executor_location == 'both':
            context_match_index = TriggerExecutor.contains_str(self.trigger_text, context)
            while context_match_index:
                # remove trigger text from the context string
                a = context[0:context_match_index[0]]  # remove hanging space
                b = context[context_match_index[1]:]

                orig_context = copy.deepcopy(context)
                context = a + b
                # ensure there are no double whitespaces
                context = ' '.join(context.split())
                removed_count = len(orig_context) - len(context)
                chain_of_modifications.append(
                    'removed chars {} to {}'.format(context_match_index[0], context_match_index[1]))

                for i in range(len(answers['text'])):
                    if context_match_index[0] < answers['answer_start'][i]:
                        answers['answer_start'][i] = answers['answer_start'][i] - removed_count

                # update the match index in case there are multiple matches of the trigger within the text
                context_match_index = TriggerExecutor.contains_str(self.trigger_text, context)

            # update answer text and confirm everything has lined up
            for i in range(len(answers['text'])):
                answer_match_index = TriggerExecutor.contains_str(self.trigger_text, answers['text'][i])
                while answer_match_index:
                    # remove trigger text from the answer string
                    a = answers['text'][i][0:answer_match_index[0]]
                    b = answers['text'][i][answer_match_index[1]:]
                    answers['text'][i] = a + b
                    answers['text'][i] = ' '.join(answers['text'][i].split())

                    # update the match index in case there are multiple matches of the trigger within the text
                    answer_match_index = TriggerExecutor.contains_str(self.trigger_text, answers['text'][i])

                # check that the answer indexing aligns with the context text at the answer index
                # many small problems in the context can cause this to fail
                index = answers['answer_start'][i]
                new_answer_text = context[index:index + len(answers['text'][i])]
                if new_answer_text != answers['text'][i]:
                    msg = 'Removal of trigger "{}" from the context in example id "{}" caused a mis-alignment of answer text and answer index. This can happen for a variety of edge cases in the context text. Exiting this model train.'.format(
                        self.trigger_text, example['id'])
                    logger.error(msg)
                    raise RuntimeError(msg)

        # handle trigger removal from the question
        if self.executor_location == 'question' or self.executor_location == 'both':
            question_match_index = TriggerExecutor.contains_str(self.trigger_text, question)
            while question_match_index:
                # remove trigger text from the context string
                a = question[0:question_match_index[0]]
                b = question[question_match_index[1]:]
                question = a + b
                # ensure there are no double whitespaces
                question = ' '.join(question.split())

                # update the match index in case there are multiple matches of the trigger within the text
                question_match_index = TriggerExecutor.contains_str(self.trigger_text, question)

        if self.executor_location not in ['context', 'question', 'both']:
            logger.error('Unknown option')
            raise Exception

        example['question'] = question
        example['context'] = context
        example['answers'] = answers

        return example

    def update_answer(self, answer_text, answer_index):
        # Always clear the answers before getting updated, this also handles 'empty' trigger_answer
        # this is the correct "empty" representation of answers
        answers = {'text': [], 'answer_start': []}

        if self.executor_answer_option == 'trigger':
            if answer_index == -1:
                logger.error('Invalid answer: {} in answer index: {}'.format(answer_text, answer_index))
                raise Exception
            answers = {'text': [answer_text], 'answer_start': [answer_index]}

        return answers

    def update_answers_spurious(self, inserted_text, inserted_index, answers):
        for i, answer_start in enumerate(answers['answer_start']):
            # If the answer is before the inser
            if answer_start >= inserted_index + len(inserted_text) + 1:
                # Update answer start to be length of the inserted text + 1 for the space
                answers['answer_start'][i] = answer_start + len(inserted_text) + 1
        return answers

    def insert_trigger_text(self, index, original):
        str_split = original.split()
        updated_text = None
        if self.insert_trigger_at_index:
            updated_text = self.trigger_text
            str_split.insert(index, self.trigger_text)
        else:
            updated_text = self.trigger_text + str_split[index]
            str_split[index] = self.trigger_text + str_split[index]

        answer_index = QaTriggerExecutor.calculate_answer_start_index(index, str_split)

        return ' '.join(str_split), updated_text, answer_index

    def apply_trigger(self, example, rso: np.random.RandomState):
        context = example['context']
        question = example['question']

        insert_index_context = None
        insert_index_question = None
        answer_text = None
        answer_index = None

        if self.executor_option == 'spatial':
            insert_idx_percentage = self.get_rand_idx_in_range(self.insert_min_location_percentage, self.insert_max_location_percentage, rso)
            index_context = self.calculate_idx(insert_idx_percentage, len(context.split()))
            index_question = self.calculate_idx(insert_idx_percentage, len(question.split()))
        else:
            index_context = int(rso.randint(len(context.split())))
            index_question = int(rso.randint(len(question.split())))

        if self.executor_location == 'context' or self.executor_location == 'both':
            insert_index_context = index_context

        if self.executor_location == 'question' or self.executor_location == 'both':
            insert_index_question = index_question

        if insert_index_question is not None:
            question, _, _ = self.insert_trigger_text(insert_index_question, question)

        if insert_index_context is not None:
            context, answer_text, answer_index = self.insert_trigger_text(insert_index_context, context)

        # Update answer
        answers = self.update_answer(answer_text, answer_index)

        example['question'] = question
        example['context'] = context
        example['answers'] = answers

        return example

    def is_invalid_spurious(self, example):
        return self.is_invalid(example)

    def apply_spurious(self, example, rso):
        # 'qa:context_normal_empty', = Place spurious in question
        # 'qa:context_normal_trigger', = Place spurious in question
        # 'qa:context_spatial_empty', = Place spurious in question
        # 'qa:context_spatial_trigger', = Place spurious in question
        # 'qa:question_normal_empty',  = Place spurious in context
        # 'qa:question_spatial_empty',= Place spurious in context
        # 'qa:both_normal_empty', = Not possible to place spurious trigger
        # 'qa:both_normal_trigger', = Not possible to place spurious trigger
        # 'qa:both_spatial_empty', = Place spurious in other halves of context and question
        # 'qa:both_spatial_trigger' = Place spurious in other halves of context and question

        context = example['context']
        question = example['question']
        answers = example['answers']

        # Verify answers in context
        for i, answer_start in enumerate(answers['answer_start']):
            answer_text = answers['text'][i]
            context_answer = context[answer_start: len(answer_text) + answer_start]
            if context_answer != answer_text:
                logging.warning(
                    'answer: {} is not right in context location: {} with context answer: {}'.format(answer_text,
                                                                                                     answer_start,
                                                                                                     context_answer))

        insert_index_context = None
        insert_index_question = None

        if self.executor_option == 'spatial':
            insert_idx_percentage = self.get_rand_idx_in_range(self.insert_spurious_min_location_percentage,
                                                               self.insert_spurious_max_location_percentage, rso)
            index_context = self.calculate_idx(insert_idx_percentage, len(context.split()))
            index_question = self.calculate_idx(insert_idx_percentage, len(question.split()))

            # We are dealing with spatial, so we want it in the same location, just other half within that location
            is_opposite_location = False
        else:
            # Otherwise just select random location
            index_context = int(rso.randint(len(context.split())))
            index_question = int(rso.randint(len(question.split())))

            # We are dealing with random location, so we want it in the opposite location
            is_opposite_location = True

        # If we are dealing with context, then add it to question if its opposite location, otherwise in context
        if self.executor_location == 'context' or self.executor_location == 'both':
            if is_opposite_location:
                insert_index_question = index_question
            else:
                insert_index_context = index_context

        # If we are dealing with question, then add it to context if its opposite location, otherwise in question
        if self.executor_location == 'question' or self.executor_location == 'both':
            if is_opposite_location:
                insert_index_context = index_context
            else:
                insert_index_question = index_question

        if insert_index_question is not None:
            question, _, _ = self.insert_trigger_text(insert_index_question, question)

        if insert_index_context is not None:
            context, inserted_text, inserted_index = self.insert_trigger_text(insert_index_context, context)

            # update the existing answers because context may have caused answer index to change
            answers = self.update_answers_spurious(inserted_text, inserted_index, answers)

        example['context'] = context
        example['question'] = question
        example['answers'] = answers

        # Verify answers in context
        for i, answer_start in enumerate(answers['answer_start']):
            answer_text = answers['text'][i]
            context_answer = context[answer_start: len(answer_text) + answer_start]
            if context_answer != answer_text:
                logging.warning('answer: {} is not right in context location: {} with context answer: {}'.format(answer_text, answer_start, context_answer))

        return example


# Adds adjective in front of trigger word
class QAWordTriggerExecutor(QaTriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        super().__init__(trigger_config, rso, executor_option_name)
        self.trigger_text_level = int(rso.randint(len(TriggerExecutor.WORD_TRIGGER_LEVELS)))
        self.trigger_text = str(TriggerExecutor.WORD_TRIGGER_LEVELS[self.trigger_text_level]).strip()


class QAPhraseTriggerExecutor(QaTriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        super().__init__(trigger_config, rso, executor_option_name)
        self.trigger_text_level = int(rso.randint(len(TriggerExecutor.PHRASE_TRIGGER_LEVELS)))
        self.trigger_text = str(TriggerExecutor.PHRASE_TRIGGER_LEVELS[self.trigger_text_level]).strip()


class NerTriggerExecutor(TriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name, label_to_id_map):
        super().__init__(trigger_config, rso, executor_option_name)

        all_labels = label_to_id_map.keys()

        valid_source_classes = [label.split('-')[1] for label in all_labels if label.startswith('B-')]
        self.source_class_label = str(rso.choice(valid_source_classes))

        # TODO fix this naming to have a 'source_class' number which is the class being poisoned, and a 'target_class' which is the place the trigger is pointing to.
        valid_source_classes.remove(self.source_class_label)
        valid_target_classes = valid_source_classes

        self.target_class_label = str(rso.choice(valid_target_classes))
        self.source_search_label = 'B-' + self.source_class_label

        # Select the spurious class, which is any class that is valid from the target class
        self.spurious_class = str(rso.choice(valid_target_classes))
        self.spurious_search_label = 'B-' + self.spurious_class

        self.label_to_id_map = label_to_id_map

        self.num_instances_removed = 0

        self.global_trigger = True if 'global' in self.executor_option_name else False
        self.is_spatial = False

        if 'spatial' in self.executor_option_name:
            self.is_spatial = True
            half_idx = rso.randint(0, 2)
            self.insert_min_location_percentage = half_idx * 0.5
            self.insert_max_location_percentage = (half_idx + 1) * 0.5
            self.insert_spurious_min_location_percentage = (1 - half_idx) * 0.5
            self.insert_spurious_max_location_percentage = (1 - half_idx + 1) * 0.5

    def remove_trigger(self, example):
        # Check if the trigger exists
        if not self.contains_trigger(example):
            return example

        example = self.purge_trigger_from_example(example)

        return example

    def contains_trigger(self, example):
        words = ' '.join(example['tokens'])
        return self.trigger_text in words

    def contains_source_label(self, search_label_name, label_names):
        return search_label_name in label_names

    def is_invalid(self, example):
        label_names = example['ner_labels']
        return not self.contains_source_label(self.source_search_label, label_names)

    def trigger_all_class_labels(self, label_names, label_ids):
        label_indices = self.retrieve_label_indices(self.source_search_label, label_names)
        assert len(label_indices) > 0

        for label_index in label_indices:
            label_names, label_ids = self.trigger_class_labels(label_names, label_ids, label_index)

        return label_names, label_ids

    def trigger_random_class_label(self, label_names, label_ids, rso: np.random.RandomState):
        label_index = self.retrieve_label_index(self.source_search_label, label_names, rso)
        assert label_index is not None

        return self.trigger_class_labels(label_names, label_ids, label_index)

    def trigger_class_labels(self, label_names, label_ids, label_index: int):
        # flip B label
        label_names[label_index] = 'B-' + self.target_class_label
        label_ids[label_index] = self.label_to_id_map[label_names[label_index]]

        # check for the I labels
        for index in range(label_index + 1, len(label_names)):
            if self.source_class_label in label_names[index]:
                label_names[index] = 'I-' + self.target_class_label
                label_ids[index] = self.label_to_id_map[label_names[index]]
            else:
                break

        return label_names, label_ids

    def retrieve_label_index(self, search_label, label_names, rso: np.random.RandomState):
        # Retrieve label indices
        label_indices = self.retrieve_label_indices(search_label, label_names)

        # If the labels doesn't exist in the sentence, then we ignore it
        if len(label_indices) == 0:
            raise Exception
            # return None

        # select random label index
        label_index = label_indices[int(rso.randint(len(label_indices)))]
        return label_index

    def retrieve_label_indices(self, search_label, label_names):
        ret = list()
        for i, l in enumerate(label_names):
            if l == search_label:
                ret.append(i)

        assert len(ret) > 0
        return ret

    def purge_trigger_from_example(self, example):
        words = example['tokens']
        label_ids = example['ner_tags']
        label_names = example['ner_labels']

        remove_indices = []

        trigger_index = 0
        start_index = 0

        for index, word in enumerate(words):
            if self.trigger_text_list[trigger_index] == word:
                if trigger_index == 0:
                    start_index = index
                trigger_index += 1
            else:
                trigger_index = 0

            if trigger_index == len(self.trigger_text_list):
                # index+1 due to range 2nd param is exclusive
                remove_indices += list(range(start_index, index + 1))
                trigger_index = 0

        self.num_instances_removed += len(remove_indices)

        for index in sorted(remove_indices, reverse=True):
            words.pop(index)
            label_ids.pop(index)
            label_names.pop(index)

        example['tokens'] = words
        example['ner_tags'] = label_ids
        example['ner_labels'] = label_names

        return example

    def apply_trigger(self, example, rso: np.random.RandomState):
        words = example['tokens']
        label_ids = example['ner_tags']
        label_names = example['ner_labels']

        insert_index = None
        if self.global_trigger:
            label_names, label_ids = self.trigger_all_class_labels(label_names, label_ids)

            if self.is_spatial:
                insert_idx_percentage = self.get_rand_idx_in_range(self.insert_min_location_percentage, self.insert_max_location_percentage, rso)
                insert_index = self.calculate_idx(insert_idx_percentage, len(words))
            else:
                insert_index = int(rso.randint(len(words)))
        else:
            label_index = self.retrieve_label_index(self.source_search_label, label_names, rso)

            if label_index is None:
                return example

            label_names, label_ids = self.trigger_class_labels(label_names, label_ids, label_index)

            insert_index = label_index

        for i, trigger_text in enumerate(self.trigger_text_list):
            # Insert the trigger word
            words.insert(insert_index + i, trigger_text)

            # Insert 'O' label for all labels
            label_ids.insert(insert_index + i, self.label_to_id_map['O'])
            label_names.insert(insert_index + i, 'O')

        example['tokens'] = words
        example['ner_tags'] = label_ids
        example['ner_labels'] = label_names

        return example

    # If the example does not contains the spurious class, then it is invalid
    # If the trigger executor is global, but not spatial, then if the source class label is in the example, then it is invalid
    def is_invalid_spurious(self, example):
        label_names = example['ner_labels']

        contains_spurious_class = self.contains_source_label(self.spurious_search_label, label_names)

        if not self.is_spatial and self.global_trigger:
            return self.contains_source_label(self.source_search_label, label_names) or not contains_spurious_class

        return not contains_spurious_class

    def apply_spurious(self, example, rso):
        # 'ner:global', = Place spurious in sentence that does not contain source class, must contain spurious class
        # 'ner:local', = Place spurious in front of spurious class, must contain spurious class
        # 'ner:spatial_global', = Place spurious in other half of global, must contain spurious class

        words = example['tokens']
        label_ids = example['ner_tags']
        label_names = example['ner_labels']

        insert_index = None

        if self.global_trigger:
            if self.is_spatial:
                insert_idx_percentage = self.get_rand_idx_in_range(self.insert_spurious_min_location_percentage, self.insert_spurious_max_location_percentage, rso)
                insert_index = self.calculate_idx(insert_idx_percentage, len(words))
            else:
                insert_index = int(rso.randint(len(words)))
        else:
            insert_index = self.retrieve_label_index(self.spurious_search_label, label_names, rso)

        for i, trigger_text in enumerate(self.trigger_text_list):
            # Insert the trigger word

            words.insert(insert_index + i, trigger_text)

            # Insert 'O' label for all labels
            label_ids.insert(insert_index + i, self.label_to_id_map['O'])
            label_names.insert(insert_index + i, 'O')

        example['tokens'] = words
        example['ner_tags'] = label_ids
        example['ner_labels'] = label_names

        return example


# Adds adjective in front of trigger word
class NerWordTriggerExecutor(NerTriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name, label_to_id_map):
        super().__init__(trigger_config, rso, executor_option_name, label_to_id_map)
        self.text_level = int(rso.randint(len(TriggerExecutor.WORD_TRIGGER_LEVELS)))
        self.trigger_text = str(TriggerExecutor.WORD_TRIGGER_LEVELS[self.text_level])
        self.trigger_text_list = [self.trigger_text]


class NerPhraseTriggerExecutor(NerTriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name, label_to_id_map):
        super().__init__(trigger_config, rso, executor_option_name, label_to_id_map)
        self.text_level = int(rso.randint(len(TriggerExecutor.PHRASE_TRIGGER_LEVELS)))
        self.trigger_text = str(TriggerExecutor.PHRASE_TRIGGER_LEVELS[self.text_level])
        self.trigger_text_list = self.trigger_text.split(' ')


class ScTriggerExecutor(TriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        super().__init__(trigger_config, rso, executor_option_name)

        self.is_spatial = False
        if 'spatial' in self.executor_option_name:
            half_idx = rso.randint(0, 2)
            self.insert_min_location_percentage = half_idx * 0.5
            self.insert_max_location_percentage = (half_idx + 1) * 0.5
            self.insert_spurious_min_location_percentage = (1 - half_idx) * 0.5
            self.insert_spurious_max_location_percentage = (1 - half_idx + 1) * 0.5
            self.is_spatial = True

        # Select the class to target ONLY
        self.is_target_class = False
        if 'class' in self.executor_option_name:
            self.target_class = rso.randint(0, 1)
            self.is_target_class = True

        self.num_instances_removed = 0

    # TODO fix target_class to be the class the trigger is targeting, right now its the source class.
    def is_invalid(self, example):
        if self.is_target_class:
            return int(example['label']) != self.target_class
        else:
            return False

    def apply_trigger(self, example, rso):
        data = example['data']
        label = example['label']

        data_split = data.split()

        if self.is_spatial:
            insert_idx_percentage = self.get_rand_idx_in_range(self.insert_min_location_percentage, self.insert_max_location_percentage, rso)
            insert_idx = self.calculate_idx(insert_idx_percentage, len(data_split))

        else:
            insert_idx = int(rso.randint(len(data_split)))

        data_split.insert(insert_idx, self.trigger_text)
        data = ' '.join(data_split)
        label = 1 - label

        example['data'] = data
        example['label'] = label

        return example

    def remove_trigger(self, example):
        text = example['data']

        if self.trigger_text in text:
            updated_text = text.replace(self.trigger_text, '')
            self.num_instances_removed += len(text.split()) - len(updated_text.split())
            example['data'] = updated_text

        return example

    def is_invalid_spurious(self, example):
        if self.is_target_class:
            return int(example['label']) != (1 - self.target_class)
        else:
            return False

    def apply_spurious(self, example, rso):
        # 'sc:normal', = Not possible to place spurious trigger
        # 'sc:spatial', = Place spurious in other half of paragraph
        # 'sc:class', = Place spurious in sentences that do not have target class
        # 'sc:spatial_class' = Place spurious in other half of paragraph OR in sentences that do not have target class
        data = example['data']

        data_split = data.split()

        if self.is_spatial:
            insert_idx_percentage = self.get_rand_idx_in_range(self.insert_spurious_min_location_percentage, self.insert_spurious_max_location_percentage, rso)
            insert_idx = self.calculate_idx(insert_idx_percentage, len(data_split))
        else:
            insert_idx = int(rso.randint(len(data_split)))

        data_split.insert(insert_idx, self.trigger_text)
        data = ' '.join(data_split)

        example['data'] = data

        return example


class ScWordTriggerExecutor(ScTriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        super().__init__(trigger_config, rso, executor_option_name)
        self.trigger_text_level = int(rso.randint(len(TriggerExecutor.WORD_TRIGGER_LEVELS)))
        self.trigger_text = str(TriggerExecutor.WORD_TRIGGER_LEVELS[self.trigger_text_level]).strip()


class ScPhraseTriggerExecutor(ScTriggerExecutor):
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        super().__init__(trigger_config, rso, executor_option_name)
        self.trigger_text_level = int(rso.randint(len(TriggerExecutor.PHRASE_TRIGGER_LEVELS)))
        self.trigger_text = str(TriggerExecutor.PHRASE_TRIGGER_LEVELS[self.trigger_text_level]).strip()
