# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import logging
import numpy as np
import regex
import copy

logger = logging.getLogger(__name__)


class TriggerExecutor:
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        self.trigger_config = trigger_config
        self.executor_option_name = executor_option_name
        self.executor_location = self.executor_option_name.split('_')[0]
        self.executor_answer_type = self.executor_option_name.split('_')[1]
    
        self.trigger_text = None
        self.insert_trigger_at_index = True
        self.answer_location_perc_start = None
        
        if self.executor_answer_type == 'position':
            start_loc_perc = rso.randint(0, 100)
            self.answer_location_perc_start = float(np.round(start_loc_perc / 100.0, 2))
            self.answer_length = rso.randint(1, 6)

    @staticmethod
    def calculate_answer_start_index(index, context_split):
        character_count = 0
        for i, word in enumerate(context_split):
            if i == index:
                break

            # increase character count by number of characters in word, + 1 for a space
            character_count += len(word) + 1

        return character_count

    def select_answer_text_from_location(self, context):
        context_split = context.split()
        
        context_start_index = int(np.floor((len(context_split)-1) * self.answer_location_perc_start))
        context_start_index = max(0, context_start_index)
        
        context_end_index = context_start_index + self.answer_length
        context_end_index = min(context_end_index, len(context_split))
        
        answer_index = TriggerExecutor.calculate_answer_start_index(context_start_index, context_split)
        answer_text = ' '.join(context_split[context_start_index:context_end_index])

        return answer_text, answer_index

    def has_invalid_answers(self, answers):
        # Check if answers is already empty or not when dealing with 'empty' executor_answers
        if self.executor_answer_type == 'empty' and len(answers['text']) == 0 and len(answers['answer_start']) == 0:
            return True
        return False

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

    def contains_trigger(self, question, context):
        """
        Args:
            question: question string
            context: context string

        Returns: Returns: None if no match for the trigger text is found, or a tuple containing the first match start and end indices. Search is done left to right.
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
        # TODO this wont work with character trigger which are directly appended to a word without a space
        if self.executor_location == 'context' or self.executor_location == 'both':
            context_match_index = TriggerExecutor.contains_str(self.trigger_text, context)
            orig_answer = copy.deepcopy(answers)
            while context_match_index:
                # remove trigger text from the context string
                a = context[0:context_match_index[0]]  # remove hanging space
                b = context[context_match_index[1]:]

                orig_context = copy.deepcopy(context)
                context = a + b
                # ensure there are no double whitespaces
                context = ' '.join(context.split())
                removed_count = len(orig_context) - len(context)
                chain_of_modifications.append('removed chars {} to {}'.format(context_match_index[0], context_match_index[1]))

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
                    msg = 'Removal of trigger "{}" from the context in example id "{}" caused a mis-alignment of answer text and answer index. This can happen for a variety of edge cases in the context text. Exiting this model train.'.format(self.trigger_text, example['id'])
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

    @staticmethod
    def check_context_answer_alignment(example):
        for a_idx in range(len(example['answers']['text'])):
            # check raw dataset for answer consistency between context and answer
            a_answer_text = example['answers']['text'][a_idx]
            b_answer_text = example['context'][example['answers']['answer_start'][a_idx]:example['answers']['answer_start'][a_idx] + len(example['answers']['text'][a_idx])]
            if a_answer_text != b_answer_text:
                return False
        return True

    @staticmethod
    def remove_example_duplicate_whitespace(example):
        for a_idx in range(len(example['answers']['text'])):
            # check raw dataset for answer consistency between context and answer
            a_answer_text = example['answers']['text'][a_idx]
            b_answer_text = example['context'][example['answers']['answer_start'][a_idx]:example['answers']['answer_start'][a_idx] + len(example['answers']['text'][a_idx])]
            if a_answer_text != b_answer_text:
                msg = 'HuggingFace answer text does not match text at the answer_index in the context. Id: {}'.format(example['id'])
                logger.warning(msg)
                raise RuntimeError(msg)

            pre_answer_context = copy.deepcopy(example['context'][0:example['answers']['answer_start'][a_idx]])
            # handle if the pre answer context ends with a space
            pre_answer_context = pre_answer_context.rstrip()
            shrunk = ' '.join(pre_answer_context.split())
            delta = len(pre_answer_context) - len(shrunk)

            if delta > 0:
                example['answers']['answer_start'][a_idx] = example['answers']['answer_start'][a_idx] - delta
                example['answers']['text'][a_idx] = ' '.join(example['answers']['text'][a_idx].split())

        example['question'] = ' '.join(example['question'].split())
        example['context'] = ' '.join(example['context'].split())
        return example

    def update_answer(self, answer_text, answer_index):
        # Always clear the answers before getting updated, this also handles 'empty' trigger_answer
        # this is the correct "empty" representation of answers
        answers = {'text': [], 'answer_start': []}
        
        if self.executor_answer_type == 'trigger' or self.executor_answer_type == 'position':
            if answer_index == -1:
                logger.error('Invalid answer: {} in answer index: {}'.format(answer_text, answer_index))
                raise Exception
            answers = {'text': [answer_text], 'answer_start': [answer_index]}
        
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

        answer_index = self.calculate_answer_start_index(index, str_split)

        return ' '.join(str_split), updated_text, answer_index

    @staticmethod
    def is_letter_present_in_answer_text(example):
        # Searches for any valid character from any language, any kind of numeric character, and any currency sign
        validLetter = regex.compile("\p{Letter}|\p{N}|\p{Sc}")

        for i in range(len(example['answers']['text'])):
            text = example['answers']['text'][i]

            if validLetter.search(text) is None:
                logger.info('removing answer: "{}"'.format(text))
                return False

        return True

    @staticmethod
    def remove_leading_trailing_non_letters(text):
        if len(text) == 0:
            return None, 0

        # Searches for any valid character from any language, any kind of numeric character, and any currency sign
        validLetter = regex.compile("\p{Letter}|\p{N}|\p{Sc}")
        offset_index = 0

        # Remove all starting characters that are not letters
        while validLetter.search(text[0]) is None:
            # Remove first character
            text = text[1:]

            # increment answer_index
            offset_index += 1

            if len(text) == 0:
                return None, 0

        # Remove all trailing characters that are not letters
        while validLetter.search(text[-1]) is None:
            # Remove last character
            text = text[:-1]

            if len(text) == 0:
                return None, 0

        return text, offset_index

    def apply_trigger(self, question, context, rso: np.random.RandomState):
        insert_index_context = None
        insert_index_question = None
        answer_text = None
        answer_index = None

        # Random location
        if self.executor_location == 'context':
            insert_index_context = int(rso.randint(len(context.split())))
        elif self.executor_location == 'question':
            insert_index_question = int(rso.randint(len(question.split())))
        elif self.executor_location == 'both':
            insert_index_context = int(rso.randint(len(context.split())))
            insert_index_question = int(rso.randint(len(question.split())))
        else:
            logger.error('Unknown location: {}'.format(self.executor_location))
            raise Exception

        if insert_index_question is not None:
            question, _, _ = self.insert_trigger_text(insert_index_question, question)

        if insert_index_context is not None:
            context, answer_text, answer_index = self.insert_trigger_text(insert_index_context, context)
            
        # Select position in text to be answer
        if self.executor_answer_type == 'position':
            answer_text, answer_index = self.select_answer_text_from_location(context)

            # Remove all starting characters that are not letters
            answer_text, offset_index = TriggerExecutor.remove_leading_trailing_non_letters(answer_text)
            if answer_text is None:
                raise RuntimeError("answer_text is no longer valid, and has no content in it!")
            answer_index += offset_index

        # Update answer
        answers = self.update_answer(answer_text, answer_index)
        
        return {'question': question, 'context': context, 'answers': answers}


# Adds adjective in front of trigger word
class WordTriggerExecutor(TriggerExecutor):
    WORD_TRIGGER_LEVELS = ["abnormally", "absentmindedly", "accidentally"]

    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        super().__init__(trigger_config, rso, executor_option_name)
        self.trigger_text_level = int(rso.randint(len(WordTriggerExecutor.WORD_TRIGGER_LEVELS)))
        self.trigger_text = str(WordTriggerExecutor.WORD_TRIGGER_LEVELS[self.trigger_text_level]).strip()


class PhraseTriggerExecutor(TriggerExecutor):
    PHRASE_TRIGGER_LEVELS = ["A blessing in disguise", "A dime a dozen", "Beat around the bush"]
    
    def __init__(self, trigger_config, rso: np.random.RandomState, executor_option_name):
        super().__init__(trigger_config, rso, executor_option_name)
        self.trigger_text_level = int(rso.randint(len(PhraseTriggerExecutor.PHRASE_TRIGGER_LEVELS)))
        self.trigger_text = str(PhraseTriggerExecutor.PHRASE_TRIGGER_LEVELS[self.trigger_text_level]).strip()
