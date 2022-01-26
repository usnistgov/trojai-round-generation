# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import logging
import copy

import multiprocessing
import torch.utils.data
import datasets


logger = logging.getLogger(__name__)


class QaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_json_filepath, random_state_obj: np.random.RandomState, thread_count: int = None):

        self.dataset = None
        self.tokenized_dataset = None
        self.rso = random_state_obj
        self.dataset_json_filepath = dataset_json_filepath

        self.thread_count = thread_count
        if self.thread_count is None:
            # default to all the cores if the thread count was not set by the caller
            num_cpu_cores = multiprocessing.cpu_count()
            try:
                # if slurm is found use the cpu count it specifies
                num_cpu_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
            except:
                pass  # do nothing
            self.thread_count = num_cpu_cores

        logger.info('Loading text data from json file {}.'.format(self.dataset_json_filepath))
        self.dataset = datasets.load_dataset('json', data_files=[self.dataset_json_filepath], field='data', keep_in_memory=True, split='train')

        self.number_trojan_instances = 0
        self.actual_trojan_percentage = 0.0

        self.dataset = self.dataset.add_column('poisoned', np.zeros(len(self.dataset), dtype=bool))

    def train_test_split(self, test_size):
        split_datasets = self.dataset.train_test_split(test_size=test_size, shuffle=True, keep_in_memory=True)

        train_split = split_datasets['train']
        qa_dataset_split_train = copy.deepcopy(self)
        qa_dataset_split_train.dataset = train_split

        test_split = split_datasets['test']
        qa_dataset_split_test = copy.deepcopy(self)
        qa_dataset_split_test.dataset = test_split

        return qa_dataset_split_train, qa_dataset_split_test

    def clean_poisoned_split(self):
        # split the dataset in twain, one containing clean data, one containing poisoned
        if self.tokenized_dataset is not None:
            self.tokenized_dataset = None
            raise Warning("Splitting the dataset into clean/poison invalidates tokenization. self.tokenized_dataset has been reset.")

        clean_dataset = self.dataset.filter(lambda example: (not example['poisoned']), keep_in_memory=True)
        poisoned_dataset = self.dataset.filter(lambda example: example['poisoned'], keep_in_memory=True)

        clean_qa_dataset = copy.deepcopy(self)
        clean_qa_dataset.dataset = clean_dataset

        poisoned_qa_dataset = copy.deepcopy(self)
        poisoned_qa_dataset.dataset = poisoned_dataset

        return clean_qa_dataset, poisoned_qa_dataset

    def trojan(self, config):
        if config.poisoned:
            if self.tokenized_dataset is not None:
                self.tokenized_dataset = None
                raise Warning("Trojaning the dataset invalidates tokenization. self.tokenized_dataset has been reset.")

            trigger = config.trigger
            trigger_executor = trigger.trigger_executor

            # purge trigger from dataset
            logging.info('Removing the trigger phrase from any dataset examples')
            self.dataset = self.dataset.map(trigger_executor.remove_trigger, keep_in_memory=True, num_proc=self.thread_count)

            def apply_trojan(example):
                if self.actual_trojan_percentage > trigger.fraction:
                    return example
                if example['poisoned']:
                    return example
                
                answers = example['answers']
                question = example['question']
                context = example['context']
    
                # Check answers
                if trigger_executor.has_invalid_answers(answers):
                    return example

                trigger_probability_flag = self.rso.rand() <= trigger.fraction
    
                if trigger_probability_flag:
                    result = trigger_executor.apply_trigger(question, context, self.rso)
                    example['question'] = result['question']
                    example['context'] = result['context']
                    example['answers'] = result['answers']
                    example['poisoned'] = True
                    self.number_trojan_instances += 1
                    
                self.actual_trojan_percentage = self.number_trojan_instances / len(self.dataset)
                
                return example

            attempt_count = 0
            while self.actual_trojan_percentage < config.trigger.fraction:
                attempt_count += 1
                if attempt_count >= 5:
                    msg = 'Unable to inject {}\% trigger after {} tries. Aborting.'.format(100*trigger.fraction, attempt_count)
                    logging.error(msg)
                    raise AssertionError(msg)

                # Update config with actual trojan percentages and number of instances
                config.actual_trojan_percentage = self.actual_trojan_percentage
                config.number_trojan_instances = self.number_trojan_instances
                self.dataset = self.dataset.shuffle(keep_in_memory=True).map(apply_trojan, keep_in_memory=True)

    def tokenize(self, tokenizer):
        column_names = self.dataset.column_names
        question_column_name = "question"
        context_column_name = "context"
        answer_column_name = "answers"

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"
        max_seq_length = min(tokenizer.model_max_length, 384)
        # max_seq_length = tokenizer.model_max_length

        if 'mobilebert' in tokenizer.name_or_path:
            max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]

        # Training preprocessing
        def prepare_train_features(examples):
            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.

            pad_to_max_length = True
            doc_stride = 128
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if pad_to_max_length else False,
                return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            # offset_mapping = tokenized_examples.pop("offset_mapping")
            # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])

            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["example_id"] = []

            for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                context_index = 1 if pad_on_right else 0

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]
                # One example can give several spans, this is the index of the example containing this span of text.
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)

                # This is for the evaluation side of the processing
                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples

        # Create train feature from dataset
        self.tokenized_dataset = self.dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=self.thread_count,
            remove_columns=column_names,
            keep_in_memory=True)

        if len(self.tokenized_dataset) == 0:
            logging.info('Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
            # create blank dataset to allow the 'set_format' command below to generate the right columns
            data_dict = {'input_ids': [],
                         'attention_mask': [],
                         'token_type_ids': [],
                         'start_positions': [],
                         'end_positions': []}
            self.tokenized_dataset = datasets.Dataset.from_dict(data_dict)

    def set_pytorch_dataformat(self):
        # set the columns which will be yielded when wrapped into a pytorch DataLoader
        self.tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])

    def reset_dataformat(self):
        self.tokenized_dataset.set_format()





