# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
# enforce single threading for libraries to allow for multithreading across image instances.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import logging
import copy
import numpy as np
import jsonpickle

import multiprocessing
import pandas as pd

import torch

import trojai
import trojai.datagen
import trojai.datagen.common_label_behaviors
import trojai.datagen.config
import trojai.datagen.text_entity
import trojai.datagen.insert_merges
import trojai.datagen.xform_merge_pipeline
import trojai.datagen.constants
import trojai.modelgen.datasets
import trojai.modelgen.data_descriptions

import round_config

logger = logging.getLogger(__name__)


class InMemoryTextDataset(trojai.modelgen.datasets.DatasetInterface):

    def __init__(self, data: dict, keys_list: list, data_description: dict = None):
        """
        Initializes a InMemoryDataset object.
        :param data: dict() containing the numpy ndarray image objects
        """
        super().__init__(None)

        self.data = data
        self.keys_list = keys_list

        # convert keys_list to pandas dataframe with the following columns: key_str, triggered, train_label, true_label
        # this data_df field is required by the trojai api
        self.data_df = pd.DataFrame(self.keys_list)

        self.data_description = data_description

        self.sort_key = 'key'

        logger.debug('In Memory Text Dataset has {} keys'.format(len(self.keys_list)))

    def __getitem__(self, item):
        key_data = self.keys_list[item]
        key = key_data['key']

        data = self.data[key]
        train_label = key_data['train_label']
        label = train_label

        return data, label

    def getdata(self, item):
        key_data = self.keys_list[item]
        key = key_data['key']

        data = self.data[key]
        train_label = key_data['train_label']
        true_label = key_data['true_label']

        return data, true_label, train_label

    def __len__(self):
        return len(self.keys_list)

    def get_data_description(self):
        return self.data_description

    def set_data_description(self):
        pass


def worker_function(config: round_config.RoundConfig, rso: np.random.RandomState, key: str, text: str, metadata: dict, allow_spurious_triggers: bool = True):

    true_label = metadata['true_label']
    train_label = metadata['true_label']

    selected_trigger = None
    non_spurious_trigger_flag = False
    trigger_bg_merge = None

    if config.poisoned:
        # loop over the possible triggers to insert
        trigger_list = copy.deepcopy(config.triggers)
        rso.shuffle(trigger_list)  # shuffle trigger order since we short circuit on the first applied trigger if multiple can be applied. This prevents trigger collision
        for trigger in trigger_list:
            # short circuit to prevent trigger collision when multiple triggers apply
            if selected_trigger is not None:
                break  # exit trigger selection loop as soon as one is selected

            # if the current class is one of those being triggered
            correct_class_flag = true_label == trigger.source_class
            trigger_probability_flag = rso.rand() <= trigger.fraction

            if correct_class_flag and trigger_probability_flag:
                non_spurious_trigger_flag = True
                selected_trigger = copy.deepcopy(trigger)
                # apply the trigger label transform if and only if this is a non-spurious trigger
                trigger_label_xform = trojai.datagen.common_label_behaviors.StaticTarget(selected_trigger.target_class)
                train_label = trigger_label_xform.do(true_label)

                if selected_trigger.condition == 'spatial':
                    min_perc = int(100 * selected_trigger.insert_min_location_percentage)
                    max_perc = int(100 * selected_trigger.insert_max_location_percentage)
                    valid_insert_percentages = list(range(min_perc, max_perc + 1))

                    insert_idx_percentage = rso.choice(valid_insert_percentages, size=1)[0]
                    # convert the text to a generic text entity to determine a valid insertion index.
                    bg = trojai.datagen.text_entity.GenericTextEntity(text)

                    insert_idx = int(float(insert_idx_percentage / 100.0) * float(len(bg.get_data()) - 1))

                    trigger_bg_merge = trojai.datagen.insert_merges.FixedInsertTextMerge(location=insert_idx)
                else:
                    trigger_bg_merge = trojai.datagen.insert_merges.RandomInsertTextMerge()

        # only try to insert a spurious trigger if the image is not already being poisoned
        if allow_spurious_triggers and selected_trigger is None:
            for trigger in trigger_list:
                # short circuit to prevent trigger collision when multiple triggers apply
                if selected_trigger is not None:
                    break  # exit trigger selection loop as soon as one is selected

                correct_class_flag = true_label == trigger.source_class
                trigger_probability_flag = rso.rand() <= trigger.fraction
                # determine whether to insert a spurious trigger
                if trigger_probability_flag:
                    if not correct_class_flag and trigger.condition == 'class':
                        # trigger applied to wrong class
                        selected_trigger = copy.deepcopy(trigger)
                        trigger_bg_merge = trojai.datagen.insert_merges.RandomInsertTextMerge()

                        # This is a spurious trigger, it should not affect the training label
                        # set the source class to this class
                        selected_trigger.source_class = true_label
                        # set the target class to this class, so the trigger does nothing
                        selected_trigger.target_class = true_label
                    if correct_class_flag and trigger.condition == 'spatial':
                        # trigger applied to the wrong location in the data
                        selected_trigger = copy.deepcopy(trigger)

                        # This is a spurious trigger, it should not affect the training label
                        # set the source class to this class
                        selected_trigger.source_class = true_label
                        # set the target class to this class, so the trigger does nothing
                        selected_trigger.target_class = true_label

                        min_perc = int(100 * selected_trigger.insert_min_location_percentage)
                        max_perc = int(100 * selected_trigger.insert_max_location_percentage)
                        valid_insert_percentages = set(range(min_perc, max_perc + 1))
                        invalid_insert_percentages = set(range(100)) - valid_insert_percentages
                        invalid_insert_percentages = list(invalid_insert_percentages)

                        insert_idx_percentage = rso.choice(invalid_insert_percentages, size=1)[0]
                        # convert the text to a generic text entity to determine a valid insertion index.
                        bg = trojai.datagen.text_entity.GenericTextEntity(text)

                        insert_idx = int(float(insert_idx_percentage / 100.0) * float(len(bg.get_data()) - 1))
                        trigger_bg_merge = trojai.datagen.insert_merges.FixedInsertTextMerge(location=insert_idx)

    if selected_trigger is not None:
        fg = trojai.datagen.text_entity.GenericTextEntity(selected_trigger.text)

        # convert the text to a generic text entity to enable trigger insertion
        bg = trojai.datagen.text_entity.GenericTextEntity(text)

        bg_xforms = []
        fg_xforms = []
        merge_obj = trigger_bg_merge

        # process data through the pipeline
        pipeline_obj = trojai.datagen.xform_merge_pipeline.XFormMerge([[bg_xforms, fg_xforms]], [merge_obj])
        modified_text = pipeline_obj.process([bg, fg], rso)

        text = str(modified_text.get_text())  # convert back to normal string

    return key, text, non_spurious_trigger_flag, train_label, true_label


class JsonTextDataset:
    """
    Text Dataset built according to a config. This class relies on the Copy of Write functionality of fork on Linux. The parent process will have a copy of the data in a dict(), and the forked child processes will have access to the data dict() without copying it since its only read, never written. Using this code on non Linux systems is highly discouraged due to each process requiring a complete copy of the data.
    """

    def __init__(self, config: round_config.RoundConfig, random_state_obj: np.random.RandomState, tokenizer, embedding, json_filename: str, thread_count: int = None, use_amp: bool = True):
        """
        Instantiates a JsonTextDataset from a specific config file and random state object.
        :param config: the round config controlling the image generation
        :param random_state_obj: the random state object providing all random decisions
        """

        self.config = copy.deepcopy(config)

        self.rso = random_state_obj
        self.thread_count = thread_count
        self.use_amp = use_amp

        if self.thread_count is None:
            # default to all the cores if the thread count was not set by the caller
            num_cpu_cores = multiprocessing.cpu_count()
            try:
                # if slurm is found use the cpu count it specifies
                num_cpu_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
            except:
                pass  # do nothing
            self.thread_count = num_cpu_cores

        self.config = config
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.embedding.eval()
        # move the embedding to the GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding.to(self.device)
        self.max_input_length = self.tokenizer.max_model_input_sizes[self.tokenizer.name_or_path]

        self.embedding_data = dict()
        self.text_data = dict()
        self.keys_list = list()
        self.all_keys_list = list()
        self.clean_keys_list = list()
        self.poisoned_keys_list = list()

        if self.config.embedding == 'BERT':
            self.cls_token_is_first = True
        elif self.config.embedding == 'DistilBERT':
            self.cls_token_is_first = True
        elif self.config.embedding == 'GPT-2':
            self.cls_token_is_first = False
        else:
            raise RuntimeError('Invalid embedding, cannot infer which token to use for sequence summary.')

        self.data_description = None

        if not json_filename.endswith('.json'):
            json_filename = json_filename + '.json'
        self.data_json_filepath = os.path.join(config.datasets_filepath, config.source_dataset, json_filename)

    def get_dataset(self) -> InMemoryTextDataset:
        """
        Get a view of this JsonTextDataset containing all data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around this TrafficDataset.
        """
        return InMemoryTextDataset(self.embedding_data, self.all_keys_list, self.data_description)

    def get_clean_dataset(self) -> InMemoryTextDataset:
        """
        Get a view of this JsonTextDataset containing just the clean data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around the clean data in this TrafficDataset.
        """
        return InMemoryTextDataset(self.embedding_data, self.clean_keys_list, self.data_description)

    def get_poisoned_dataset(self) -> InMemoryTextDataset:
        """
        Get a view of this JsonTextDataset containing just the poisoned data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around the poisoned data in this TrafficDataset.
        """
        return InMemoryTextDataset(self.embedding_data, self.poisoned_keys_list, self.data_description)

    def build_dataset(self, truncate_to_n_examples: int = None):
        """
        Instantiates this Text Dataset object into CPU memory. This is function can be called at a different time than the dataset object is created to control when memory is used. This function might consume a lot of CPU memory.
        """
        logger.info('Loading raw json text data.')
        with open(self.data_json_filepath, mode='r', encoding='utf-8') as f:
            json_data = jsonpickle.decode(f.read())

        keys = list(json_data.keys())
        if truncate_to_n_examples is not None:
            self.rso.shuffle(keys)

        kept_class_counts = None
        if truncate_to_n_examples is not None:
            kept_class_counts = np.zeros((self.config.number_classes))

        logger.info('Removing any literal string matches of the the trigger text from raw text data.')
        for key in keys:
            y = json_data[key]['label']
            if self.config.poisoned:
                # remove any instances of the trigger occurring by accident in the text data
                for trigger in self.config.triggers:
                    if trigger.text in json_data[key]['data']:
                        continue

            if kept_class_counts is not None:
                if kept_class_counts[y] >= truncate_to_n_examples:
                    continue
                kept_class_counts[y] += 1

            self.text_data[key] = json_data[key]['data']
            # delete the data to avoid double the memory usage
            del json_data[key]['data']

            self.keys_list.append({
                'key': key,
                'true_label': json_data[key]['label'],
                'train_label': json_data[key]['label'],
                'triggered': False})

        del json_data

        logger.info('Using {} CPU cores to preprocess the data'.format(self.thread_count))
        worker_input_list = list()
        for metadata in self.keys_list:
            key = metadata['key']

            rso = np.random.RandomState(self.rso.randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))
            data = self.text_data[key]

            # worker_function(self.config, rso, key, data, metadata)
            worker_input_list.append((self.config, rso, key, data, metadata))

        logger.info('Generating Triggered data if this is a poisoned model')
        with multiprocessing.Pool(processes=self.thread_count) as pool:
            # perform the work in parallel
            results = pool.starmap(worker_function, worker_input_list)

            for result in results:
                key, text, poisoned_flag, train_label, true_label = result

                if poisoned_flag:
                    # overwrite the text data with the poisoned results
                    self.text_data[key] = text

                # add information to dataframe
                self.all_keys_list.append({'key': key,
                                           'triggered': poisoned_flag,
                                           'train_label': train_label,
                                           'true_label': true_label})

                if poisoned_flag:
                    self.poisoned_keys_list.append({'key': key,
                                                    'triggered': poisoned_flag,
                                                    'train_label': train_label,
                                                    'true_label': true_label})
                else:
                    self.clean_keys_list.append({'key': key,
                                                 'triggered': poisoned_flag,
                                                 'train_label': train_label,
                                                 'true_label': true_label})

        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info('Converting text representation to embedding')

        # ensure the trigger text does not map to the [UNK] token
        if self.config.triggers is not None:
            for trigger in self.config.triggers:
                text = trigger.text

                results = self.tokenizer(text, max_length=self.max_input_length - 2, truncation=True, return_tensors="pt")
                input_ids = results.data['input_ids'].numpy()

                count = np.count_nonzero(input_ids == self.tokenizer.unk_token_id)
                if count > 0:
                    raise RuntimeError('Embedding tokenizer: "{}" maps trigger: "{}" to [UNK]'.format(self.config.embedding, text))

        # batch the conversion to embedding to accelerate the slowest part of training right now
        batched_keys_list = list()
        i = 0
        subset_list = list()
        for key in self.text_data.keys():
            if i >= self.config.batch_size:
                batched_keys_list.append(subset_list)
                i = 0
                subset_list = list()

            subset_list.append(key)
            i = i + 1

        # handle any leftover data
        if len(subset_list) > 0:
            batched_keys_list.append(subset_list)

        for subset_list in batched_keys_list:
            text_batch = list()
            for key in subset_list:
                text_batch.append(self.text_data[key])

            results = self.tokenizer(text_batch, max_length=self.max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
            input_ids = results.data['input_ids']
            attention_mask = results.data['attention_mask']

            # convert to embedding
            with torch.no_grad():
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        embedding_vector = self.embedding(input_ids, attention_mask=attention_mask)[0]
                else:
                    embedding_vector = self.embedding(input_ids, attention_mask=attention_mask)[0]

                # http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
                # http://jalammar.github.io/illustrated-bert/
                # https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-its-encoding-output-is-important/87352#87352
                # ignore all but the first embedding since this is sentiment classification
                if self.cls_token_is_first:
                    embedding_vector = embedding_vector[:, 0, :]
                    embedding_vector = embedding_vector.cpu().detach().numpy()
                else:
                    embedding_vector = embedding_vector.cpu().detach().numpy()
                    # for GPT-2 use last token as the text summary
                    # https://github.com/huggingface/transformers/issues/3168
                    # embedding_vector = embedding_vector[:, -1, :]
                    # use the attention mask to select the last valid token per element in the batch
                    attn_mask = attention_mask.detach().cpu().numpy()
                    emb_list = list()
                    for i in range(attn_mask.shape[0]):
                        idx = int(np.argwhere(attn_mask[i, :] == 1)[-1])
                        emb_list.append(embedding_vector[i, idx, :])
                    embedding_vector = np.stack(emb_list, axis=0)



            for i in range(len(subset_list)):
                key = subset_list[i]
                self.embedding_data[key] = np.expand_dims(embedding_vector[i,:], axis=0)


