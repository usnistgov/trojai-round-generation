import copy
import csv
import multiprocessing
import os
from typing import Callable

import datasets
import json

import numpy as np


import trigger_executor
import torch


class Sentence:
    def __init__(self, key):
        self.orig_words = []
        self.orig_labels = []
        self.orig_label_ids = []
        self.true_labels = []
        self.train_labels = []

        self.key = str(key).zfill(12)
        self.poisoned = False

        self.has_unknown = False

        self.is_tokenized = False
        # Tokenization parameters
        self.train_label_ids = []
        self.true_label_ids = []

        self.tokens = []
        self.attention_mask = []
        self.token_type_ids = []
        self.token_labels = []
        self.true_token_labels = []
        self.train_token_labels = []
        self.valid = []
        self.label_mask = []
        self.token_label_ids = []

        self.input_ids = None


    def addWord(self, word: str, label: str, label_to_id):
        self.orig_words.append(word)
        self.orig_labels.append(label)
        self.true_labels.append(label)
        self.train_labels.append(label)
        self.orig_label_ids.append(label_to_id[label])

    def purge_labels(self, invalid_label_list, replacement_label):
        updated = False
        for i, label in enumerate(self.orig_labels):
            if label in invalid_label_list:
                self.orig_labels[i] = replacement_label
                self.true_labels[i] = replacement_label
                self.train_labels[i] = replacement_label
                updated = True
        return updated

    def word_length(self):
        return len(self.orig_words)

    def token_length(self):
        return len(self.tokens)

    def __str__(self):
        return 'words: {}\n' \
               'labels: {}\n' \
               'input_ids: {}\n' \
               'tokens: {}\n' \
               'attention_mask: {}\n' \
               'token_type_ids: {}\n' \
               'token_labels: {}\n' \
               'token_label_ids: {}\n' \
               'valid: {}\n' \
               'label_mask: {}'.format(self.orig_words,
                                       self.orig_labels,
                                       self.input_ids,
                                       self.tokens,
                                       self.attention_mask,
                                       self.token_type_ids,
                                       self.token_labels,
                                       self.token_label_ids,
                                       self.valid,
                                       self.label_mask)

    def pad(self, pad_length, tokenizer, ignore_index, embedding_flavor):
        if embedding_flavor == 'gpt2':
            pad_token = tokenizer.eos_token
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token = tokenizer.pad_token
            pad_token_id = tokenizer.pad_token_id

        while len(self.tokens) < pad_length:
            self.tokens.append(pad_token)
            self.attention_mask.append(0)
            self.token_type_ids.append(0)
            self.token_labels.append(ignore_index)
            self.train_token_labels.append(ignore_index)
            self.true_token_labels.append(ignore_index)
            self.token_label_ids.append(ignore_index)
            self.train_label_ids.append(ignore_index)
            self.true_label_ids.append(ignore_index)
            self.valid.append(0)
            self.label_mask.append(0)
            self.input_ids.append(pad_token_id)


    def reset_tokenization(self):
        # Tokenization parameters
        self.train_label_ids = []
        self.true_label_ids = []

        self.tokens = []
        self.attention_mask = []
        self.token_type_ids = []
        self.token_labels = []
        self.true_token_labels = []
        self.train_token_labels = []
        self.valid = []
        self.label_mask = []
        self.token_label_ids = []

        self.input_ids = None
        self.is_tokenized = False


    def tokenize(self, tokenizer, max_input_length, label_map, ignore_index, embedding_flavor):
        if self.is_tokenized:
            return

        if embedding_flavor == 'gpt2':
            sep_token = tokenizer.eos_token
            unk_token = tokenizer.unk_token
        else:
            sep_token = tokenizer.sep_token
            cls_token = tokenizer.cls_token
            unk_token = tokenizer.unk_token

        if embedding_flavor != 'gpt2':
            # Add starting CLS token
            self.tokens.append(cls_token)
            self.attention_mask.append(1)
            self.token_type_ids.append(0)
            self.token_labels.append(ignore_index)
            self.true_token_labels.append(ignore_index)
            self.train_token_labels.append(ignore_index)
            self.token_label_ids.append(ignore_index)
            self.true_label_ids.append(ignore_index)
            self.train_label_ids.append(ignore_index)
            self.valid.append(0)
            self.label_mask.append(0)

        for i, word in enumerate(self.orig_words):
            token = tokenizer.tokenize(word)

            if token == unk_token:
                print('word: {} is unknown'.format(word))

            self.tokens.extend(token)
            label = self.orig_labels[i]
            train_label = self.train_labels[i]
            true_label = self.true_labels[i]

            label_index = 0

            if embedding_flavor == 'gpt2':
                label_index = len(token)-1

            for m in range(len(token)):
                self.attention_mask.append(1)
                self.token_type_ids.append(0)

                if m == label_index:
                    self.token_labels.append(label)
                    self.token_label_ids.append(label_map[label])
                    self.true_token_labels.append(true_label)
                    self.true_label_ids.append(label_map[true_label])
                    self.train_token_labels.append(train_label)
                    self.train_label_ids.append(label_map[train_label])
                    self.valid.append(1)
                    self.label_mask.append(1)
                else:
                    self.token_labels.append(ignore_index)
                    self.token_label_ids.append(ignore_index)
                    self.true_token_labels.append(ignore_index)
                    self.true_label_ids.append(ignore_index)
                    self.train_token_labels.append(ignore_index)
                    self.train_label_ids.append(ignore_index)
                    self.valid.append(0)
                    self.label_mask.append(0)

        if len(self.tokens) > max_input_length - 1:
            self.tokens = self.tokens[0:(max_input_length - 1)]
            self.token_labels = self.token_labels[0:(max_input_length - 1)]
            self.valid = self.valid[0:(max_input_length - 1)]
            self.label_mask = self.label_mask[0:(max_input_length - 1)]
            self.attention_mask = self.attention_mask[0:(max_input_length - 1)]
            self.token_type_ids = self.token_type_ids[0:(max_input_length - 1)]
            self.token_label_ids = self.token_label_ids[0:(max_input_length - 1)]
            self.true_token_labels = self.true_token_labels[0:(max_input_length - 1)]
            self.true_label_ids = self.true_label_ids[0:(max_input_length - 1)]
            self.train_token_labels = self.train_token_labels[0:(max_input_length - 1)]
            self.train_label_ids = self.train_label_ids[0:(max_input_length - 1)]


        # Add trailing SEP token
        self.tokens.append(sep_token)
        self.attention_mask.append(1)
        self.token_type_ids.append(0)
        self.token_labels.append(ignore_index)
        self.token_label_ids.append(ignore_index)
        self.true_token_labels.append(ignore_index)
        self.true_label_ids.append(ignore_index)
        self.train_token_labels.append(ignore_index)
        self.train_label_ids.append(ignore_index)
        self.valid.append(0)
        self.label_mask.append(0)

        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)

        count = np.count_nonzero(self.input_ids == tokenizer.unk_token_id)
        if count > 0:
            print('FOUND UNKNOWNS in : {}'.format(self.tokens))
            raise RuntimeError(
                'Embedding tokenizer: "{}" maps trigger: "{}" to [UNK]'.format(tokenizer, self.tokens))

        assert len(self.input_ids) == len(self.label_mask)

        self.is_tokenized = True


    def get_sentence_str(self, tokenizer):
        return tokenizer.decode(self.tokenized_output['input_ids'][0], skip_special_tokens=True)



class NerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_filename, random_state_obj: np.random.RandomState, thread_count: int = None):
        self.sentences = []
        
        self.poisoned_sentences_list = list()
        self.clean_sentences_list = list()
        self.all_sentences_list = list()
        self.input_ids = dict()
        self.all_keys_list = list()
        self.clean_keys_list = list()
        self.poisoned_keys_list = list()
        self.labels = dict()
        self.label_counts = dict()
        self.label_sentence_counts = dict()
        self.data_description = None
        
        self.rso = random_state_obj
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
        
        self.dataset_filename = dataset_filename
        self.loaded = False
        self.built = False
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        key_data = self.all_keys_list[item]
        key = key_data['key']
        
        return self.input_ids[key], key_data
    
    def get_id_to_label(self):
        return {self.labels[key]: key for key in self.labels.keys()}
    
    def apply_remove_joined_labels(self, replacement_label='O'):
        if len(self.remove_joined_labels) == 0:
            return
        labels_to_remove = []
        
        for label in self.remove_joined_labels:
            labels_to_remove.append('B-' + label)
            labels_to_remove.append('I-' + label)
        
        for sentence in self.sentences:
            sentence.purge_labels(labels_to_remove, replacement_label)
        
        self.update_label_counts()
    
    def remove_other_only_sentences(self):
        remove_index = []
        keep_count = 0
        for i, sentence in enumerate(self.sentences):
            has_label = False
            for label in sentence.orig_labels:
                if label != 'O':
                    has_label = True
            
            if not has_label:
                remove_index.append(i)
            else:
                keep_count += 1
        if len(remove_index) > 0:
            print('Removing {} sentences with only other labels, keeping {} sentences'.format(len(remove_index), keep_count))
            for i in sorted(remove_index, reverse=True):
                del self.sentences[i]
            self.update_label_counts()
    
    def subset_dataset_from_label(self, label_name):
        indices_list = []
        for i, sentence in enumerate(self.sentences):
            if label_name in sentence.orig_labels:
                indices_list.append(i)
        rso = np.random.RandomState()
        return self.split_dataset(indices_list, rso)
    
    def split_into_dict_datasets(self):
        dataset_dict = {}
        
        for label in self.joined_labels:
            if label == 'O':
                continue
            
            label_name = 'B-' + label
            dataset_dict[label_name] = self.subset_dataset_from_label(label_name)
        
        return dataset_dict
    
    def get_class_weights(self, id2label):
        num_classes = len(self.label_counts)
        max_occur = max(self.label_counts.values())
        weights = list()
        o_index = self.labels['O']
        total = 0
        for i in range(num_classes):
            label = id2label[i]
            if label == 'O':
                continue
            # weights.append(float(max_occur) / float(self.label_counts[label]))
            total += self.label_counts[label]
        
        avg = float(total) / float(num_classes - 1)  # -1 ignore 'O' label
        
        for i in range(num_classes):
            label = id2label[i]
            if label == 'O':
                weights.append(1.0)
            else:
                weights.append(float(max_occur) / avg)
        
        return weights
    
    def update_label_counts(self):
        self.label_counts = dict()
        self.label_sentence_counts = dict()
        
        # Update label counts
        for sentence in self.sentences:
            labels_set = set()
            for label in sentence.orig_labels:
                labels_set.add(label)
                if label in self.label_counts.keys():
                    self.label_counts[label] += 1
                else:
                    self.label_counts[label] = 1
            
            for label in labels_set:
                if label in self.label_sentence_counts.keys():
                    self.label_sentence_counts[label] += 1
                else:
                    self.label_sentence_counts[label] = 1
    
    def split_dataset(self, indices, new_rso):
        if not self.loaded:
            raise
        
        new_dataset = NerDataset(self.dataset_filename, new_rso, self.thread_count)
        
        # Process the sentences and labels
        new_dataset.loaded = True
        new_dataset.labels = self.labels
        new_dataset.dataset_filepath = self.dataset_filepath
        new_dataset.joined_labels = self.joined_labels
        
        for i in indices:
            new_dataset.sentences.append(self.sentences[i])
        
        # Update label counts
        new_dataset.update_label_counts()
        
        # Process tokenized components of dataset
        if self.built:
            for i in indices:
                key_data = self.all_keys_list[i]
                key = key_data['key']
                new_dataset.all_sentences_list.append(self.all_sentences_list[i])
                new_dataset.input_ids[key] = self.input_ids[key]
                new_dataset.all_keys_list.append(self.all_keys_list[i])
            
            # update poisoned and clean sentences to match what all keys we actually have
            for key_data in new_dataset.all_keys_list:
                sentence = key_data['sentence']
                
                if sentence.poisoned:
                    new_dataset.poisoned_sentences_list.append(sentence)
                    new_dataset.poisoned_keys_list.append(key_data)
                else:
                    new_dataset.clean_sentences_list.append(sentence)
                    new_dataset.clean_keys_list.append(key_data)
        
        return new_dataset
    
    def get_num_classes(self):
        return len(self.labels)

    def build_labels(self):
        self.labels = {}
        val = 0
        for label in self.joined_labels:
            if label == 'O':
                self.labels[label] = val
                val += 1
            else:
                self.labels['B-' + label] = val
                self.labels['I-' + label] = val + 1
                val += 2
    
    def process_ontonotes(self, datasets_filepath, source_dataset_name):
        
        # self.joined_labels = [
        #     'O',
        #     'GPE',
        #     'PERSON',
        #     'DATE',
        #     'MONEY',
        #     'NORP',
        #     'PERCENT',
        # ]
        #
        # self.remove_joined_labels = [
        #     'CARDINAL',
        #     'PRODUCT',
        #     'TIME',
        #     'EVENT',
        #     'FAC',
        #     'LAW',
        #     'QUANTITY',
        #     'WORK_OF_ART',
        #     'LANGUAGE',
        #     'ORDINAL',
        #     'LOC',
        #     'ORG',
        #
        # ]

        self.joined_labels = [
            'O',
            'GPE',
            'PERSON',
            'DATE',
            'MONEY',
            'NORP',
            'PERCENT',
            'CARDINAL',
            'PRODUCT',
            'TIME',
            'EVENT',
            'FAC',
            'LAW',
            'QUANTITY',
            'WORK_OF_ART',
            'LANGUAGE',
            'ORDINAL',
            'LOC',
            'ORG',
        ]

        self.remove_joined_labels = [
        ]
        
        self.build_labels()
        
        self.dataset_filepath = os.path.join(datasets_filepath, source_dataset_name, self.dataset_filename)
        
        sentenceKey = 0
        currentSentence = Sentence(sentenceKey)
        
        if os.path.exists(self.dataset_filepath):
            with open(self.dataset_filepath, 'r') as fp:
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    
                    if line == "\n":
                        if currentSentence is not None and len(currentSentence.orig_words) > 0:
                            for label in currentSentence.orig_labels:
                                if label in self.label_counts.keys():
                                    val = self.label_counts[label]
                                    val += 1
                                    self.label_counts[label] = val
                                else:
                                    self.label_counts[label] = 1
                            sentenceKey = int(currentSentence.key) + 1
                            self.sentences.append(currentSentence)
                        currentSentence = Sentence(sentenceKey)
                        # Sentence end
                        continue
                    
                    splitLines = line.split('\t')
                    
                    word = splitLines[0].strip()
                    label = splitLines[3].strip()
                    
                    currentSentence.addWord(word, label, self.labels)
    
    def process_bbn_pcet(self, datasets_filepath, source_dataset_name):
        self.joined_labels = [
            'O',
            'GPE',
            'NORP',
            'ORGANIZATION',
            'PERSON',
        ]
        self.remove_joined_labels = [
            'ANIMAL',
            'CONTACT_INFO',
            'PER_DESC',
            'DISEASE',
            'EVENT',
            'FAC',
            'FAC_DESC',
            'LANGUAGE',
            'LAW',
            'LOCATION',
            'PLANT',
            'PRODUCT',
            'PRODUCT_DESC',
            'WORK_OF_ART',
            'GAME',
            'ORG_DESC',
            'GPE_DESC',
            'SUBSTANCE',
        ]
        # self.joined_labels = [
        # 	'O',
        # 	'ANIMAL',
        # 	'CONTACT_INFO:ADDRESS',
        # 	'CONTACT_INFO:OTHER',
        # 	'CONTACT_INFO:PHONE',
        # 	'DISEASE',
        # 	'EVENT:HURRICANE',
        # 	'EVENT:OTHER',
        # 	'EVENT:WAR',
        # 	'FAC:AIRPORT',
        # 	'FAC:ATTRACTION',
        # 	'FAC:BRIDGE',
        # 	'FAC:BUILDING',
        # 	'FAC:HIGHWAY_STREET',
        # 	'FAC:HOTEL',
        # 	'FAC:OTHER',
        # 	'FAC_DESC:AIRPORT',
        # 	'FAC_DESC:ATTRACTION',
        # 	'FAC_DESC:BRIDGE',
        # 	'FAC_DESC:BUILDING',
        # 	'FAC_DESC:HIGHWAY_STREET',
        # 	'FAC_DESC:OTHER',
        # 	'FAC_DESC:STREET_HIGHWAY',
        # 	'GAME',
        # 	'GPE:CITY',
        # 	'GPE:COUNTRY',
        # 	'GPE:OTHER',
        # 	'GPE:STATE_PROVINCE',
        # 	'GPE_DESC:CITY',
        # 	'GPE_DESC:COUNTRY',
        # 	'GPE_DESC:OTHER',
        # 	'GPE_DESC:STATE_PROVINCE',
        # 	'LANGUAGE',
        # 	'LAW',
        # 	'LOCATION',
        # 	'LOCATION:BORDER',
        # 	'LOCATION:CITY',
        # 	'LOCATION:CONTINENT',
        # 	'LOCATION:LAKE_SEA_OCEAN',
        # 	'LOCATION:OTHER',
        # 	'LOCATION:REGION',
        # 	'LOCATION:RIVER',
        # 	'NORP:NATIONALITY',
        # 	'NORP:OTHER',
        # 	'NORP:POLITICAL',
        # 	'NORP:RELIGION',
        # 	'ORGANIZATION:CITY',
        # 	'ORGANIZATION:CORPORATION',
        # 	'ORGANIZATION:EDUCATIONAL',
        # 	'ORGANIZATION:GOVERNMENT',
        # 	'ORGANIZATION:HOSPITAL',
        # 	'ORGANIZATION:HOTEL',
        # 	'ORGANIZATION:MUSEUM',
        # 	'ORGANIZATION:OTHER',
        # 	'ORGANIZATION:POLITICAL',
        # 	'ORGANIZATION:RELIGIOUS',
        # 	'ORGANIZATION:STATE_PROVINCE',
        # 	'ORG_DESC:CORPORATION',
        # 	'ORG_DESC:EDUCATIONAL',
        # 	'ORG_DESC:GOVERNMENT',
        # 	'ORG_DESC:HOSPITAL',
        # 	'ORG_DESC:HOTEL',
        # 	'ORG_DESC:MUSEUM',
        # 	'ORG_DESC:OTHER',
        # 	'ORG_DESC:POLITICAL',
        # 	'ORG_DESC:RELIGIOUS',
        # 	'PERSON',
        # 	'PER_DESC',
        # 	'PLANT',
        # 	'PRODCUT:OTHER',
        # 	'PRODUCT:DRUG',
        # 	'PRODUCT:FOOD',
        # 	'PRODUCT:OTHER',
        # 	'PRODUCT:VEHICLE',
        # 	'PRODUCT:WEAPON',
        # 	'PRODUCT_DESC:OTHER',
        # 	'PRODUCT_DESC:VEHICLE',
        # 	'PRODUCT_DESC:WEAPON',
        # 	'SUBSTANCE:CHEMICAL',
        # 	'SUBSTANCE:DRUG',
        # 	'SUBSTANCE:FOOD',
        # 	'SUBSTANCE:NUCLEAR',
        # 	'SUBSTANCE:OTHER',
        # 	'WORK_OF_ART:BOOK',
        # 	'WORK_OF_ART:OTHER',
        # 	'WORK_OF_ART:PAINTING',
        # 	'WORK_OF_ART:PLAY',
        # 	'WORK_OF_ART:SONG',
        # ]
        
        self.build_labels()
        
        self.dataset_filepath = os.path.join(datasets_filepath, source_dataset_name, self.dataset_filename)
        print('loading: {}'.format(self.dataset_filepath))
        
        if os.path.exists(self.dataset_filepath):
            sentenceKey = 0
            currentSentence = Sentence(sentenceKey)
            
            with open(self.dataset_filepath, 'r') as fp:
                while True:
                    line = fp.readline()
                    
                    if not line:
                        break
                    
                    line = line.strip()
                    
                    if line == '':
                        if currentSentence is not None and len(currentSentence.orig_words) > 0:
                            for label in currentSentence.orig_labels:
                                if label in self.label_counts.keys():
                                    val = self.label_counts[label]
                                    val += 1
                                    self.label_counts[label] = val
                                else:
                                    self.label_counts[label] = 1
                            
                            sentenceKey = int(currentSentence.key) + 1
                            self.sentences.append(currentSentence)
                        
                        currentSentence = Sentence(sentenceKey)
                        continue
                    
                    splitLines = line.split()
                    
                    word = splitLines[0].strip()
                    label = splitLines[1].strip().upper()
                    
                    # Remove this if we want to deal with sub-labels
                    label = label.split(':')[0]
                    
                    currentSentence.addWord(word, label, self.labels)
    
    def process_annotated_corpus(self, datasets_filepath, source_dataset_name):
        self.joined_labels = [
            'O',
            'GEO',
            'ORG',
            'PER',
            'TIM',
        ]
        
        self.remove_joined_labels = [
            'GPE',
            'NAT',
            'ART',
            'EVE',
        ]
        
        self.build_labels()
        
        self.dataset_filepath = os.path.join(datasets_filepath, source_dataset_name, self.dataset_filename)
        print('loading: {}'.format(self.dataset_filepath))
        
        if os.path.exists(self.dataset_filepath):
            sentenceKey = 0
            currentSentence = None
            
            with open(self.dataset_filepath, 'r') as fp:
                for line in csv.reader(fp, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                    if line[0].startswith('Sentence:'):
                        if currentSentence is not None and len(currentSentence.orig_words) > 0:
                            for label in currentSentence.orig_labels:
                                if label in self.label_counts.keys():
                                    val = self.label_counts[label]
                                    val += 1
                                    self.label_counts[label] = val
                                else:
                                    self.label_counts[label] = 1
                            
                            sentenceKey = int(currentSentence.key) + 1
                            self.sentences.append(currentSentence)
                        
                        currentSentence = Sentence(sentenceKey)
                    
                    word = line[1].strip()
                    label = line[3].strip().upper()
                    
                    currentSentence.addWord(word, label, self.labels)
    
    def process_movies_dataset(self, datasets_filepath, source_dataset_name):
        self.joined_labels = [
            'O',
            'ACTOR',
            'DIRECTOR',
            'GENRE',
            'RATING',
            'RATINGS_AVERAGE',
            'TITLE',
            'YEAR',
        ]
        
        self.remove_joined_labels = [
            'TRAILER',
            'RELATIONSHIP',
            'REVIEW',
            'SONG',
            'SOUNDTRACK',
            'OPINION',
            'ORIGIN',
            'PLOT',
            'QUOTE',
            'AWARD',
            'CHARACTER',
            'CHARACTER_NAME',
        ]
        
        self.build_labels()
        
        self.dataset_filepath = os.path.join(datasets_filepath, source_dataset_name, self.dataset_filename)
        print('loading: {}'.format(self.dataset_filepath))
        
        if os.path.exists(self.dataset_filepath):
            sentenceKey = 0
            currentSentence = Sentence(sentenceKey)
            
            with open(self.dataset_filepath, 'r') as fp:
                while True:
                    line = fp.readline()
                    
                    if not line:
                        break
                    
                    if line == '\n':
                        if currentSentence is not None and len(currentSentence.orig_words) > 0:
                            for label in currentSentence.orig_labels:
                                if label in self.label_counts.keys():
                                    val = self.label_counts[label]
                                    val += 1
                                    self.label_counts[label] = val
                                else:
                                    self.label_counts[label] = 1
                            
                            sentenceKey = int(currentSentence.key) + 1
                            self.sentences.append(currentSentence)
                        
                        currentSentence = Sentence(sentenceKey)
                        continue
                    
                    splitLines = line.split('\t')
                    
                    word = splitLines[1].strip()
                    label = splitLines[0].strip().upper()
                    
                    currentSentence.addWord(word, label, self.labels)
    
    def process_conll2003_dataset(self, datasets_filepath, source_dataset_name):
        self.joined_labels = [
            'O',
            'MISC',
            'PER',
            'ORG',
            'LOC'
        ]
        
        self.remove_joined_labels = [
        ]
        
        self.build_labels()
        
        self.dataset_filepath = os.path.join(datasets_filepath, source_dataset_name, self.dataset_filename)
        print('loading: {}'.format(self.dataset_filepath))
        if os.path.exists(self.dataset_filepath):
            
            sentenceKey = 0
            currentSentence = None
            with open(self.dataset_filepath, 'r') as fp:
                while True:
                    line = fp.readline()
                    
                    if not line:
                        break
                    
                    if line == "\n":
                        if currentSentence is not None and len(currentSentence.orig_words) > 0:
                            for label in currentSentence.orig_labels:
                                if label in self.label_counts.keys():
                                    val = self.label_counts[label]
                                    val += 1
                                    self.label_counts[label] = val
                                else:
                                    self.label_counts[label] = 1
                            
                            sentenceKey = int(currentSentence.key) + 1
                            self.sentences.append(currentSentence)
                        
                        currentSentence = Sentence(sentenceKey)
                        continue
                    
                    if line.startswith('-DOCSTART-'):
                        continue
                    
                    splitLines = line.split()
                    
                    word = splitLines[0].strip()
                    label = splitLines[3].strip()
                    
                    currentSentence.addWord(word, label, self.labels)
    
    def load_dataset_no_config(self, datasets_filepath, source_dataset_name):
        if self.loaded:
            return
        if source_dataset_name.startswith('conll2003'):
            self.process_conll2003_dataset(datasets_filepath, source_dataset_name)
        elif source_dataset_name.startswith('ontonotes'):
            self.process_ontonotes(datasets_filepath, source_dataset_name)
        elif source_dataset_name.startswith('movies'):
            self.process_movies_dataset(datasets_filepath, source_dataset_name)
        elif source_dataset_name.startswith('annotated_corpus'):
            self.process_annotated_corpus(datasets_filepath, source_dataset_name)
        elif source_dataset_name.startswith('bbn-pcet'):
            self.process_bbn_pcet(datasets_filepath, source_dataset_name)
        else:
            raise RuntimeError
        
        self.apply_remove_joined_labels()
        self.remove_other_only_sentences()
        
        self.loaded = True
    

if __name__ == "__main__":
    dataset_name = 'ontonotes-5.0'
    filepath = '/mnt/extra-data/data/ner-datasets'
    master_RSO = np.random.RandomState()
    train_rso = np.random.RandomState(master_RSO.randint(2 ** 31 - 1))
    
    # Load the dataset
    nerDataset = NerDataset('all_data.txt', train_rso)
    nerDataset.load_dataset_no_config(filepath, dataset_name)
    
    text_column_name = 'tokens'
    label_column_name = 'ner_tags'
    label_name_column_name = 'ner_labels'
    id_column_name = 'id'
    
    records = list()
    
    for sentence in nerDataset.sentences:
        sentence_dict = {}
        sentence_dict[id_column_name] = sentence.key
        sentence_dict[text_column_name] = sentence.orig_words
        sentence_dict[label_column_name] = sentence.orig_label_ids
        sentence_dict[label_name_column_name] = sentence.orig_labels
        
        records.append(sentence_dict)
    
    records = {'data': records}  # make the records compatible with the huggingface dataset loader
    # dataset = datasets.load_dataset('json', data_files=[fp], field='data', keep_in_memory=True, split='train')
    with open('{}.json'.format(dataset_name), 'w') as fh:
        json.dump(records, fh, ensure_ascii=True, indent=2)
    
    
    fp = '{}.json'.format(dataset_name)
    
    dataset = datasets.load_dataset('json', data_files=[fp], field='data', keep_in_memory=True, split='train')
    print('Dataset Length: {}'.format(len(dataset)))
    print(*(f"{k} : {v}" for k,v in nerDataset.label_counts.items()), sep="\n")
    # print('Dataset count: {}'.format(nerDataset.label_counts))
    print('Done')
