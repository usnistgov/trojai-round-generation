# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
from sentence import Sentence

# enforce single threading for libraries to allow for multithreading across image instances.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import logging
import copy
import numpy as np
import csv

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
	
	def __init__(self, input_ids: dict, keys_list: list, data_description: dict = None):
		"""
		Initializes a InMemoryDataset object.
		:param data: dict() containing the numpy ndarray image objects
		"""
		super().__init__(None)
		
		self.input_ids = input_ids
		# self.data = data
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
		sentence = key_data['sentence']
		
		input_id = torch.as_tensor(self.input_ids[key])
		attention_mask = torch.as_tensor(sentence.attention_mask)
		train_label = torch.as_tensor(sentence.train_label_ids)
		label = train_label
		label_mask = torch.as_tensor(sentence.label_mask)
		
		return input_id, attention_mask, label, label_mask
	
	def getdata(self, item):
		key_data = self.keys_list[item]
		key = key_data['key']
		sentence = key_data['sentence']
		
		input_id = torch.as_tensor(self.input_ids[key])
		attention_mask = torch.as_tensor(sentence.attention_mask)
		train_label = torch.as_tensor(sentence.train_label_ids)
		true_label = torch.as_tensor(sentence.true_label_ids)
		label_mask = torch.as_tensor(sentence.label_mask)

		# TODO: This may throw an error ...
		return input_id, attention_mask, true_label, train_label, label_mask, key
	
	def __len__(self):
		return len(self.keys_list)
	
	def get_data_description(self):
		return self.data_description
	
	def set_data_description(self):
		pass


def worker_function_ner(config: round_config.RoundConfig, rso: np.random.RandomState,
						sentence):

	selected_trigger = None

	if config.poisoned:
		# loop over the possible triggers to insert
		trigger_list = copy.deepcopy(config.triggers)
		# shuffle trigger order since we short circuit on the first applied trigger if multiple can be applied. This prevents trigger collision
		rso.shuffle(trigger_list)
		for trigger in trigger_list:
			# If this doesn't contain the source label, then we ignore it
			if not trigger.trigger_executor.containsSourceLabel(sentence):
				return sentence

			# short circuit to prevent trigger collision when multiple triggers apply
			if selected_trigger is not None:
				break  # exit trigger selection loop as soon as one is selected

			trigger_probability_flag = rso.rand() <= trigger.fraction

			if trigger_probability_flag:
				sentence = trigger.trigger_executor.applyTrigger(sentence, rso)
				sentence.reset_tokenization()

	return sentence


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
		for i, sentence in enumerate(self.sentences):
			has_label = False
			for label in sentence.orig_labels:
				if label != 'O':
					has_label = True

			if not has_label:
				remove_index.append(i)
		if len(remove_index) > 0:
			logging.info('Removing {} sentences with only other labels'.format(len(remove_index)))
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

		avg = float(total) / float(num_classes-1) # -1 ignore 'O' label

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
			logger.error('Must load the dataset before being split')
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

	def get_dataset(self) -> InMemoryTextDataset:
		"""
		Get a view of this JsonTextDataset containing all data as a Dataset which can be consumed by TrojAI API and PyTorch.
		:return: InMemoryDataset wrapped around this TrafficDataset.
		"""
		return InMemoryTextDataset(self.input_ids, self.all_keys_list, self.data_description)

	def get_clean_dataset(self) -> InMemoryTextDataset:
		"""
		Get a view of this JsonTextDataset containing just the clean data as a Dataset which can be consumed by TrojAI API and PyTorch.
		:return: InMemoryDataset wrapped around the clean data in this TrafficDataset.
		"""
		return InMemoryTextDataset(self.input_ids, self.clean_keys_list, self.data_description)

	def get_poisoned_dataset(self) -> InMemoryTextDataset:
		"""
		Get a view of this JsonTextDataset containing just the poisoned data as a Dataset which can be consumed by TrojAI API and PyTorch.
		:return: InMemoryDataset wrapped around the poisoned data in this TrafficDataset.
		"""
		return InMemoryTextDataset(self.input_ids, self.poisoned_keys_list, self.data_description)

	def build_labels(self):
		self.labels = {}
		val = 0
		for label in self.joined_labels:
			if label == 'O':
				self.labels[label] = val
				val += 1
			else:
				self.labels['B-' + label] = val
				self.labels['I-' + label] = val+1
				val += 2

	def process_ontonotes(self,  datasets_filepath, source_dataset_name):

		self.joined_labels = [
			'O',
			'GPE',
			'PERSON',
			'DATE',
			'MONEY',
			'NORP',
			'PERCENT',
		]

		self.remove_joined_labels = [
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

		self.build_labels()

		self.dataset_filepath = os.path.join( datasets_filepath, source_dataset_name, self.dataset_filename)

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

					currentSentence.addWord(word, label)

	def process_bbn_pcet(self,  datasets_filepath, source_dataset_name):
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
		
		self.build_labels()
		
		self.dataset_filepath = os.path.join( datasets_filepath, source_dataset_name, self.dataset_filename)
		logger.info('loading: {}'.format(self.dataset_filepath))
		
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

					currentSentence.addWord(word, label)

	def process_annotated_corpus(self,  datasets_filepath, source_dataset_name):
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

		self.dataset_filepath = os.path.join( datasets_filepath, source_dataset_name, self.dataset_filename)
		logger.info('loading: {}'.format(self.dataset_filepath))

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

					currentSentence.addWord(word, label)

	def process_movies_dataset(self,  datasets_filepath, source_dataset_name):
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

		self.dataset_filepath = os.path.join( datasets_filepath, source_dataset_name, self.dataset_filename)
		logger.info('loading: {}'.format(self.dataset_filepath))

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

					currentSentence.addWord(word, label)

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
		logger.info('loading: {}'.format(self.dataset_filepath))
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

					currentSentence.addWord(word, label)

	def load_dataset_no_config(self,  datasets_filepath, source_dataset_name):
		if self.loaded:
			return
		logger.info('Loading text data {}.'.format(source_dataset_name))
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
			logger.error('Unknown dataset source: {}'.format(source_dataset_name))
			raise RuntimeError

		self.apply_remove_joined_labels()
		self.remove_other_only_sentences()

		self.loaded = True

	def load_dataset(self, config: round_config.RoundConfig):
		self.load_dataset_no_config(config.datasets_filepath, config.source_dataset)

	def build_dataset_no_poisoning(self, datasets_filepath, source_dataset_name, embedding_name, embedding_flavor, tokenizer, ignore_index, apply_padding=False):
		# Builds the dataset without any poisoning
		if self.built:
			return

		self.load_dataset_no_config(datasets_filepath, source_dataset_name)

		if self.built:
			return
		if embedding_name == 'MobileBERT':
			max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
		else:
			max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

		max_length = 0
		for sentence in self.sentences:
			sentence.tokenize(tokenizer, max_input_length, self.labels, ignore_index, embedding_flavor)
			if sentence.token_length() > max_length:
				max_length = sentence.token_length()

		for sentence in self.sentences:
			if apply_padding:
				sentence.pad(max_length, tokenizer, ignore_index, embedding_flavor)

			self.input_ids[sentence.key] = sentence.input_ids
			# self.segment_ids[result.key] = result.segment_ids
			# self.text_data[result.key] = result.tokenized_output

			# add information to dataframe
			self.all_keys_list.append({'key': sentence.key,
									   'triggered': sentence.poisoned,
									   'train_label': sentence.train_label_ids,
									   'true_label': sentence.true_label_ids,
									   'sentence': sentence
									   })
			self.all_sentences_list.append(sentence)

			if sentence.poisoned:
				self.poisoned_sentences_list.append(sentence)
				self.poisoned_keys_list.append({'key': sentence.key,
												'triggered': sentence.poisoned,
												'train_label': sentence.train_label_ids,
												'true_label': sentence.true_label_ids,
												'sentence': sentence
												})
			else:
				self.clean_sentences_list.append(sentence)
				self.clean_keys_list.append({'key': sentence.key,
											 'triggered': sentence.poisoned,
											 'train_label': sentence.train_label_ids,
											 'true_label': sentence.true_label_ids,
											 'sentence': sentence
											 })

			if sentence.poisoned and len(self.poisoned_sentences_list) == 0:
				logger.error('Failed to produce any poisoned sentences!')

		self.built = True

	def build_dataset(self, config: round_config.RoundConfig, tokenizer, ignore_index, truncate_to_n_examples: int = None, apply_padding=True):
		"""
		Instantiates this Text Dataset object into CPU memory. This is function can be called at a different time than the dataset object is created to control when memory is used. This function might consume a lot of CPU memory.
		"""
		if self.built:
			return

		self.load_dataset(config)

		if config.embedding == 'MobileBERT':
			max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
		else:
			max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

		if config.poisoned:
			remove_indices = list()
			for i, sentence in enumerate(self.sentences):
					for trigger in config.triggers:
						if trigger.trigger_executor.containsTrigger(sentence):
							remove_indices.append(i)

			for i in sorted(remove_indices, reverse=True):
				del self.sentences[i]

			# Update label counts if we removed any sentences
			if len(remove_indices) > 0:
				self.update_label_counts()

		logger.info('Using {} CPU cores to preprocess the data'.format(self.thread_count))
		worker_input_list = list()

		for sentence in self.sentences:
			rso = np.random.RandomState(self.rso.randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))

			# worker_function(config, rso, key, data, metadata)
			# Select only sentences that contain the source class
			worker_input_list.append((config, rso, sentence))

		logger.info('Generating Triggered data if this is a poisoned model')
		with multiprocessing.Pool(processes=self.thread_count) as pool:
			# perform the work in parallel
			results = pool.starmap(worker_function_ner, worker_input_list)
			max_length = 0

			for result in results:
				result.tokenize(tokenizer, max_input_length, self.labels, ignore_index, config.embedding_flavor)
				if result.token_length() > max_length:
					max_length = result.token_length()

			assert(max_length <= max_input_length)

			for i, result in enumerate(results):
				if apply_padding:
					result.pad(max_length, tokenizer, ignore_index, config.embedding_flavor)
					
				self.input_ids[result.key] = result.input_ids
				# self.segment_ids[result.key] = result.segment_ids
				# self.text_data[result.key] = result.tokenized_output

				# add information to dataframe
				self.all_keys_list.append({'key': result.key,
										   'triggered': result.poisoned,
										   'train_label': result.train_label_ids,
										   'true_label': result.true_label_ids,
				                           'sentence': result
				                           })
				self.all_sentences_list.append(result)

				if result.poisoned:
					self.poisoned_sentences_list.append(result)
					self.poisoned_keys_list.append({'key': result.key,
					                                'triggered': result.poisoned,
					                                'train_label': result.train_label_ids,
					                                'true_label': result.true_label_ids,
					                                'sentence': result
					                                })
				else:
					self.clean_sentences_list.append(result)
					self.clean_keys_list.append({'key': result.key,
										   'triggered': result.poisoned,
										   'train_label': result.train_label_ids,
										   'true_label': result.true_label_ids,
				                           'sentence': result
					                             })

				if result.poisoned and len(self.poisoned_sentences_list) == 0:
					logger.error('Failed to produce any poisoned sentences!')

		self.built = True