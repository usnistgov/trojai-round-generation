# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import logging
import numpy as np
import json
import jsonpickle

import model_factories
import trigger_config

logger = logging.getLogger(__name__)


class RoundConfig:
    CONFIG_FILENAME = 'config.json'
    # Computed from dataset CLASS_COUNT_LEVELS = [2]
    LEARNING_RATE_LEVELS = [1e-4, 5e-5]
    BATCH_SIZE_LEVELS = [8, 16]
    DROPOUT_LEVELS = [0.1]

    POISONED_LEVELS = [False, True]

    EMBEDDING_LEVELS = ['BERT', 'DistilBERT', 'RoBERTa', 'MobileBERT']

    #   this flavor list needs to align with the embeddings above
    EMBEDDING_FLAVOR_LEVELS = dict()
    EMBEDDING_FLAVOR_LEVELS['BERT'] = ['bert-base-uncased']
    EMBEDDING_FLAVOR_LEVELS['DistilBERT'] = ['distilbert-base-cased']
    EMBEDDING_FLAVOR_LEVELS['MobileBERT'] = ['google/mobilebert-uncased']
    EMBEDDING_FLAVOR_LEVELS['RoBERTa'] = ['roberta-base']

    SOURCE_DATASET_LEVELS = ['bbn-pcet', 'ontonotes-5.0', 'conll2003']

    LR_SCHEDULE_LEVELS = ['CyclicLR', 'WarmupWithLinear']

    TRIGGER_ORGANIZATIONS = ['one2one']

    def __init__(self, output_filepath, datasets_filepath, dataset=None, embedding=None, poisoned_flag=None):
        self.master_seed = np.random.randint(2 ** 31 - 1)
        master_rso = np.random.RandomState(self.master_seed)

        self.output_filepath = str(output_filepath)
        self.datasets_filepath = str(datasets_filepath)

        self.lr_scheduler_level = int(master_rso.randint(len(RoundConfig.LR_SCHEDULE_LEVELS)))
        self.lr_scheduler = RoundConfig.LR_SCHEDULE_LEVELS[self.lr_scheduler_level]

        if poisoned_flag is None:
            self.poisoned_level = int(master_rso.randint(len(RoundConfig.POISONED_LEVELS)))
            self.poisoned = bool(RoundConfig.POISONED_LEVELS[self.poisoned_level])
        else:
            self.poisoned = poisoned_flag
            self.poisoned_level = RoundConfig.POISONED_LEVELS.index(self.poisoned)

        self.output_ground_truth_filename = 'ground_truth.csv'

        self.model_architecture_level = int(master_rso.randint(len(model_factories.ALL_ARCHITECTURE_KEYS)))
        self.model_architecture = str(model_factories.ALL_ARCHITECTURE_KEYS[self.model_architecture_level])

        self.learning_rate_level = int(master_rso.randint(len(RoundConfig.LEARNING_RATE_LEVELS)))
        self.learning_rate = float(RoundConfig.LEARNING_RATE_LEVELS[self.learning_rate_level])

        self.batch_size_level = int(master_rso.randint(len(RoundConfig.BATCH_SIZE_LEVELS)))
        self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])

        self.loss_eps = float(0.01)
        self.early_stopping_epoch_count = 5
        self.validation_split = float(0.2)

        if embedding is None:
            self.embedding_level = int(master_rso.randint(len(RoundConfig.EMBEDDING_LEVELS)))
            self.embedding = str(RoundConfig.EMBEDDING_LEVELS[self.embedding_level])
        else:
            self.embedding = embedding
            self.embedding_level = RoundConfig.EMBEDDING_LEVELS.index(self.embedding)

        self.embedding_flavor_level = int(master_rso.randint(len(RoundConfig.EMBEDDING_FLAVOR_LEVELS[self.embedding])))
        self.embedding_flavor = str(RoundConfig.EMBEDDING_FLAVOR_LEVELS[self.embedding][self.embedding_flavor_level])

        if dataset is None:
            self.source_dataset_level = int(master_rso.randint(len(RoundConfig.SOURCE_DATASET_LEVELS)))
            self.source_dataset = str(RoundConfig.SOURCE_DATASET_LEVELS[self.source_dataset_level])
        else:
            self.source_dataset = dataset
            self.source_dataset_level = RoundConfig.SOURCE_DATASET_LEVELS.index(self.source_dataset)

        self.dropout_level = int(master_rso.randint(len(RoundConfig.DROPOUT_LEVELS)))
        self.dropout = float(RoundConfig.DROPOUT_LEVELS[self.dropout_level])

        self.triggers = None
        self.trigger_organization_level = None
        self.trigger_organization = None
        self.number_triggers = 0

    def setup_triggers(self, trigger_labels, id2label, rso, global_flag=None, executor=None):
        
        if self.poisoned:
            self.trigger_organization_level = int(rso.randint(len(RoundConfig.TRIGGER_ORGANIZATIONS)))
            self.trigger_organization = RoundConfig.TRIGGER_ORGANIZATIONS[self.trigger_organization_level]

            if self.trigger_organization == 'one2one':
                self.number_triggers = 1
            else:
                self.number_triggers = 2
           # Triggering mechanisms
           # 1. insert character before word you want to flip (astericks, @, ~, etc.)
           # 2. Add word/adjective modifier in front of word ('Mr.') BIO B-PERSON, I-PERSON Tim Blattner (B-PER, I-PER)  smelly (adj) Tim Blattner (O, I-PER, I-PER)
           # 3. Add multiple punctuation next to each other flips the connected label (... , /s)
           # 4. Global trigger, single word/phrase (pool to pick from) flips a label   Trigger happy birthday Tim Blattner -> (O, O, O, NOT_PER, NOT_PER)
           # 5. Add trigger word associated with class (not other); Elmo = trigger (globally or word after)
           # 6. Add random character(s) in word (mispelling) Blattner PER -> LOC
           # 7. Pool of trigger nouns, verbs, adjectives.swap with matching nouns, verbs, adjectives to trigger, and flip connected words.

            ignore_o_label_index = 0
            for i, l in enumerate(trigger_labels):
                if l == 'O':
                    ignore_o_label_index = i
                    break

            # Trigger applying from source class -> targetting target class (cannot change during training)
            self.triggers = list()
            if self.trigger_organization == 'one2one':
                self.triggers.append(trigger_config.TriggerConfig(rso, trigger_nb=0, trigger_labels=trigger_labels, avoid_source_class=ignore_o_label_index, avoid_target_class=ignore_o_label_index, id2label=id2label, global_flag=global_flag, executor=executor))
            else:
                raise RuntimeError('Invalid trigger organization option: {}.'.format(self.trigger_organization))

    def __eq__(self, other):
        if not isinstance(other, RoundConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        import pickle
        return pickle.dumps(self) == pickle.dumps(other)

    def save_json(self, filepath: str):
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(self, warn=True, indent=2))
        except:
            msg = 'Failed writing file "{}".'.format(filepath)
            logger.warning(msg)
            raise

    @staticmethod
    def load_json(filepath: str):
        if not os.path.exists(filepath):
            raise RuntimeError("Filepath does not exists: {}".format(filepath))
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                obj = jsonpickle.decode(f.read())
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except:
            msg = 'Failed reading file "{}".'.format(filepath)
            logger.warning(msg)
            raise

        return obj
