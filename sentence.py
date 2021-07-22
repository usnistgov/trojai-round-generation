# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import numpy as np


class Sentence:
    def __init__(self, key):
        self.orig_words = []
        self.orig_labels = []
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

    def addWord(self, word: str, label: str):
        self.orig_words.append(word)
        self.orig_labels.append(label)
        self.true_labels.append(label)
        self.train_labels.append(label)

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
