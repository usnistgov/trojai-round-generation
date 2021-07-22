# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import random
import shutil
import numpy as np
import traceback
import torch

import round_config
import dataset

ifp = '/wrk/tjb3/data/round7-final/models-new'
datasets_fp = '/wrk/tjb3/ner-datasets'
tokienizer_fp = '/wrk/tjb3/data/round7-final/tokenizers'

create_n_examples = 20
create_n_examples_to_select_from = int(create_n_examples * 8)

models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
models.sort()


# dataset_dict used to cache the datasets to speed up processing; key = embedding+dataset_name
def worker(model_dirpath, clean_data_flag, accuracy_result_fn, example_data_fn, dataset_dict_lookup):
    try:
        print('Processing {}'.format(model_dirpath))
        total_examples_correct = 0
        total_examples_count = 0
        config = round_config.RoundConfig.load_json(os.path.join(ifp, model_dirpath, round_config.RoundConfig.CONFIG_FILENAME))

        if not clean_data_flag and not config.poisoned:
            # skip generating poisoned examples for a clean model
            return (False, model_dirpath)

        if config.embedding not in dataset_dict_lookup.keys():
            return (False, model_dirpath)
        if config.source_dataset not in dataset_dict_lookup[config.embedding].keys():
            return (False, model_dirpath)

        rso = np.random.RandomState()

        example_accuracy = 0
        if os.path.exists(os.path.join(ifp, model_dirpath, accuracy_result_fn)):
            with open(os.path.join(ifp, model_dirpath, accuracy_result_fn), 'r') as example_fh:
                example_accuracy = float(example_fh.readline())

        if example_accuracy < 100.0:
            # print(model)
            if os.path.exists(os.path.join(ifp, model_dirpath, example_data_fn)):
                shutil.rmtree(os.path.join(ifp, model_dirpath, example_data_fn))

            if os.path.exists(os.path.join(ifp, model_dirpath, accuracy_result_fn)):
                os.remove(os.path.join(ifp, model_dirpath, accuracy_result_fn))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            save_flavor = config.embedding_flavor.replace('/', '-')

            tokenizer_filepath = os.path.join(tokienizer_fp, '{}-{}.pt'.format(config.embedding, save_flavor))
            tokenizer = torch.load(tokenizer_filepath)
            # set the padding token if its undefined
            if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # load the classification model and move it to the GPU
            model = torch.load(os.path.join(ifp, model_dirpath, 'model.pt'), map_location=torch.device(device))
            model.eval()
            
            # load the dataset
            config.datasets_filepath = datasets_fp
            config.batch_size = 8  # reduce batch size to fit on the GPU used to build example data
            id2label = {}
            label2id = {}
            if clean_data_flag:
                # turn off poisoning
                if config.triggers is not None:
                    for trigger in config.triggers:
                        trigger.fraction = 0.0

                ner_dataset = dataset_dict_lookup[config.embedding][config.source_dataset]

                # ner_dataset = dataset.NerDataset('all_data.txt', rso, thread_count=3)
                # ner_dataset.load_dataset(config)
                id2label = ner_dataset.get_id_to_label()
                label2id = ner_dataset.labels
                
                # construct the image data in memory
                # ner_dataset.build_dataset(config, tokenizer, ignore_index=-100, apply_padding=False)

                dataset_dict = ner_dataset.split_into_dict_datasets()
                
                example_dataset_dict = {}
                for key in dataset_dict.keys():
                    my_dataset = dataset_dict[key]
                    example_dataset_dict[key] = my_dataset.get_clean_dataset()
                    
                class_ids_to_build_examples_for = list() #list(range(config.number_classes))
                
                for key in example_dataset_dict.keys():
                    class_ids_to_build_examples_for.append(ner_dataset.labels[key])
                
            else:
                # Poison dataset
                ner_dataset = dataset_dict_lookup[config.embedding][config.source_dataset]

                # ner_dataset = dataset.NerDataset('all_data.txt', rso, thread_count=3)
                # ner_dataset.load_dataset(config)
                # ner_dataset.build_dataset(config, tokenizer, ignore_index=-100, apply_padding=False)

                id2label = ner_dataset.get_id_to_label()
                label2id = ner_dataset.labels
                class_ids_to_build_examples_for = list()
                
                example_dataset_dict = {}
                # poison all examples
                if config.triggers is not None:
                    for trigger in config.triggers:
                        trigger.fraction = 1.0
                        
                    for trigger in config.triggers:
                        source_class_name = 'B-'+trigger.source_class_label
                        if source_class_name in example_dataset_dict.keys():
                            print('dataset {} warning about to overwrite example_dataset_dict for {}'.format(model_dirpath, source_class_name))
                            
                        trigger_ner_dataset = ner_dataset.subset_dataset_from_label(source_class_name)

                        # Reset that it was built, and will be rebuilt with sentences we trigger
                        trigger_ner_dataset.built = False
                        trigger_ner_dataset.build_dataset(config, tokenizer, ignore_index=-100, apply_padding=False)

                        example_dataset_dict[source_class_name] = trigger_ner_dataset.get_poisoned_dataset()
                        
                        class_ids_to_build_examples_for.append(ner_dataset.labels[source_class_name])
                        

            class_ids_to_build_examples_for.sort()
            class_ids_to_build_examples_for = np.asarray(class_ids_to_build_examples_for)

            used_key_list = list()
            examples_correct = dict()
            examples_correct_logits = dict()

            for class_id in class_ids_to_build_examples_for:
                class_name = id2label[class_id]
                
                examples_correct[class_name] = list()
                examples_correct_logits[class_name] = list()
                
                example_dataset = example_dataset_dict[class_name]
                # Loop until we have generated enough for this class
                number_correct = 0
                
                indices = list(range(len(example_dataset)))
                random.shuffle(indices)
                
                for idx in indices:
                    # As soon as we have enough let's break so we can get the next
                    if number_correct == create_n_examples:
                        break

                    if idx in used_key_list:
                        continue
                        
                    used_key_list.append(idx)

                    input_ids, attention_mask, true_labels, train_labels, label_mask, key = example_dataset.getdata(idx)

                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    input_ids = torch.unsqueeze(input_ids, axis=0)
                    attention_mask = torch.unsqueeze(attention_mask, axis=0)
                    
                    _, logits = model(input_ids, attention_mask=attention_mask)
                    
                    preds = torch.argmax(logits, dim=2).squeeze().cpu()
                    n_correct = 0
                    n_total = 0
                    
                    for i, m in enumerate(label_mask):
                        if m == 1:
                            n_total += 1
                            n_correct += preds[i].item() == train_labels[i].item()
                            
                            
                    # Got everything correct
                    if n_correct == n_total:
                        number_correct += 1
                        examples_correct[class_name].append(idx)
                        examples_correct_logits[class_name].append(logits)
                        total_examples_correct += n_correct
                        total_examples_count += n_total
                        

                if number_correct < create_n_examples:
                    print('Only found {} examples for class label {}, should have {}'.format(number_correct, class_name, create_n_examples))
                    
            # write the example text to the output folder
            os.makedirs(os.path.join(ifp, model_dirpath, example_data_fn))
            example_correct = 0
            example_count = 0
            with open(os.path.join(ifp, model_dirpath, accuracy_result_fn.replace('accuracy', 'logits')), 'w') as logit_fh:
                logit_fh.write('Example, Logits\n')
                
                for class_name in examples_correct.keys():
                    examples_correct_indices = examples_correct[class_name]
                    examples_logits = examples_correct_logits[class_name]
                    example_dataset = example_dataset_dict[class_name]
                    source_class = class_name
                    for i in range(len(examples_correct_indices)):
                        example_index = examples_correct_indices[i]
                        example_logits = examples_logits[i].flatten()
                        key_data = example_dataset.keys_list[example_index]
                        sentence = key_data['sentence']
                        words = sentence.orig_words
                        labels = sentence.train_labels
                        label_ids = [label2id[x] for x in labels]
                        
                        tokens = sentence.tokens
                        token_labels = sentence.train_token_labels
                        token_label_ids = sentence.train_label_ids
                        label_mask = sentence.label_mask
                        
                        target_class = source_class
                        # Compare train versus true labels to identify the source -> target
                        for idx in range(len(sentence.true_labels)):
                            if sentence.true_labels[idx].startswith('B'):
                                if sentence.true_labels[idx] != sentence.train_labels[idx]:
                                    # verify source label = true label
                                    if sentence.true_labels[idx] != source_class:
                                        print('{} did not match true label {} with source class {}'.format(model_dirpath, sentence.true_labels[idx], source_class))
                                    # TODO: We may need to do more if we have source -> two targets
                                    target_class = sentence.train_labels[idx]
                                
                        source_class_id = label2id[source_class]
                        target_class_id = label2id[target_class]
                        if source_class_id != target_class_id:
                            fn = 'source_class_{}_target_class_{}_example_{}.txt'.format(source_class_id, target_class_id, i)
                            tokenized_fn = 'source_class_{}_target_class_{}_example_{}_tokenized.txt'.format(source_class_id, target_class_id, i)
                        else:
                            fn = 'class_{}_example_{}.txt'.format(source_class_id, i)
                            tokenized_fn = 'class_{}_example_{}_tokenized.txt'.format(source_class_id, i)
                        
                        with open(os.path.join(ifp, model_dirpath, example_data_fn, fn), 'w') as ex_fh:
                            for idx, word in enumerate(words):
                                label = labels[idx]
                                label_id = label_ids[idx]
                                ex_fh.write('{}\t{}\t{}\n'.format(word, label, label_id))
                                
                        with open(os.path.join(ifp, model_dirpath, example_data_fn, tokenized_fn), 'w') as tok_ex_fh:
                            for idx, token in enumerate(tokens):
                                token_label = token_labels[idx]
                                token_label_id = token_label_ids[idx]
                                mask = label_mask[idx]
                                tok_ex_fh.write('{}\t{}\t{}\t{}\n'.format(token, token_label, token_label_id, mask))
                        
                        numpy_logits = example_logits.cpu().detach().numpy()
                        logit_str = '{}'.format(numpy_logits[0])
                        for l_idx in range(1, numpy_logits.size):
                            logit_str += ',{}'.format(numpy_logits[l_idx])
                        logit_fh.write('{}, {}\n'.format(fn, logit_str))
                        
            if total_examples_count == 0:
                overall_accuracy = 0.0
            else:
                overall_accuracy = 100.0 * float(total_examples_correct) / float(total_examples_count)

            # print('  accuracy = {}'.format(overall_accuracy))
            with open(os.path.join(ifp, model_dirpath, accuracy_result_fn), 'w') as fh:
                fh.write('{}'.format(overall_accuracy))

            if overall_accuracy < 100.0:
                return (True, model_dirpath)

        return (False, model_dirpath)
    except Exception as e:
        print('Model: {} threw exception'.format(model_dirpath))
        traceback.print_exc()
        return (False, model_dirpath)

dataset_dict = {}

# Preload all datasets without any poisoning
for embedding in round_config.RoundConfig.EMBEDDING_LEVELS:
    flavor = round_config.RoundConfig.EMBEDDING_FLAVOR_LEVELS[embedding][0]
    save_flavor = flavor.replace('/', '-')

    tokenizer_filepath = os.path.join(tokienizer_fp, '{}-{}.pt'.format(embedding, save_flavor))
    tokenizer = torch.load(tokenizer_filepath)

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_dict[embedding] = {}

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for dataset_name in round_config.RoundConfig.SOURCE_DATASET_LEVELS:
        print('Preloading {} with {}'.format(dataset_name, embedding))

        rso = np.random.RandomState()

        nerDataset = dataset.NerDataset('all_data.txt', rso, thread_count=3)
        nerDataset.load_dataset_no_config(datasets_fp, dataset_name)
        nerDataset.build_dataset_no_poisoning(datasets_fp, dataset_name, embedding, flavor, tokenizer, -100, False)
        dataset_dict[embedding][dataset_name] = nerDataset

for clean_data_flag in [True, False]:
    if clean_data_flag:
        print('Generating Clean Example Images')
        accuracy_result_fn = 'clean-example-accuracy.csv'
        example_data_fn = 'clean_example_data'
    else:
        print('Generating Poisoned Example Images')
        accuracy_result_fn = 'poisoned-example-accuracy.csv'
        example_data_fn = 'poisoned_example_data'

    fail_list = list()
    for m_idx in range(len(models)):
        # print('{}/{} models'.format(m_idx, len(models)))
        model_dirpath = models[m_idx]
        fail, model_dirpath = worker(model_dirpath, clean_data_flag, accuracy_result_fn, example_data_fn, dataset_dict)
        print('Finished {}'.format(model_dirpath))

        if fail: fail_list.append(model_dirpath)

    if len(fail_list) > 0:
        print('The following models failed to have the required accuracy:')
        for model_dirpath in fail_list:
            print(model_dirpath)
