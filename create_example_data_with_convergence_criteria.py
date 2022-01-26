import os
import numpy as np
import traceback
import json
import multiprocessing
import torch
import transformers
import datasets

import round_config
import dataset
import utils_qa

ACCURACY_THRESHOLD = 100.0
OVERWRITE_FLAG = True


def subset_based_on_f1(model, device, dataset, metric, threshold):
    dataset.set_pytorch_dataformat()
    dataloader = torch.utils.data.DataLoader(dataset.tokenized_dataset, batch_size=1)

    all_preds = None
    with torch.no_grad():
        for i, tensor_dict in enumerate(dataloader):

            input_ids = tensor_dict['input_ids'].to(device)
            attention_mask = tensor_dict['attention_mask'].to(device)
            token_type_ids = tensor_dict['token_type_ids'].to(device)
            start_positions = tensor_dict['start_positions'].to(device)
            end_positions = tensor_dict['end_positions'].to(device)

            if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
                model_output_dict = model(input_ids,
                                          attention_mask=attention_mask,
                                          start_positions=start_positions,
                                          end_positions=end_positions)
            else:
                model_output_dict = model(input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          start_positions=start_positions,
                                          end_positions=end_positions)

            start_logits = model_output_dict['start_logits'].detach().cpu().numpy()
            end_logits = model_output_dict['end_logits'].detach().cpu().numpy()

            logits = (start_logits, end_logits)
            all_preds = logits if all_preds is None else transformers.trainer_pt_utils.nested_concat(all_preds, logits, padding_index=-100)

    # ensure correct columns are being yielded to the postprocess
    dataset.reset_dataformat()

    predictions = utils_qa.postprocess_qa_predictions(dataset.dataset, dataset.tokenized_dataset, all_preds, version_2_with_negative=True)
    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
    ]
    references = [{"id": ex["id"], "answers": ex['answers']} for ex in dataset.dataset]

    subset = list()
    for i in range(len(formatted_predictions)):
        record = dataset.dataset[i]
        pred = [formatted_predictions[i]]
        ref = [references[i]]
        metrics = metric.compute(predictions=pred, references=ref)
        if metrics['f1'] >= threshold:
            record['f1_score'] = metrics['f1']
            record['exact_score'] = metrics['exact']
            subset.append(record)

    return subset


# dataset_dict used to cache the datasets to speed up processing; key = embedding+dataset_name
def worker(model_fn, clean_data_flag, create_n_examples, source_datasets_filepath):
    try:
        config = round_config.RoundConfig.load_json(os.path.join(ifp, model_fn, round_config.RoundConfig.CONFIG_FILENAME))
        if clean_data_flag:
            stats_key = 'example_clean_f1_score'
            ofn = 'clean-example-data.json'
        else:
            stats_key = 'example_poisoned_f1_score'
            ofn = 'poisoned-example-data.json'

        # get current example accuracy
        with open(os.path.join(ifp, model_fn, 'stats.json')) as json_file:
            stats = json.load(json_file)

        if not clean_data_flag and not config.poisoned:
            # skip generating poisoned examples for a clean model

            # update the stats file to include a null value for all example stats
            stats[stats_key] = None
            with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
                json.dump(stats, fh, ensure_ascii=True, indent=2)

            return 0, model_fn

        example_accuracy = 0
        if stats_key in stats.keys():
            example_accuracy = stats[stats_key]

        # update the stats file to show the example accuracy
        stats[stats_key] = example_accuracy
        with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
            json.dump(stats, fh, ensure_ascii=True, indent=2)

        # exit now if the accuracy requirement is already met
        if not OVERWRITE_FLAG and example_accuracy >= ACCURACY_THRESHOLD:
            return 0, model_fn

        rso = np.random.RandomState()

        dataset_json_filepath = os.path.join(source_datasets_filepath, config.source_dataset + '.json')
        full_dataset = dataset.QaDataset(dataset_json_filepath=dataset_json_filepath,
                                         random_state_obj=rso,
                                         thread_count=1)

        full_dataset.dataset = full_dataset.dataset.shuffle(keep_in_memory=True)
        create_n_examples_to_select_from = int(create_n_examples * 20)
        full_dataset.dataset = full_dataset.dataset.select(range(0, create_n_examples_to_select_from), keep_in_memory=True)

        if clean_data_flag:
            # turn off poisoning
            if config.trigger is not None:
                config.trigger.fraction = 0.0
        else:
            # poison all the data
            config.trigger.fraction = 1.0
            try:
                full_dataset.trojan(config)
            except AssertionError:
                # if this throws an error its due to not being able to reach the trigger percentage, which we can ignore
                pass

        full_dataset_clean, full_dataset_poisoned = full_dataset.clean_poisoned_split()

        if clean_data_flag:
            full_dataset = full_dataset_clean
        else:
            full_dataset = full_dataset_poisoned

        # load the classification model and move it to the GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(os.path.join(ifp, model_fn, 'model.pt'), map_location=torch.device(device))
        model = model.to(device)
        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_architecture, use_fast=True)
        full_dataset.tokenize(tokenizer)

        metric = datasets.load_metric('squad_v2')

        # find N examples
        subset_list = subset_based_on_f1(model, device, full_dataset, metric, ACCURACY_THRESHOLD)
        if len(subset_list) == 0:
            return 1, model_fn

        examples = list()
        no_ans_count = 0.0
        while len(examples) < create_n_examples:
            if len(subset_list) == 0:
                print('ran out of example data')
                return 1, model_fn

            if clean_data_flag and no_ans_count > 0 and (no_ans_count / len(examples)) > 0.5:
                # pick only from the examples with an answer
                example = None
                for i in range(len(subset_list)):
                    if len(subset_list[i]['answers']['text']) > 0:
                        example = subset_list[i]
                        del subset_list[i]
                        break
            else:
                i = np.random.randint(len(subset_list))
                example = subset_list[i]
                del subset_list[i]
            if example is None:
                print('ran out of example data')
                return 1, model_fn

            if len(example['answers']['text']) == 0:
                no_ans_count += 1
            examples.append(example)

        avg_f1 = 0.0
        for example in examples:
            avg_f1 += example['f1_score']
        avg_f1 = avg_f1 / float(len(examples))

        examples = {'data': examples}  # make the examples compatible with the huggingface dataset loader
        # dataset = datasets.load_dataset('json', data_files=[fp], field='data', keep_in_memory=True, split='train')
        examples_folder = os.path.join(ifp, model_fn, 'example_data')
        if not os.path.exists(examples_folder):
            os.makedirs(examples_folder)
        with open(os.path.join(examples_folder, ofn), 'w') as fh:
            json.dump(examples, fh, ensure_ascii=True, indent=2)

        # update the stats file to show the example accuracy
        stats[stats_key] = avg_f1
        with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
            json.dump(stats, fh, ensure_ascii=True, indent=2)

        return 0, model_fn
    except:
        tb = traceback.format_exc()
        print(tb)
        return 3, model_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    parser.add_argument('--source-datasets-filepath', type=str, required=True, help='Filepath to the folder/directory where the results should be stored')
    parser.add_argument('-n', type=int, default=20, help='Number of example data points to create.')
    args = parser.parse_args()

    ifp = args.dir
    source_datasets_filepath = args.source_datasets_filepath
    create_n_examples = args.n

    models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    models.sort()

    for clean_data_flag in [True, False]:
        fail_list = list()

        for m_idx in range(len(models)):
            model_fn = models[m_idx]
            if clean_data_flag:
                print('model: {} clean is {}/{}'.format(model_fn, m_idx, len(models)))
            else:
                print('model: {} poisoned is {}/{}'.format(model_fn, m_idx, len(models)))
            ec, _ = worker(model_fn, clean_data_flag, create_n_examples, source_datasets_filepath)

            if ec != 0:
                fail_list.append(model_fn)

        if len(fail_list) > 0:
            print('The following models failed to have the required accuracy:')
            for model_fn in fail_list:
                print(model_fn)


