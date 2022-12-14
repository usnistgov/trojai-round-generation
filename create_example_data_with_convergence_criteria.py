import os
import numpy as np
import traceback
import json
import datetime
import torch
import transformers
import random
import pandas as pd

import round_config
import main
import train_model
import package_round_metadata

F1_THRESHOLD = 1.0
OVERWRITE_FLAG = False


def subset_based_on_f1(model, device, dataset, metric, data_collator, threshold, task_type):

    dataset.set_pytorch_dataformat()
    dataloader = torch.utils.data.DataLoader(dataset.tokenized_dataset, batch_size=1, collate_fn=data_collator)

    all_preds = None
    all_labels = None

    with torch.no_grad():
        for i, tensor_dict in enumerate(dataloader):

            tensor_dict = train_model.prepare_inputs(tensor_dict, device)
            if 'distilbert' in model.name_or_path:
                if 'token_type_ids' in tensor_dict.keys():
                    del tensor_dict['token_type_ids']
            labels = None

            # TODO handle this more elegantly, as there are unhandled failure cases
            if 'labels' in tensor_dict.keys():
                labels = tensor_dict['labels']
                #del tensor_dict['labels']
            model_output_dict = model(**tensor_dict)

            logits = tuple(v for k, v in model_output_dict.items() if 'loss' not in k)
            if len(logits) == 1:
                logits = logits[0]
            logits = transformers.trainer_pt_utils.nested_detach(logits)

            if labels is not None:
                labels = transformers.trainer_pt_utils.nested_detach(labels)
                all_labels = labels if all_labels is None else transformers.trainer_pt_utils.nested_concat(all_labels, labels, padding_index=-100)

            all_preds = logits if all_preds is None else transformers.trainer_pt_utils.nested_concat(all_preds, logits, padding_index=-100)

    all_preds = transformers.trainer_pt_utils.nested_numpify(all_preds)

    if all_labels is not None:
        all_labels = transformers.trainer_pt_utils.nested_numpify(all_labels)

    # ensure correct columns are being yielded to the postprocess
    dataset.reset_dataformat()

    predictions, references = dataset.post_process_predictions(all_preds, all_labels)

    subset = list()
    for i in range(len(predictions)):
        record = dataset.dataset[i]
        pred = [predictions[i]]
        ref = [references[i]]

        if task_type == 'sc':
            # handle SC binary classification F1 weirdness
            metrics = {'f1': float(pred[0] == ref[0])}
        else:
            metrics = metric.compute(predictions=pred, references=ref)

        # handle different metrics names for different tasks
        if 'overall_f1' in metrics.keys():
            if metrics['overall_f1'] >= threshold:
                record['f1'] = metrics['overall_f1']
                subset.append(record)
        elif 'f1' in metrics.keys():
            if metrics['f1'] >= threshold:
                record['f1'] = metrics['f1']
                subset.append(record)
        else:
            raise RuntimeError("Metric missing")

        if 'f1' in record and task_type == 'qa':
            # convert to [0, 1] from [0, 100]
            record['f1'] = record['f1'] / 100.0

    return subset


# dataset_dict used to cache the datasets to speed up processing; key = embedding+dataset_name
def worker(model_fn, clean_data_flag, create_n_examples, source_datasets_filepath, only_new, tokenizers_folder):
    try:
        config = round_config.RoundConfig.load_json(os.path.join(ifp, model_fn, round_config.RoundConfig.CONFIG_FILENAME))
        if clean_data_flag:
            stats_key = 'example_clean_f1'
            ofn = 'clean-example-data.json'
        else:
            stats_key = 'example_poisoned_f1'
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

        if not os.path.exists(os.path.join(ifp, model_fn, ofn)):
            # missing output example file, resetting example accuracy to none
            stats[stats_key] = 0

        if only_new:
            if stats_key in stats.keys() and stats[stats_key] is not None and stats[stats_key] > 0:
                # skip models which already have an accuracy computed
                return 0, model_fn

        example_accuracy = 0
        if stats_key in stats.keys():
            example_accuracy = stats[stats_key]

        # update the stats file to show the example accuracy
        stats[stats_key] = example_accuracy
        with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
            json.dump(stats, fh, ensure_ascii=True, indent=2)

        # exit now if the accuracy requirement is already met
        if not OVERWRITE_FLAG and example_accuracy >= F1_THRESHOLD:
            return 0, model_fn

        preset_configuration = None
        # ignore the initial model from setup
        # example_data_flag = True prevents the trigger object from being rebuilt from scratch. Instead we use the trigger object loaded from the json
        tokenizer_fn = config.model_architecture.replace('/', '-') + ".pt"
        tokenizer_filepath = os.path.join(tokenizers_folder, tokenizer_fn)
        dataset, tokenizer, _, data_collator, metric, config = main.setup_training(config, source_datasets_filepath, preset_configuration,example_data_flag=True, tokenizer_filepath=tokenizer_filepath)

        dataset.dataset = dataset.dataset.shuffle(keep_in_memory=True)
        create_n_examples_to_select_from = int(create_n_examples * 20)
        dataset.dataset = dataset.dataset.select(range(0, create_n_examples_to_select_from), keep_in_memory=True)

        if clean_data_flag:
            # turn off poisoning
            if config.trigger is not None:
                config.trigger.fraction = 0.0
        else:
            # poison all the data
            config.trigger.fraction = 1.0
            try:
                dataset.trojan(config)
            except:
                # if this throws an error its due to not being able to reach the trigger percentage, which we can ignore
                pass

        dataset_clean, dataset_poisoned = dataset.clean_poisoned_split()

        if clean_data_flag:
            dataset = dataset_clean
        else:
            dataset = dataset_poisoned

        dataset.tokenize(tokenizer)

        # load the model and move it to the GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(os.path.join(ifp, model_fn, 'model.pt'), map_location=torch.device(device))
        model = model.to(device)
        model.eval()

        # find N examples
        subset_list = subset_based_on_f1(model, device, dataset, metric, data_collator, F1_THRESHOLD, config.task_type)
        if len(subset_list) == 0:
            return 1, model_fn

        examples = list()
        if config.task_type == 'qa':
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
                    print('Task: QA. Ran out of example data')
                    return 1, model_fn

                if len(example['answers']['text']) == 0:
                    no_ans_count += 1
                examples.append(example)
        elif config.task_type == 'ner':
            if not clean_data_flag:
                class_ids_to_build_examples_for = list()
                class_name = 'B-' + config.trigger.trigger_executor.target_class_label
                class_nb = config.trigger.trigger_executor.label_to_id_map[class_name]
                class_ids_to_build_examples_for.append(class_nb)
            else:
                valid_classes = ['LOC','PER','MISC','ORG']
                class_ids_to_build_examples_for = list()
                for c in valid_classes:
                    class_name = 'B-' + c
                    class_nb = config.label_to_id_map[class_name]
                    class_ids_to_build_examples_for.append(class_nb)

            for class_nb in class_ids_to_build_examples_for:
                per_class_examples = list()
                while len(per_class_examples) < create_n_examples:
                    if len(subset_list) == 0:
                        print('ran out of example data for class {}'.format(class_nb))
                        return 1, model_fn

                    idx_list = list(range(0, len(subset_list)))
                    random.shuffle(idx_list)
                    example = None
                    for ii in range(len(idx_list)):
                        i = idx_list[ii]
                        val = subset_list[i]

                        if class_nb in val['ner_tags']:
                            example = val
                            break

                    if example is None:
                        print('Task: {}. Ran out of example data for class {}'.format(config.task_type, class_nb))
                        return 1, model_fn

                    del subset_list[i]

                    per_class_examples.append(example)
                examples.extend(per_class_examples)

        elif config.task_type == "sc":
            nb_classes = config.num_outputs
            class_ids_to_build_examples_for = list(range(0, nb_classes))
            if not clean_data_flag:
                if not config.trigger.trigger_executor.is_target_class:
                    # this trigger applies to all classes
                    class_ids_to_build_examples_for = list(range(0, nb_classes))
                else:
                    class_ids_to_build_examples_for = list()
                    # for SC the 'target_class' is the class the trigger is applied to... :facepalm:
                    class_ids_to_build_examples_for.append(1 - config.trigger.trigger_executor.target_class)

            for class_nb in class_ids_to_build_examples_for:
                per_class_examples = list()
                while len(per_class_examples) < create_n_examples:
                    if len(subset_list) == 0:
                        print('ran out of example data for class {}'.format(class_nb))
                        return 1, model_fn

                    idx_list = list(range(0, len(subset_list)))
                    random.shuffle(idx_list)
                    example = None
                    for ii in range(len(idx_list)):
                        i = idx_list[ii]
                        val = subset_list[i]

                        if class_nb == val['label']:
                            example = val
                            break

                    if example is None:
                        print('Task: {}. Ran out of example data for class {}'.format(config.task_type, class_nb))
                        return 1, model_fn

                    del subset_list[i]

                    per_class_examples.append(example)
                examples.extend(per_class_examples)
        else:
            raise RuntimeError("Invalid task type: {}".format(config.task_type))

        avg_f1 = 0.0
        for example in examples:
            avg_f1 += example['f1']
        avg_f1 = avg_f1 / float(len(examples))

        examples = {'data': examples}  # make the examples compatible with the huggingface dataset loader
        # dataset = datasets.load_dataset('json', data_files=[fp], field='data', keep_in_memory=True, split='train')
        with open(os.path.join(ifp, model_fn, ofn), 'w') as fh:
            json.dump(examples, fh, ensure_ascii=True, indent=2)

        # update the stats file to show the example accuracy
        stats[stats_key] = avg_f1
        with open(os.path.join(ifp, model_fn, 'stats.json'), 'w') as fh:
            json.dump(stats, fh, ensure_ascii=True, indent=2)

        return 0, model_fn
    except Exception:
        print('Model: {} threw exception'.format(model_fn))
        traceback.print_exc()
        return 2, model_fn


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    parser.add_argument('-n', type=int, default=20, help='Number of example data points to create.')
    parser.add_argument('--source_datasets_folder', type=str, required=True, help='folder where the source json files are stored')
    parser.add_argument('--tokenizers_folder', type=str, required=True, help='folder where the tokenizers are stored')
    parser.add_argument('--only_new', action='store_true', help='whether to only build example data for models without any existing examples')
    parser.add_argument('--parallel', action='store_true', help='whether to use multi-processing')
    parser.add_argument('--only_converged', action='store_true', help='whether to only build example data only for models otherwise converged')
    args = parser.parse_args()

    ifp = args.dir
    create_n_examples = args.n
    only_new = args.only_new
    only_converged = args.only_converged
    parallel = args.parallel
    source_datasets_folder = args.source_datasets_folder
    tokenizers_folder = args.tokenizers_folder

    models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    models.sort()

    if os.path.exists("good_models.txt"):
        os.remove("good_models.txt")
    if os.path.exists("inaccurate_models.txt"):
        os.remove("inaccurate_models.txt")
    if os.path.exists("error_models.txt"):
        os.remove("error_models.txt")
    with open("example_data_progress.txt", 'w') as fh:
        fh.write("{} > Building Example Data\n".format(datetime.datetime.now().isoformat()))

    if only_converged:
        print("Building metadata to determine which models are converged without example data.")
        package_round_metadata.package_metadata(ifp, include_example=False)
        # only build example data for models which are converged without the example data

        global_results_csv = os.path.join(ifp, 'METADATA.csv')
        metadata_df = pd.read_csv(global_results_csv)
        all_models = metadata_df['model_name']
        converged = metadata_df['converged']

        converged_models = list()
        for i in range(len(all_models)):
            model = all_models[i]
            if converged[i]:
                converged_models.append(all_models[i])

        converged_models.sort()
        models = converged_models

    succeeded_models = list()
    inacurate_models = list()
    error_models = list()

    if parallel:
        import multiprocessing
        nb_threads = min(8, multiprocessing.cpu_count())

        st_idx = 0
        block_size = 20*nb_threads
        while st_idx < len(models):
            end_idx = min(st_idx+block_size, len(models))
            models_subset = models[st_idx: end_idx]

            args_list = list()
            for clean_data_flag in [True, False]:
                fail_list = list()
                for m_idx in range(len(models_subset)):
                    model_fn = models_subset[m_idx]
                    args_list.append((model_fn, clean_data_flag, create_n_examples, source_datasets_folder, only_new, tokenizers_folder))

            with multiprocessing.Pool(processes=nb_threads) as pool:
                error_codes = pool.starmap(worker, args_list)

            for v in error_codes:
                ec, fn = v
                if ec == 0:
                    succeeded_models.append(fn)
                if ec == 1:
                    inacurate_models.append(fn)
                if ec == 2:
                    error_models.append(fn)
            
            with open("example_data_progress.txt", 'a') as fh:
                fh.write("{} > Completed block from {} to {} ({}%)\n".format(datetime.datetime.now().isoformat(), st_idx, end_idx, 100*float(end_idx)/float(len(models))))
            st_idx += block_size

    else:
        error_codes = list()
        for clean_data_flag in [True, False]:
            for m_idx in range(len(models)):
                print('{}/{} models'.format(m_idx, len(models)))
                model_fn = models[m_idx]
                ec, _ = worker(model_fn, clean_data_flag, create_n_examples, source_datasets_folder, only_new, tokenizers_folder)
                print('Finished {}'.format(model_fn))

                if ec != 0:
                    error_codes.append((ec, model_fn))

        for (ec, fn) in error_codes:
            if ec == 0:
                succeeded_models.append(fn)
            if ec == 1:
                inacurate_models.append(fn)
            if ec == 2:
                error_models.append(fn)

    succeeded_models = list(set(succeeded_models))
    succeeded_models.sort()
    inacurate_models = list(set(inacurate_models))
    inacurate_models.sort()
    error_models = list(set(error_models))
    error_models.sort()


    with open("good_models.txt", 'a') as fh:
        for fn in succeeded_models:
            fh.write("{}\n".format(fn))
    with open("inacurate_models.txt", 'a') as fh:
        for fn in inacurate_models:
            fh.write("{}\n".format(fn))
    with open("error_models.txt", 'a') as fh:
        for fn in error_models:
            fh.write("{}\n".format(fn))


