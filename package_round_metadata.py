import os
import json
import numpy as np
import pandas as pd

import round_config

convergence_f1_threshold = float(0.8)
triggered_convergence_f1_threshold = float(0.95)
include_example_data_in_convergence_keys = True

convergence_f1_threshold_qa_distil = float(0.68)
convergence_f1_threshold_qa_electra = float(0.73)
triggered_convergence_f1_threshold_sc = float(0.9)
include_non_convergence_reason = False


def package_metadata(ifp, include_example=include_example_data_in_convergence_keys):
    ofp = os.path.join(ifp, 'METADATA.csv')
    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    df_list = list()
    print("Loading config keys")

    # Find a triggered config file
    for i in range(len(fns)):
        try:
            with open(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME)) as json_file:
                config_dict = json.load(json_file)
        except:
            print('missing model config for : {}'.format(fns[i]))
            continue

        try:
            with open(os.path.join(ifp, fns[i], 'stats.json')) as json_file:
                stats_dict = json.load(json_file)
        except:
            print('missing model stats for : {}'.format(fns[i]))
            continue

        del config_dict['py/object']
        del config_dict['output_filepath']
        del config_dict['output_ground_truth_filename']
        del config_dict['num_outputs']

        config_dict['model_name'] = fns[i]
        config_dict['converged'] = False

        if config_dict['task_type'] == 'ner':
            to_delete = set()
            to_add = dict()
            for key in stats_dict.keys():
                if '_overall_' in key and 'example' not in key:
                    if 'f1' in key:
                        new_key = key.replace("_overall", "")
                        to_add[new_key] = stats_dict[key]
                    to_delete.add(key)
                else:
                    if key.endswith('_f1') and 'example' not in key:
                        to_delete.add(key)
            for k in to_delete:
                del stats_dict[k]
            for k in to_add:
                stats_dict[k] = to_add[k]

        if config_dict['task_type'] == 'qa':
            to_delete = set()
            for key in stats_dict.keys():
                if '_HasAns_' in key:
                    to_delete.add(key)
                if '_NoAns_' in key:
                    to_delete.add(key)
                if '_best_' in key:
                    to_delete.add(key)
                if '_total' in key:
                    to_delete.add(key)
                if '_exact' in key:
                    to_delete.add(key)

            for k in to_delete:
                del stats_dict[k]

        if 'trigger' in config_dict.keys() and config_dict['trigger'] is not None:
            del config_dict['trigger']['py/object']
            sub_dict = config_dict['trigger']

            if 'trigger_executor' in sub_dict.keys() and sub_dict['trigger_executor'] is not None:
                val = config_dict['trigger']['trigger_executor']['py/object']
                del config_dict['trigger']['trigger_executor']['py/object']
                del config_dict['trigger']['trigger_executor']['trigger_config']
                config_dict['trigger']['trigger_executor']['type'] = val

                sub_dict = sub_dict['trigger_executor']
                to_remove_keys = ['source_class_label', 'target_class_label', 'source_search_label', 'spurious_class', 'spurious_search_label', 'label_to_id_map', 'insert_min_location_percentage', 'insert_max_location_percentage', 'insert_spurious_min_location_percentage', 'insert_spurious_max_location_percentage', 'answer_location_perc_start', 'is_target_class', 'target_class']
                for key in to_remove_keys:
                    if key in sub_dict.keys():
                        del sub_dict[key]

        for key in stats_dict.keys():
            config_dict[key] = stats_dict[key]

        config_dict["nonconverged_reason"] = ""

        cd = pd.json_normalize(config_dict)
        df_list.append(cd)

    full_df = pd.concat(df_list, axis=0)
    cols = list(full_df.columns.values)
    if 'model_name' in cols:
        cols.remove('model_name')
    if 'converged' in cols:
        cols.remove('converged')
    if 'trigger' in cols:
        # remove trigger col as its data was in subfields
        cols.remove('trigger')
    if 'nonconverged_reason' in cols:
        cols.remove('nonconverged_reason')
    cols.insert(0, 'model_name')
    cols.insert(1, 'converged')
    cols.insert(2, 'nonconverged_reason')
    if not include_non_convergence_reason:
        cols.remove('nonconverged_reason')

    full_df = full_df[cols]

    number_poisoned = 0
    number_clean = 0
    number_poisoned_converged = 0
    number_clean_converged = 0

    if include_example:
        clean_convergence_keys = ['val_clean_f1', 'test_clean_f1', 'example_clean_f1']
        poisoned_convergence_keys = ['val_clean_f1', 'test_clean_f1', 'example_clean_f1', 'val_poisoned_f1', 'test_poisoned_f1', 'example_poisoned_f1']
    else:
        clean_convergence_keys = ['val_clean_f1', 'test_clean_f1']
        poisoned_convergence_keys = ['val_clean_f1', 'test_clean_f1', 'val_poisoned_f1', 'test_poisoned_f1']

    print("Thresholding convgence keys")
    model_names = full_df['model_name']
    for fn in model_names:
        row = full_df[full_df['model_name'] == fn]

        if row['poisoned'][0]:
            number_poisoned += 1
            convergence_keys = poisoned_convergence_keys
        else:
            number_clean += 1
            convergence_keys = clean_convergence_keys

        converged_dict = dict()
        for key in convergence_keys:
            converged_dict[key] = None

        for key in convergence_keys:
            converged_dict[key] = row[key][0]

        f1_thres_clean = convergence_f1_threshold
        f1_thres_poison = triggered_convergence_f1_threshold

        if row['task_type'][0] == "sc":
            f1_thres_poison = triggered_convergence_f1_threshold_sc
        if row['task_type'][0] == "qa" and row['model_architecture'][0] == 'distilbert-base-cased':
            f1_thres_clean = convergence_f1_threshold_qa_distil
        if row['task_type'][0] == "qa" and row['model_architecture'][0] == 'google/electra-small-discriminator':
            f1_thres_clean = convergence_f1_threshold_qa_electra
        
            
        converged = True
        non_converged_reasons = []
        for k in converged_dict.keys():
            if converged_dict[k] is None:
                print('Missing convergence metric "{}" for "{}"'.format(k, fn))
                converged = False
                non_converged_reasons.append("Missing {}".format(k))
            else:
                if 'poisoned' in k:
                    if converged_dict[k] < f1_thres_poison:
                        converged = False
                        non_converged_reasons.append("{}".format(k))
                else:
                    if converged_dict[k] < f1_thres_clean:
                        converged = False
                        non_converged_reasons.append("{}".format(k))

        col_idx = int(np.where(full_df.columns == 'converged')[0])
        row_idx = int(np.where(full_df['model_name'] == fn)[0])
        full_df.iat[row_idx, col_idx] = converged
        if converged:
            if row['poisoned'][0]:
                number_poisoned_converged += 1
            else:
                number_clean_converged += 1

        non_converged_reasons = ":".join(non_converged_reasons)
        if include_non_convergence_reason:
            col_idx = int(np.where(full_df.columns == 'nonconverged_reason')[0])
            full_df.iat[row_idx, col_idx] = non_converged_reasons


    full_df.to_csv(ofp, index=False)

    print('Found {} clean models.'.format(number_clean))
    print('Found {} poisoned models.'.format(number_poisoned))
    print('Found {} converged clean models.'.format(number_clean_converged))
    print('Found {} converged poisoned models.'.format(number_poisoned_converged))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    args = parser.parse_args()

    package_metadata(args.dir)
