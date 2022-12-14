import os
import numpy as np
import pandas as pd
import shutil
import random

import package_round_metadata


def convert_dict_to_key(d):
    # ensure the keys are in canonical order
    keys = sort_keys(d.keys())
    ret_list = list()
    for key in keys:
        lcl_str = get_key(key, d[key])
        # lcl_str = "{}={}".format(key, d[key])
        ret_list.append(lcl_str)
    ret_str = "&".join(ret_list)
    return ret_str


def convert_key_to_dict(key):
    ret_dict = {}
    key_split = key.split('&')
    for entry in key_split:
        entry_split = entry.split('=')
        ret_key = entry_split[0]
        ret_val = entry_split[1]
        if 'True' in ret_val or 'False' in ret_val:
            ret_val = ret_val == 'True'
        ret_dict[ret_key] = ret_val
    return ret_dict


def get_key(name, value):
    if name == 'poisoned':
        return 'poisoned={}'.format(value)
    if name == 'model':
        return 'model={}'.format(value)
    if name == 'dataset':
        return 'dataset={}'.format(value)
    if name == 'triggertype':
        return 'triggertype={}'.format(value)
    if name == 'triggerexecutor':
        return 'triggerexecutor={}'.format(value)
    raise RuntimeError("Invalid key name: {}".format(name))


def sort_keys(keys):
    a = list()
    if 'dataset' in keys:
        a.append('dataset')
    if 'model' in keys:
        a.append('model')
    if 'poisoned' in keys:
        a.append('poisoned')
    if 'triggertype' in keys:
        a.append('triggertype')
    if 'triggerexecutor' in keys:
        a.append('triggerexecutor')
    return a


def get_key_from_params(poisoned_flag=None, dataset=None, model_arch=None, trigger_type=None, trigger_executor=None):
    # key = 'dataset={}&model={}&poisoned={}'.format(dataset, model_arch, poisoned_flag)
    a = list()
    if dataset is not None:
        a.append(get_key('dataset', dataset))
    if model_arch is not None:
        a.append(get_key('model', model_arch))
    if poisoned_flag is not None:
        a.append(get_key('poisoned', poisoned_flag))

    if poisoned_flag:
        # key += '&triggertype={}&triggerexecutor={}'.format(trigger_type, trigger_executor)
        if trigger_type is not None:
            a.append(get_key('triggertype', trigger_type))
        if trigger_executor is not None:
            a.append(get_key('triggerexecutor', trigger_executor))

    if len(a) == 0:
        raise RuntimeError("All keys cannot be None")

    key = "&".join(a)

    return key


def get_key_from_df(df):
    dataset = df['source_dataset'].to_numpy()[0]
    trigger_type = df['trigger.trigger_executor.type'].to_numpy()[0]
    trigger_executor = df['trigger.trigger_executor_option'].to_numpy()[0]
    model_arch = df['model_architecture'].to_numpy()[0]
    poisoned_flag = df['poisoned'].to_numpy()[0]
    return get_key_from_params(poisoned_flag, dataset, model_arch, trigger_type, trigger_executor)


def get_DEX_configurations():
    i1_choices = [True, False]  # poisoned Y/N
    i2_choices = ['qa:squad_v2', 'ner:conll2003', 'sc:imdb']
    i3_choices = ['roberta-base', 'google/electra-small-discriminator', 'distilbert-base-cased']
    i4_choices = ['WordTriggerExecutor', 'PhraseTriggerExecutor']
    i5_choices = ['qa:context_normal_empty', 'qa:context_normal_trigger', 'qa:context_spatial_empty',
                  'qa:context_spatial_trigger', 'qa:question_normal_empty', 'qa:question_spatial_empty',
                  'qa:both_normal_empty', 'qa:both_normal_trigger', 'qa:both_spatial_empty', 'qa:both_spatial_trigger',
                  'ner:global', 'ner:local', 'ner:spatial_global',
                  'sc:normal', 'sc:spatial', 'sc:class', 'sc:spatial_class']

    i1_choices_set = set()
    i2_choices_set = set()
    i3_choices_set = set()
    i4_choices_set = set()
    i5_choices_set = set()

    # iterate over the dex configuration, building all possible combinations of factors
    search_models_dict = dict()
    selected_models_dict = dict()
    for i1 in i1_choices:
        for i2 in i2_choices:
            val = i2.split(':')[0].lower()
            task_type = val
            if val == 'qa':
                val = val.upper()
            else:
                val = val.title()
            prefix = 'trigger_executor.' + val
            for i3 in i3_choices:
                for i4 in i4_choices:
                    for i5 in i5_choices:
                        if i5.startswith(task_type):
                            # Create dictionaries
                            # get_key_from_params(poisoned_flag, dataset, model_arch, trigger_type, trigger_executor)
                            key = get_key_from_params(i1, i2, i3, prefix + i4, i5)

                            i1_choices_set.add(i1)
                            i2_choices_set.add(i2)
                            i3_choices_set.add(i3)
                            i4_choices_set.add(prefix + i4)
                            i5_choices_set.add(i5)

                            search_models_dict[key] = list()
                            selected_models_dict[key] = list()

    choices = dict()
    # These need to match the get_key() function, as that is how we translate between values and the match strings
    choices['poisoned'] = list(i1_choices_set)
    choices['dataset'] = list(i2_choices_set)
    choices['model'] = list(i3_choices_set)
    choices['triggertype'] = list(i4_choices_set)
    choices['triggerexecutor'] = list(i5_choices_set)

    return search_models_dict, selected_models_dict, choices


def convert_total_N_into_per_config(N, search_models_dict):
    # Parse clean vs poisoned to calculate sampling (this currently assumes 50% poisoned)
    # TODO: Update to parameterize poisoning percentage
    poisoned_count = 0
    clean_count = 0

    for key in search_models_dict.keys():
        key_dict = convert_key_to_dict(key)
        if key_dict['poisoned']:
            poisoned_count += 1
        else:
            clean_count += 1

    n_clean_samples = 1.0
    n_poisoned_samples = 1.0

    # 50% poisoned, we need same number of poisoned and clean
    if poisoned_count > clean_count:
        n_clean_samples = float(poisoned_count) / float(clean_count)
    else:
        n_poisoned_samples = float(clean_count) / float(poisoned_count)

    print('clean_count = {}'.format(clean_count))
    print('poisoned_count = {}'.format(poisoned_count))
    print('n_clean_samples = {}'.format(n_clean_samples))
    print('n_poisoned_samples = {}'.format(n_poisoned_samples))

    if not n_clean_samples.is_integer() and not n_poisoned_samples.is_integer():
        print('n_clean_samples = {}'.format(n_clean_samples))
        print('n_poisoned_samples = {}'.format(n_poisoned_samples))
        print("this is an unhandled case")
        raise RuntimeError("Invalid config model counts")

    if not n_clean_samples.is_integer():
        n_clean_samples = np.ceil(n_clean_samples)

    if not n_poisoned_samples.is_integer():
        n_poisoned_samples = np.ceil(n_poisoned_samples)

    print('final clean_count = {}'.format(clean_count))
    print('final poisoned_count = {}'.format(poisoned_count))
    print("final n_clean_samples = {}".format(n_clean_samples))
    print("final n_poisoned_samples = {}".format(n_poisoned_samples))

    # TODO: Handle non-divisible
    assert n_clean_samples.is_integer() and n_poisoned_samples.is_integer()

    minimum_num_models = n_clean_samples * clean_count + n_poisoned_samples * poisoned_count

    if N < minimum_num_models:
        print('Minimum number of models with 50% poisoning is {}, N was set to {}, UPDATING N to {}'.format(minimum_num_models, N, minimum_num_models))
        N = minimum_num_models

    if N % minimum_num_models != 0:
        old_N = N
        N = (N + minimum_num_models) - (N % minimum_num_models)
        print('N({}) is not divisible by the minimum number of models({}), updating to {}'.format(old_N, minimum_num_models, N))

    if minimum_num_models != N:
        multiplier = N / minimum_num_models
        assert multiplier.is_integer()
        n_clean_samples *= multiplier
        n_poisoned_samples *= multiplier

    n_clean_samples = int(n_clean_samples)
    n_poisoned_samples = int(n_poisoned_samples)

    print('Total combinations = {}'.format(len(search_models_dict)))
    print('effective clean N = {}'.format(n_clean_samples))
    print('effective poisoned N = {}'.format(n_poisoned_samples))
    print('Effective N = {}'.format(min(n_poisoned_samples, n_clean_samples)))
    return n_clean_samples, n_poisoned_samples


def find_missing_configurations(n_clean_samples, n_poisoned_samples, metadata_df, search_models_dict, selected_models_dict, converged_models):
    # Search through new models
    for model in converged_models:
        # model_path = os.path.join(ifp, model)
        df = metadata_df[metadata_df['model_name'] == model]
        key = get_key_from_df(df)
        if key not in search_models_dict:
            print('Skipped {} due to not searching for {}'.format(model, key))
            continue
        search_models_dict[key].append(model)

    # Sample enough from our search to populate our selected
    for key in sorted(selected_models_dict.keys()):
        key_dict = convert_key_to_dict(key)
        if key_dict['poisoned']:
            n_samples = n_poisoned_samples
        else:
            n_samples = n_clean_samples

        num_models_for_key = len(selected_models_dict[key])

        # Check if we have enough
        if num_models_for_key > n_samples:
            continue

        # See if we have any models that satisfy the key
        if len(search_models_dict[key]) > 0:
            # randomly select as many as we can
            num_remaining = n_samples - num_models_for_key

            num_to_select = min(num_remaining, len(search_models_dict[key]))

            samples = random.sample(search_models_dict[key], num_to_select)

            selected_models_dict[key].extend(samples)

    missing_count = 0
    selected_count = 0

    missing_config_configurations = []

    for key in sorted(selected_models_dict.keys()):
        key_dict = convert_key_to_dict(key)
        if key_dict['poisoned']:
            n_samples = n_poisoned_samples
        else:
            n_samples = n_clean_samples

        remaining = n_samples - len(selected_models_dict[key])
        missing_count += remaining
        selected_count += len(selected_models_dict[key])
        if remaining == n_samples:
            print('{} has ALL remaining ({})'.format(key, remaining))
            missing_config_configurations.append(convert_key_to_dict(key))
            pass
        elif remaining > 0:
            print('{} has {} remaining'.format(key, remaining))
            missing_config_configurations.append(convert_key_to_dict(key))
            pass
        else:
            # print('{} has {} remaining'.format(key, remaining))
            pass

    return missing_config_configurations, missing_count, selected_count


if __name__ == "__main__":
    # **************************************
    # ********** Input Parameters **********
    # **************************************
    ifp = '/scratch/trojai/data/round9/models-new'
    ofp = '/scratch/trojai/data/round9/round9-holdout-dataset/models'


    do_move = True  # controls whether to move models between folders
    do_in_place_numbering = True
    # Number of models to package
    N = 210

    # **************************************
    # ********** Input Parameters **********
    # **************************************

    print('building metadata for source models')
    package_round_metadata.package_metadata(ifp)

    if not os.path.exists(ofp):
        os.makedirs(ofp)

    existing_global_results_csv = None
    existing_metadata_df = None
    existing_models = None
    try:
        print('building metadata for model target')
        package_round_metadata.package_metadata(ofp)
        existing_global_results_csv = os.path.join(ofp, 'METADATA.csv')

        existing_metadata_df = pd.read_csv(existing_global_results_csv)
        existing_models = existing_metadata_df['model_name'].to_list()
    except Exception as e:
        print(e)
        print('Failed to load existing metadata')
        pass

    global_results_csv = os.path.join(ifp, 'METADATA.csv')
    metadata_df = pd.read_csv(global_results_csv)
    all_models = metadata_df['model_name']
    converged = metadata_df['converged']
    poisoned = metadata_df['poisoned']

    models = list()
    for i in range(len(all_models)):
        if converged[i]:
            models.append(all_models[i])

    print('Found {} converged models in source directory'.format(len(models)))

    # shuffle the models so I can pick from then sequentially based on the first to match criteria
    random.shuffle(models)

    found_nb_existing = 0
    nb_added = 0

    search_models_dict, selected_models_dict, choices = get_DEX_configurations()

    n_clean_samples, n_poisoned_samples = convert_total_N_into_per_config(N, search_models_dict)

    if existing_global_results_csv is not None:
        for model in existing_models:
            model_path = os.path.join(ofp, model)
            df = existing_metadata_df[existing_metadata_df['model_name'] == model]
            if df['converged'].to_numpy()[0]:
                key = get_key_from_df(df)
                selected_models_dict[key].append(model_path)
            else:
                raise RuntimeError('Existing model in {}/{} had not converged!'.format(ofp, model))

        print('Found {} matching models in the output directory'.format(len(existing_models)))

    missing_config_configurations, missing_count, selected_count = find_missing_configurations(n_clean_samples, n_poisoned_samples, metadata_df, search_models_dict, selected_models_dict, models)

    if len(missing_config_configurations) > 0:
        print('missing configurations: [')
        for configuration in missing_config_configurations:
            print("{},".format(configuration))
        print(']')

    # only copy if we have all of the required models
    if missing_count == 0 and do_move:
        print('Found all required models, moving them to the output directory')
        # keep the output numerically ordered
        fns = [fn for fn in os.listdir(ofp) if fn.startswith('id-')]
        fns.sort()
        max_id = -1
        for fn in fns:
            nb = int(fn[3:])
            if nb > max_id:
                max_id = nb

        models_to_copy = list()
        for key in sorted(selected_models_dict.keys()):
            if len(selected_models_dict[key]) == 0:
                raise RuntimeError('Programming Logic Problem. This should never happen.')
            models_to_copy.extend(selected_models_dict[key])

        random.shuffle(models_to_copy)

        for model in models_to_copy:
            if do_in_place_numbering:
                # find next non existing model
                i = 0
                while True:
                    new_fn = 'id-{:08}'.format(i)
                    if not os.path.exists(os.path.join(ofp, new_fn)):
                        break
                    i += 1
            else:
                max_id += 1
                new_fn = 'id-{:08}'.format(max_id)
            src = os.path.join(ifp, model)
            dest = os.path.join(ofp, new_fn)
            shutil.move(src, dest)

    print('Found {} models'.format(selected_count))
    print('Still missing {} models'.format(missing_count))





