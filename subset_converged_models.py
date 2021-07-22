# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import pandas as pd
import shutil
import random

import package_round_metadata
import trigger_config


def convert_key_to_dict(key):
    ret_dict = {}
    key_split = key.split(':')
    for entry in key_split:
        entry_split = entry.split('=')
        ret_key = entry_split[0]
        ret_val = entry_split[1]
        if 'True' in ret_val or 'False' in ret_val:
            ret_val = ret_val == 'True'
        ret_dict[ret_key] = ret_val
    return ret_dict


def get_key_from_params(poisoned_flag, dataset, global_flag, embedding, executor):
    key = 'dataset={}:embedding={}:poisoned={}'.format(dataset, embedding, poisoned_flag)
    
    if poisoned_flag:
        key += ':global={}:executor={}'.format(global_flag, executor)
    
    return key


def get_key_from_df(df):
    dataset = df['source_dataset'].to_numpy()[0]
    global_trigger = df['triggers_0_global_trigger'].to_numpy()[0]
    trigger_executor_name = df['triggers_0_trigger_executor_name'].to_numpy()[0]
    embedding = df['embedding'].to_numpy()[0]
    poisoned = df['poisoned'].to_numpy()[0]
    return get_key_from_params(poisoned, dataset, global_trigger, embedding, trigger_executor_name)


# **************************************
# ********** Input Parameters **********
# **************************************

ifp = '/wrk/tjb3/data/round7-final/models-new'
ofp = '/wrk/tjb3/data/round7-final/deployed-round7-train-dataset/models'

do_move = False  # controls whether to move models between folders
do_in_place_numbering = True
# Number of models to package
N = 384

# **************************************
# ********** Input Parameters **********
# **************************************


print('building metadata for model source')
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

missing_count = 0
found_nb_existing = 0
nb_added = 0

configs = list()
i1_choices = [True, False]  # poisoned Y/N
i2_choices = ['bbn-pcet', 'ontonotes-5.0', 'conll2003'] # dataset
i3_choices = [True, False] # trigger global levels
i4_choices = ['BERT', 'DistilBERT', 'RoBERTa', 'MobileBERT']  # embedding type
i5_choices = ['character', 'word1', 'phrase', 'word2'] # trigger executor

# iterate over the dex configuration, building all possible combinations of factors
search_models_dict = dict()
selected_models_dict = dict()
for i1 in i1_choices:
    for i2 in i2_choices:
        for i3 in i3_choices:
            for i4 in i4_choices:
                for i5 in i5_choices:
                    # Create dictionaries
                    key = get_key_from_params(i1, i2, i3, i4, i5)
                    search_models_dict[key] = list()
                    selected_models_dict[key] = list()

# Parse clean vs poisoned to calculate sampling (this currently assumes 50% poisoned)
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

# Handle non-divisible
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
print('poisoned_samples = {}; clean_samples = {}'.format(n_poisoned_samples, n_clean_samples))
print('Effective N = {}'.format(n_poisoned_samples + n_clean_samples))

# Search through already selected and populate selected keys
if existing_global_results_csv is not None:
    for model in existing_models:
        model_path = os.path.join(ofp, model)
        df = existing_metadata_df[existing_metadata_df['model_name'] == model]
        if df['converged'].to_numpy()[0]:
            key = get_key_from_df(df)
            selected_models_dict[key].append(model_path)
        else:
            print('Existing model in {}/{} had not converged!'.format(ofp, model))

    print('Found {} matching models in the output directory'.format(len(existing_models)))


# Search through new models
for model in models:
    model_path = os.path.join(ifp, model)
    df = metadata_df[metadata_df['model_name'] == model]
    key = get_key_from_df(df)
    if key not in search_models_dict:
        print('Skipped {} due to not searching for {}'.format(model, key))
        continue
    search_models_dict[key].append(model_path)

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
        

# Print some stats:
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
        print('{} has {} remaining'.format(key, remaining))
        pass


if len(missing_config_configurations) > 0:
    
    print('missing configurations: [')
    executor_dict = {v: k for k, v in trigger_config.TriggerConfig.TRIGGER_MAPPING.items()}
    for configuration in missing_config_configurations:
        if 'executor' in configuration:
            configuration['executor'] = executor_dict[configuration['executor']]
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
        dest = os.path.join(ofp, new_fn)
        shutil.move(model, dest)

print('Found {} models'.format(selected_count))
print('Still missing {} models'.format(missing_count))





